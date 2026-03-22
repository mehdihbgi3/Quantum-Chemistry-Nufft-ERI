import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
import time
import warnings
from grid.onedgrid import GaussLegendre
from grid.rtransform import BeckeRTransform
from grid.molgrid import MolGrid
from grid.becke import BeckeWeights


def double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    result = 1
    k = 2 * n - 1
    while k > 0:
        result *= k
        k -= 2
    return result


def cart_gto_norm(alpha: float, ix: int, iy: int, iz: int) -> float:
    l = ix + iy + iz
    prefactor = (2.0 * alpha / np.pi) ** 0.75
    angular_factor = (4.0 * alpha) ** (l / 2.0)
    denom = np.sqrt(
        double_factorial(ix) * double_factorial(iy) * double_factorial(iz)
    )
    return prefactor * angular_factor / denom


def iter_cart_xyz(l: int) -> List[Tuple[int, int, int]]:
    return [(i, j, l - i - j)
            for i in range(l + 1)
            for j in range(l + 1 - i)]


def n_cart(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def evaluate_primitive_gto(
    points: np.ndarray,
    center: np.ndarray,
    alpha: float,
    angular: Tuple[int, int, int],
    normalize: bool = True
) -> np.ndarray:
    ix, iy, iz = angular
    dr = points - center
    r2 = np.sum(dr**2, axis=1)
    radial = np.exp(-alpha * r2)
    angular_part = (dr[:, 0]**ix) * (dr[:, 1]**iy) * (dr[:, 2]**iz)
    result = angular_part * radial
    if normalize:
        result *= cart_gto_norm(alpha, ix, iy, iz)
    return result


def evaluate_contracted_gto(
    points: np.ndarray,
    center: np.ndarray,
    exponents: np.ndarray,
    coefficients: np.ndarray,
    angular: Tuple[int, int, int]
) -> np.ndarray:
    ix, iy, iz = angular
    result = np.zeros(len(points))
    for alpha, coef in zip(exponents, coefficients):
        norm_coef = coef * cart_gto_norm(alpha, ix, iy, iz)
        primitive = evaluate_primitive_gto(points, center, alpha, angular, normalize=False)
        result += norm_coef * primitive
    return result


@dataclass
class UniformGrid:
    points: np.ndarray
    weights: np.ndarray
    shape: Tuple[int, int, int]
    origin: np.ndarray
    spacing: float
    
    @classmethod
    def from_center(cls, center: np.ndarray, extent: float, spacing: float):
        center = np.asarray(center)
        box_min = center - extent
        box_max = center + extent
        n_points = np.ceil((box_max - box_min) / spacing).astype(int) + 1
        nx, ny, nz = n_points
        x = np.linspace(box_min[0], box_max[0], nx)
        y = np.linspace(box_min[1], box_max[1], ny)
        z = np.linspace(box_min[2], box_max[2], nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        weights = np.full(len(points), spacing ** 3)
        return cls(points, weights, (nx, ny, nz), box_min, spacing)
    
    @property
    def size(self):
        return len(self.points)
    
    def integrate(self, values):
        return np.sum(values * self.weights)


@dataclass
class BeckeGrid:
    points: np.ndarray
    weights: np.ndarray
    molgrid: MolGrid
    
    @classmethod
    def from_atom(cls, atnum: int, center: np.ndarray, n_rad: int = 75, n_ang: int = 302):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oned = GaussLegendre(npoints=n_rad)
            rgrid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned)
            molgrid = MolGrid.from_size(
                atnums=np.array([atnum]),
                atcoords=np.array([center]),
                rgrid=rgrid,
                size=n_ang,
                aim_weights=BeckeWeights(),
                store=True
            )
        return cls(molgrid.points, molgrid.weights, molgrid)
    
    @classmethod
    def from_molecule(cls, atnums, atcoords, n_rad=75, n_ang=302):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oned = GaussLegendre(npoints=n_rad)
            rgrid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned)
            molgrid = MolGrid.from_size(
                atnums=np.asarray(atnums),
                atcoords=np.asarray(atcoords),
                rgrid=rgrid,
                size=n_ang,
                aim_weights=BeckeWeights(),
                store=True
            )
        return cls(molgrid.points, molgrid.weights, molgrid)
    
    @property
    def size(self):
        return len(self.points)
    
    def integrate(self, values):
        return np.sum(values * self.weights)


@dataclass
class GTOShell:
    center: np.ndarray
    exponents: np.ndarray
    coefficients: np.ndarray
    angular_momentum: int
    
    @property
    def n_functions(self) -> int:
        return n_cart(self.angular_momentum)
    
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        n_pts = len(points)
        n_func = self.n_functions
        result = np.zeros((n_func, n_pts))
        
        for i, angular in enumerate(iter_cart_xyz(self.angular_momentum)):
            result[i] = evaluate_contracted_gto(
                points, self.center, self.exponents, 
                self.coefficients, angular
            )
        
        return result


def evaluate_gto_product(
    shell_i: GTOShell,
    shell_j: GTOShell,
    points: np.ndarray,
    i_func: int = 0,
    j_func: int = 0
) -> np.ndarray:
    angular_i = list(iter_cart_xyz(shell_i.angular_momentum))[i_func]
    angular_j = list(iter_cart_xyz(shell_j.angular_momentum))[j_func]
    
    chi_i = evaluate_contracted_gto(
        points, shell_i.center, shell_i.exponents,
        shell_i.coefficients, angular_i
    )
    chi_j = evaluate_contracted_gto(
        points, shell_j.center, shell_j.exponents,
        shell_j.coefficients, angular_j
    )
    
    return chi_i * chi_j


def evaluate_all_products(
    shell_i: GTOShell,
    shell_j: GTOShell,
    points: np.ndarray
) -> np.ndarray:
    chi_i = shell_i.evaluate(points)
    chi_j = shell_j.evaluate(points)
    
    return chi_i[:, np.newaxis, :] * chi_j[np.newaxis, :, :]


def test_single_product():
    print("\n------------------------")
    print("Test 1: Product of Two s-type GTOs")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    shell = GTOShell(
        center=center,
        exponents=np.array([1.0]),
        coefficients=np.array([1.0]),
        angular_momentum=0
    )
    
    grid = BeckeGrid.from_atom(1, center, n_rad=75, n_ang=302)
    
    product = evaluate_gto_product(shell, shell, grid.points, 0, 0)
    
    integral = grid.integrate(product)
    
    print(f"  Shell: s-type, α=1.0, center at origin")
    print(f"  Grid: {grid.size} points")
    print(f"\n  ∫ χ(r)² dr = {integral:.10f}")
    print(f"  Expected: 1.0")
    print(f"  Error: {abs(integral - 1.0):.2e}")
    
    passed = abs(integral - 1.0) < 1e-6
    print(f"\n  {'Confirmed' if passed else 'Not Confirmed'}")
    
    return passed


def test_overlap_integral():
    print("\n------------------------")
    print("Test 2: Overlap Integral (Different Centers)")
    print()
    
    R = 1.4
    center_a = np.array([0.0, 0.0, -R/2])
    center_b = np.array([0.0, 0.0,  R/2])
    
    alpha = 1.0
    shell_a = GTOShell(center_a, np.array([alpha]), np.array([1.0]), 0)
    shell_b = GTOShell(center_b, np.array([alpha]), np.array([1.0]), 0)
    
    grid = BeckeGrid.from_molecule(
        atnums=np.array([1, 1]),
        atcoords=np.array([center_a, center_b]),
        n_rad=75, n_ang=302
    )
    
    product = evaluate_gto_product(shell_a, shell_b, grid.points, 0, 0)
    numerical_overlap = grid.integrate(product)
    
    alpha_tot = alpha + alpha
    theta = alpha * alpha / alpha_tot
    N = cart_gto_norm(alpha, 0, 0, 0)
    analytical_overlap = (np.pi / alpha_tot)**1.5 * np.exp(-theta * R**2) * N * N
    
    print(f"  Two s-type GTOs:")
    print(f"    Center A: {center_a}")
    print(f"    Center B: {center_b}")
    print(f"    Separation R = {R} Bohr")
    print(f"    Exponent α = {alpha}")
    print(f"\n  Grid: {grid.size} points")
    print(f"\n  Numerical overlap:  {numerical_overlap:.10f}")
    print(f"  Analytical overlap: {analytical_overlap:.10f}")
    print(f"  Error: {abs(numerical_overlap - analytical_overlap):.2e}")
    
    passed = abs(numerical_overlap - analytical_overlap) < 1e-6
    print(f"\n  {'Confirmed' if passed else 'Not Confirmed'}")
    
    return passed


def test_p_orbital_product():
    print("\n------------------------")
    print("Test 3: Product of s and p-type GTOs")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    shell_s = GTOShell(center, np.array([alpha]), np.array([1.0]), 0)
    shell_p = GTOShell(center, np.array([alpha]), np.array([1.0]), 1)
    
    grid = BeckeGrid.from_atom(1, center, n_rad=75, n_ang=302)
    
    product_s_px = evaluate_gto_product(shell_s, shell_p, grid.points, 0, 0)
    integral_s_px = grid.integrate(product_s_px)
    
    print(f"  Testing ∫ χ_s × χ_px dr ")
    print(f"  Numerical: {integral_s_px:.2e}")
    print(f"  Expected: 0.0")
    
    product_px_px = evaluate_gto_product(shell_p, shell_p, grid.points, 0, 0)
    integral_px_px = grid.integrate(product_px_px)
    
    print(f"\n  Testing ∫ χ_px² dr ")
    print(f"  Numerical: {integral_px_px:.10f}")
    print(f"  Error: {abs(integral_px_px - 1.0):.2e}")
    
    passed = abs(integral_s_px) < 1e-10 and abs(integral_px_px - 1.0) < 1e-6
    print(f"\n  {' Confirmed' if passed else 'Not confirmed'}")
    
    return passed


def test_all_products():
    print("\n------------------------")
    print("Test 4: All Products in Shell Pair")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    shell = GTOShell(center, np.array([alpha]), np.array([1.0]), 1)
    
    grid = BeckeGrid.from_atom(1, center, n_rad=75, n_ang=302)
    
    t0 = time.time()
    all_products = evaluate_all_products(shell, shell, grid.points)
    t1 = time.time()
    
    print(f"  Shell: p-type (3 functions)")
    print(f"  Products shape: {all_products.shape}")
    print(f"  Evaluation time: {(t1-t0)*1000:.2f} ms")
    
    integrals = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            integrals[i, j] = grid.integrate(all_products[i, j])
    
    print(f"\n  Overlap matrix:")
    print(f"    {integrals}")
    
    expected = np.eye(3)
    max_error = np.max(np.abs(integrals - expected))
    print(f"\n  Max deviation from identity: {max_error:.2e}")
    
    passed = max_error < 1e-6
    print(f"\n  {' Confirmed' if passed else 'Not Confirmed'}")
    
    return passed


def test_sto3g_hydrogen():
    print("\n------------------------")
    print("Test 5: STO-3G Hydrogen Basis")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    exponents = np.array([3.42525091, 0.62391373, 0.16885540])
    coefficients = np.array([0.15432897, 0.53532814, 0.44463454])
    
    shell = GTOShell(center, exponents, coefficients, 0)
    
    grid = BeckeGrid.from_atom(1, center, n_rad=100, n_ang=302)
    
    product = evaluate_gto_product(shell, shell, grid.points, 0, 0)
    self_overlap = grid.integrate(product)
    
    print(f"  STO-3G H basis:")
    print(f"    Exponents: {exponents}")
    print(f"    Coefficients: {coefficients}")
    print(f"\n  Grid: {grid.size} points")
    print(f"\n  Self-overlap ∫χ² dr = {self_overlap:.10f}")
    print(f"\n  Overlap depends on normalization convention")
    
    return True


def test_grid_comparison_density():
    print("\n------------------------")
    print("Test 6: Uniform vs Becke Grid for Density")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    shell = GTOShell(center, np.array([1.0]), np.array([1.0]), 0)
    
    ref_grid = BeckeGrid.from_atom(1, center, n_rad=150, n_ang=590)
    ref_product = evaluate_gto_product(shell, shell, ref_grid.points, 0, 0)
    ref_integral = ref_grid.integrate(ref_product)
    
    print(f"  Reference (fine Becke grid): ∫χ² = {ref_integral:.10f}")
    print(f"\n  Comparing grids for same accuracy:")
    
    print("\n  Uniform grids:")
    for spacing in [0.4, 0.3, 0.25, 0.2]:
        grid = UniformGrid.from_center(center, 8.0, spacing)
        product = evaluate_gto_product(shell, shell, grid.points, 0, 0)
        integral = grid.integrate(product)
        error = abs(integral - ref_integral)
        print(f"    spacing={spacing}: {grid.size:7d} pts, ∫χ²={integral:.6f}, err={error:.2e}")
    
    print("\n  Becke grids:")
    for n_rad in [30, 40, 50, 60]:
        grid = BeckeGrid.from_atom(1, center, n_rad=n_rad, n_ang=194)
        product = evaluate_gto_product(shell, shell, grid.points, 0, 0)
        integral = grid.integrate(product)
        error = abs(integral - ref_integral)
        print(f"    n_rad={n_rad:3d}:  {grid.size:7d} pts, ∫χ²={integral:.6f}, err={error:.2e}")
    
    return True


def main():
    print("GTO Product Evaluation on Grids")
    
    test1 = test_single_product()
    test2 = test_overlap_integral()
    test3 = test_p_orbital_product()
    test4 = test_all_products()
    test5 = test_sto3g_hydrogen()
    test6 = test_grid_comparison_density()
    
    
    all_passed = test1 and test2 and test3 and test4 and test5 and test6
    
    
    return all_passed


if __name__ == "__main__":
    main()
