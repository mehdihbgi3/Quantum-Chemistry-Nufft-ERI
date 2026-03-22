import numpy as np
from typing import Tuple
from dataclasses import dataclass
import time
import warnings
import finufft
from scipy.special import roots_legendre
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
    denom = np.sqrt(double_factorial(ix) * double_factorial(iy) * double_factorial(iz))
    return prefactor * angular_factor / denom


def evaluate_primitive_gto(points, center, alpha, angular, normalize=True):
    ix, iy, iz = angular
    dr = points - center
    r2 = np.sum(dr**2, axis=1)
    radial = np.exp(-alpha * r2)
    angular_part = (dr[:, 0]**ix) * (dr[:, 1]**iy) * (dr[:, 2]**iz)
    result = angular_part * radial
    if normalize:
        result *= cart_gto_norm(alpha, ix, iy, iz)
    return result


@dataclass
class UniformGrid:
    points: np.ndarray
    weights: np.ndarray
    shape: Tuple[int, int, int]
    origin: np.ndarray
    spacing: float
    
    @classmethod
    def from_center(cls, center, extent, spacing):
        center = np.asarray(center)
        box_min = center - extent
        box_max = center + extent
        n_points = np.ceil((box_max - box_min) / spacing).astype(int) + 1
        nx, ny, nz = int(n_points[0]), int(n_points[1]), int(n_points[2])
        x = np.linspace(box_min[0], box_max[0], nx)
        y = np.linspace(box_min[1], box_max[1], ny)
        z = np.linspace(box_min[2], box_max[2], nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        weights = np.full(len(points), spacing ** 3)
        return cls(points, weights, (nx, ny, nz), box_min.copy(), spacing)
    
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
    def from_atom(cls, atnum, center, n_rad=75, n_ang=302):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oned = GaussLegendre(npoints=n_rad)
            rgrid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned)
            molgrid = MolGrid.from_size(
                atnums=np.array([atnum]),
                atcoords=np.array([center]),
                rgrid=rgrid, size=n_ang,
                aim_weights=BeckeWeights(), store=True
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
                rgrid=rgrid, size=n_ang,
                aim_weights=BeckeWeights(), store=True
            )
        return cls(molgrid.points, molgrid.weights, molgrid)
    
    @property
    def size(self):
        return len(self.points)
    
    def integrate(self, values):
        return np.sum(values * self.weights)


def create_spherical_k_grid(n_r: int, n_theta: int, n_phi: int, 
                            k_max: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
   
    
    x_r, w_r = roots_legendre(n_r)
    r = 0.5 * k_max * (x_r + 1)
    w_r = 0.5 * k_max * w_r
    
    t_theta, w_theta = roots_legendre(n_theta)
    theta = np.pi * (t_theta + 1)
    w_theta = np.pi * w_theta
    
    t_phi, w_phi = roots_legendre(n_phi)
    
    n_total = n_r * n_theta * n_phi
    kv = np.zeros((n_total, 3))
    weights = np.zeros(n_total)
    
    idx = 0
    for i_r in range(n_r):
        for i_phi in range(n_phi):
            sin_phi = np.sqrt(1.0 - t_phi[i_phi]**2)
            cos_phi = t_phi[i_phi]
            for i_theta in range(n_theta):
                kv[idx, 0] = r[i_r] * sin_phi * np.cos(theta[i_theta])
                kv[idx, 1] = r[i_r] * sin_phi * np.sin(theta[i_theta])
                kv[idx, 2] = r[i_r] * cos_phi
                weights[idx] = r[i_r]**2 * w_r[i_r] * w_phi[i_phi] * w_theta[i_theta]
                idx += 1
    
    return kv, weights


def ft_nufft_type3(
    func_vals: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    k_vectors: np.ndarray,
    eps: float = 1e-9,
    threshold: float = 1e-14
) -> np.ndarray:
    c_full = func_vals * weights
    
    max_abs = np.max(np.abs(c_full))
    if max_abs > 0:
        mask = np.abs(c_full) > threshold * max_abs
    else:
        mask = np.ones(len(c_full), dtype=bool)
    
    c = c_full[mask].astype(np.complex128)
    x = points[mask, 0].astype(np.float64)
    y = points[mask, 1].astype(np.float64)
    z = points[mask, 2].astype(np.float64)
    
    s = k_vectors[:, 0].astype(np.float64)
    t = k_vectors[:, 1].astype(np.float64)
    u = k_vectors[:, 2].astype(np.float64)
    
    f_k = finufft.nufft3d3(x, y, z, c, s, t, u, eps=eps, isign=1)
    
    return f_k


def ft_direct_sum(
    func_vals: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    k_vectors: np.ndarray,
    threshold: float = 1e-14
) -> np.ndarray:
    c = func_vals * weights
    
    max_abs = np.max(np.abs(c))
    if max_abs > 0:
        mask = np.abs(c) > threshold * max_abs
    else:
        mask = np.ones(len(c), dtype=bool)
    
    c_filt = c[mask]
    pts_filt = points[mask]
    
    f_k = np.zeros(len(k_vectors), dtype=np.complex128)
    for i, k in enumerate(k_vectors):
        phases = np.exp(1j * (pts_filt @ k))
        f_k[i] = np.sum(c_filt * phases)
    
    return f_k


def ft_gaussian_analytical(k: np.ndarray, alpha: float, center: np.ndarray) -> np.ndarray:
    k2 = np.sum(k**2, axis=1)
    k_dot_R = k @ center
    return np.exp(-k2 / (4 * alpha)) * np.exp(1j * k_dot_R)


def ft_s_gto_analytical(k: np.ndarray, alpha: float, center: np.ndarray) -> np.ndarray:
    N = cart_gto_norm(alpha, 0, 0, 0)
    k2 = np.sum(k**2, axis=1)
    k_dot_R = k @ center
    prefactor = N * (np.pi / alpha) ** 1.5
    return prefactor * np.exp(-k2 / (4 * alpha)) * np.exp(1j * k_dot_R)


def test_spherical_k_grid():
    print("\n-------------------------------------")
    print("Test 1: Spherical k-space Grid")
    print()
    
    kv, weights = create_spherical_k_grid(10, 10, 10, k_max=10.0)
    
    print(f"  Grid: 10 × 10 × 10 = 1000 points")
    print(f"  |k| range: [{np.linalg.norm(kv, axis=1).min():.4f}, {np.linalg.norm(kv, axis=1).max():.4f}]")
    
    total_weight = np.sum(weights)
    expected = 4 * np.pi / 3 * 10.0**3
    error = abs(total_weight - expected) / expected
    print(f"\n  Total weight: {total_weight:.2f}")
    print(f"  Expected (4π/3 × k_max³): {expected:.2f}")
    print(f"  Relative error: {error:.2e}")
    
    passed = error < 0.01
    print(f"\n  {'Confirmed' if passed else 'Not Confirmed'}")
    return passed


def test_nufft_vs_direct():
    print("\n-------------------------------------")
    print("Test 2: NUFFT vs Direct Summation")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    grid = BeckeGrid.from_atom(1, center, n_rad=30, n_ang=110)
    
    alpha = 1.0
    r2 = np.sum(grid.points**2, axis=1)
    f = (alpha / np.pi)**1.5 * np.exp(-alpha * r2)
    
    kv, _ = create_spherical_k_grid(5, 5, 5, k_max=5.0)
    
    t0 = time.time()
    F_nufft = ft_nufft_type3(f, grid.points, grid.weights, kv, eps=1e-12)
    t_nufft = time.time() - t0
    
    t0 = time.time()
    F_direct = ft_direct_sum(f, grid.points, grid.weights, kv)
    t_direct = time.time() - t0
    
    error = np.abs(F_nufft - F_direct)
    max_error = np.max(error)
    
    print(f"  Grid points: {grid.size}")
    print(f"  k-points: {len(kv)}")
    print(f"  NUFFT time: {t_nufft*1000:.2f} ms")
    print(f"  Direct time: {t_direct*1000:.2f} ms")
    print(f"  Max difference: {max_error:.2e}")
    
    passed = max_error < 1e-10
    print(f"\n  {'Confirmed' if passed else 'Not Confirmed'}")
    return passed


def test_nufft_gaussian():
    print("\n-------------------------------------")
    print("Test 3: NUFFT on Normalized Gaussian")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    grid = BeckeGrid.from_atom(1, center, n_rad=100, n_ang=590)
    
    alpha = 1.0
    r2 = np.sum(grid.points**2, axis=1)
    f = (alpha / np.pi)**1.5 * np.exp(-alpha * r2)
    
    integral = grid.integrate(f)
    print(f"  ∫ f(r) dr = {integral:.10f} ")
    print(f"  Grid: {grid.size} points")
    
    kv, _ = create_spherical_k_grid(10, 10, 10, k_max=6.0)
    
    t0 = time.time()
    F_nufft = ft_nufft_type3(f, grid.points, grid.weights, kv, eps=1e-12)
    t_nufft = time.time() - t0
    
    k2 = np.sum(kv**2, axis=1)
    F_analytical = np.exp(-k2 / (4 * alpha))
    
    error = np.abs(F_nufft - F_analytical)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"  k-points: {len(kv)}, k_max = 6.0")
    print(f"  NUFFT time: {t_nufft*1000:.2f} ms")
    print(f"\n  Max error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    
    passed = max_error < 1e-5
    print(f"\n  {'Confirmed' if passed else 'Not Confirmed'} (threshold: 1e-5)")
    return passed


def test_nufft_s_gto():
    print("\n-------------------------------------")
    print("Test 4: NUFFT on s-type GTO")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    grid = BeckeGrid.from_atom(1, center, n_rad=100, n_ang=590)
    gto_vals = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
    
    norm = grid.integrate(gto_vals**2)
    print(f"  ∫|χ|² dr = {norm:.10f}")
    print(f"  Grid: {grid.size} points")
    
    kv, _ = create_spherical_k_grid(10, 10, 10, k_max=6.0)
    
    F_nufft = ft_nufft_type3(gto_vals, grid.points, grid.weights, kv, eps=1e-12)
    F_analytical = ft_s_gto_analytical(kv, alpha, center)
    
    error = np.abs(F_nufft - F_analytical)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"  k-points: {len(kv)}")
    print(f"\n  Max error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    
    passed = max_error < 1e-5
    print(f"\n  {'Confirmed' if passed else 'Not Confirmed'} (threshold: 1e-5)")
    return passed


def test_nufft_off_center():
    print("\n-------------------------------------")
    print("Test 5: NUFFT for Off-center GTO")
    print()
    
    center = np.array([1.0, 0.5, -0.3])
    alpha = 1.0
    
    grid = BeckeGrid.from_atom(1, center, n_rad=100, n_ang=590)
    gto_vals = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
    
    kv, _ = create_spherical_k_grid(10, 10, 10, k_max=6.0)
    
    F_nufft = ft_nufft_type3(gto_vals, grid.points, grid.weights, kv, eps=1e-12)
    F_analytical = ft_s_gto_analytical(kv, alpha, center)
    
    error = np.abs(F_nufft - F_analytical)
    max_error = np.max(error)
    
    print(f"  GTO center: {center}")
    print(f"  Grid: {grid.size} points")
    print(f"  Max error: {max_error:.2e}")
    
    idx = 500
    print(f"\n  Phase check at k={kv[idx]}:")
    print(f"    NUFFT: {np.angle(F_nufft[idx]):.6f}")
    print(f"    Analytical: {np.angle(F_analytical[idx]):.6f}")
    
    passed = max_error < 1e-5
    print(f"\n  {'Confirmed' if passed else 'Not Confirmed'}")
    return passed


def test_gto_product_ft():
    print("\n-------------------------------------")
    print("Test 6: Fourier Transform of GTO Product (Density)")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    grid = BeckeGrid.from_atom(1, center, n_rad=100, n_ang=590)
    
    gto = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
    density = gto * gto
    
    norm = grid.integrate(density)
    print(f"  ∫|χ|² dr = {norm:.10f}")
    
    kv, _ = create_spherical_k_grid(10, 10, 10, k_max=8.0)
    F_density = ft_nufft_type3(density, grid.points, grid.weights, kv, eps=1e-12)
    
    N = cart_gto_norm(alpha, 0, 0, 0)
    alpha_eff = 2 * alpha
    k2 = np.sum(kv**2, axis=1)
    F_expected = N**2 * (np.pi / alpha_eff)**1.5 * np.exp(-k2 / (4 * alpha_eff))
    
    error = np.abs(F_density - F_expected)
    max_error = np.max(error)
    
    print(f"  Grid: {grid.size} points")
    print(f"  k-points: {len(kv)}")
    print(f"\n  Max error: {max_error:.2e}")
    
    k0_idx = np.argmin(k2)
    F_k0 = F_density[k0_idx].real
    print(f"  F(k≈0) = {F_k0:.6f}")
    
    passed = max_error < 1e-5
    print(f"\n  {'Confirmed' if passed else 'Not Confirmed'}")
    return passed


def test_convergence_study():
    print("\n-------------------------------------")
    print("Test 7: Convergence Study")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    kv, _ = create_spherical_k_grid(8, 8, 8, k_max=5.0)
    F_analytical = ft_s_gto_analytical(kv, alpha, center)
    
    print(f"  Testing s-GTO FT accuracy vs grid size:")
    print(f"  {'n_rad':>6} {'n_ang':>6} {'Grid pts':>10} {'Max Error':>12} {'Time (ms)':>10}")
    print("  " + "-"*50)
    
    configs = [
        (40, 194),
        (60, 302),
        (80, 434),
        (100, 590),
        (120, 770),
    ]
    
    for n_rad, n_ang in configs:
        grid = BeckeGrid.from_atom(1, center, n_rad=n_rad, n_ang=n_ang)
        gto = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
        
        t0 = time.time()
        F_nufft = ft_nufft_type3(gto, grid.points, grid.weights, kv, eps=1e-12)
        t_elapsed = (time.time() - t0) * 1000
        
        max_error = np.max(np.abs(F_nufft - F_analytical))
        print(f"  {n_rad:>6} {n_ang:>6} {grid.size:>10} {max_error:>12.2e} {t_elapsed:>10.2f}")
    
    return True


def main():
    print("Fourier Transform Implementation (FFT vs NUFFT)")
    
    test1 = test_spherical_k_grid()
    test2 = test_nufft_vs_direct()
    test3 = test_nufft_gaussian()
    test4 = test_nufft_s_gto()
    test5 = test_nufft_off_center()
    test6 = test_gto_product_ft()
    test7 = test_convergence_study()
    
    
    all_passed = test1 and test2 and test3 and test4 and test5 and test6 and test7
    

    
    return all_passed


if __name__ == "__main__":
    main()
