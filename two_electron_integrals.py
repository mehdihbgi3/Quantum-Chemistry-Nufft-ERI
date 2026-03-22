import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
import time
import warnings
import finufft
from scipy.special import roots_legendre, erf

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


def iter_cart_xyz(l):
    return [(i, j, l - i - j) for i in range(l + 1) for j in range(l + 1 - i)]


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


def evaluate_contracted_gto(points, center, exponents, coefficients, angular):
    ix, iy, iz = angular
    result = np.zeros(len(points))
    for alpha, coef in zip(exponents, coefficients):
        norm_coef = coef * cart_gto_norm(alpha, ix, iy, iz)
        primitive = evaluate_primitive_gto(points, center, alpha, angular, normalize=False)
        result += norm_coef * primitive
    return result


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


def ft_nufft_type3(func_vals, points, weights, k_vectors, eps=1e-9, threshold=1e-14):
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
    
    return finufft.nufft3d3(x, y, z, c, s, t, u, eps=eps, isign=1)


def compute_eri_fourier(
    density_bra: np.ndarray,
    density_ket: np.ndarray,
    grid_bra: BeckeGrid,
    grid_ket: BeckeGrid,
    kv: np.ndarray,
    k_weights: np.ndarray,
    eps: float = 1e-9
) -> float:
    F_bra = ft_nufft_type3(density_bra, grid_bra.points, grid_bra.weights, kv, eps=eps)
    
    F_ket = ft_nufft_type3(density_ket, grid_ket.points, grid_ket.weights, kv, eps=eps)
    F_ket_neg = np.conj(F_ket)
    
    k2 = np.sum(kv**2, axis=1)
    
    k2_safe = np.where(k2 > 1e-20, k2, 1e-20)
    coulomb_kernel = 1.0 / k2_safe
    
    integrand = F_bra * F_ket_neg * coulomb_kernel
    eri = np.sum(k_weights * integrand.real) / (2 * np.pi**2)
    
    return eri


def compute_eri_direct(
    density_bra: np.ndarray,
    density_ket: np.ndarray,
    grid_bra: BeckeGrid,
    grid_ket: BeckeGrid
) -> float:
    eri = 0.0
    
    for i in range(grid_bra.size):
        r1 = grid_bra.points[i]
        w1 = grid_bra.weights[i]
        rho1 = density_bra[i]
        
        if abs(rho1 * w1) < 1e-15:
            continue
        
        r12 = np.linalg.norm(grid_ket.points - r1, axis=1)
        
        r12_safe = np.where(r12 > 1e-10, r12, 1e-10)
        
        coulomb = 1.0 / r12_safe
        
        eri += w1 * rho1 * np.sum(grid_ket.weights * density_ket * coulomb)
    
    return eri


def eri_ssss_analytical(alpha_a, alpha_b, alpha_c, alpha_d, 
                        Ra, Rb, Rc, Rd):
    zeta = alpha_a + alpha_b
    eta = alpha_c + alpha_d
    
    P = (alpha_a * Ra + alpha_b * Rb) / zeta
    Q = (alpha_c * Rc + alpha_d * Rd) / eta
    
    PQ = np.linalg.norm(P - Q)
    
    rho = zeta * eta / (zeta + eta)
    
    Rab = np.linalg.norm(Ra - Rb)
    Rcd = np.linalg.norm(Rc - Rd)
    
    Kab = np.exp(-alpha_a * alpha_b / zeta * Rab**2)
    Kcd = np.exp(-alpha_c * alpha_d / eta * Rcd**2)
    
    Na = cart_gto_norm(alpha_a, 0, 0, 0)
    Nb = cart_gto_norm(alpha_b, 0, 0, 0)
    Nc = cart_gto_norm(alpha_c, 0, 0, 0)
    Nd = cart_gto_norm(alpha_d, 0, 0, 0)
    
    T = rho * PQ**2
    
    if T < 1e-10:
        F0 = 1.0
    else:
        F0 = 0.5 * np.sqrt(np.pi / T) * erf(np.sqrt(T))
    
    prefactor = 2 * np.pi**(2.5) / (zeta * eta * np.sqrt(zeta + eta))
    eri = prefactor * Kab * Kcd * F0 * Na * Nb * Nc * Nd
    
    return eri


def test_eri_same_center():
    print("\n-----------------------------")
    print("Test 1: (ss|ss) Integral - Same Center")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    grid = BeckeGrid.from_atom(1, center, n_rad=80, n_ang=434)
    
    gto = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
    density = gto * gto
    
    kv, k_weights = create_spherical_k_grid(15, 15, 15, k_max=10.0)
    
    t0 = time.time()
    eri_fourier = compute_eri_fourier(density, density, grid, grid, kv, k_weights)
    t_fourier = time.time() - t0
    
    eri_analytical = eri_ssss_analytical(alpha, alpha, alpha, alpha,
                                         center, center, center, center)
    
    error = abs(eri_fourier - eri_analytical)
    rel_error = error / abs(eri_analytical)
    
    print(f"  All GTOs: s-type, α={alpha}, center at origin")
    print(f"  Real-space grid: {grid.size} points")
    print(f"  k-space grid: {len(kv)} points")
    print(f"\n  Fourier ERI: {eri_fourier:.10f}")
    print(f"  Analytical:  {eri_analytical:.10f}")
    print(f"  Abs error: {error:.2e}")
    print(f"  Rel error: {rel_error:.2e}")
    print(f"  Time: {t_fourier*1000:.1f} ms")
    
    passed = rel_error < 1e-4
    return passed


def test_eri_two_centers():
    print("\n-----------------------------")
    print("Test 2: (ss|ss) Integral - Two Centers")
    print()
    
    R = 1.4
    Ra = np.array([0.0, 0.0, -R/2])
    Rb = np.array([0.0, 0.0,  R/2])
    alpha = 1.0
    
    grid = BeckeGrid.from_molecule(
        atnums=np.array([1, 1]),
        atcoords=np.array([Ra, Rb]),
        n_rad=80, n_ang=434
    )
    
    gto_a = evaluate_primitive_gto(grid.points, Ra, alpha, (0, 0, 0))
    gto_b = evaluate_primitive_gto(grid.points, Rb, alpha, (0, 0, 0))
    
    density_aa = gto_a * gto_a
    density_bb = gto_b * gto_b
    density_ab = gto_a * gto_b
    
    kv, k_weights = create_spherical_k_grid(15, 15, 15, k_max=10.0)
    
    eri_aaaa_f = compute_eri_fourier(density_aa, density_aa, grid, grid, kv, k_weights)
    eri_aaaa_a = eri_ssss_analytical(alpha, alpha, alpha, alpha, Ra, Ra, Ra, Ra)
    
    eri_aabb_f = compute_eri_fourier(density_aa, density_bb, grid, grid, kv, k_weights)
    eri_aabb_a = eri_ssss_analytical(alpha, alpha, alpha, alpha, Ra, Ra, Rb, Rb)
    
    eri_abab_f = compute_eri_fourier(density_ab, density_ab, grid, grid, kv, k_weights)
    eri_abab_a = eri_ssss_analytical(alpha, alpha, alpha, alpha, Ra, Rb, Ra, Rb)
    
    print(f"  Centers: Ra={Ra}, Rb={Rb}")
    print(f"  Grid: {grid.size} points")
    
    print(f"\n  (aa|aa):")
    print(f"    Fourier:    {eri_aaaa_f:.8f}")
    print(f"    Analytical: {eri_aaaa_a:.8f}")
    print(f"    Rel error:  {abs(eri_aaaa_f - eri_aaaa_a)/abs(eri_aaaa_a):.2e}")
    
    print(f"\n  (aa|bb) [Coulomb type]:")
    print(f"    Fourier:    {eri_aabb_f:.8f}")
    print(f"    Analytical: {eri_aabb_a:.8f}")
    print(f"    Rel error:  {abs(eri_aabb_f - eri_aabb_a)/abs(eri_aabb_a):.2e}")
    
    print(f"\n  (ab|ab) [Exchange type]:")
    print(f"    Fourier:    {eri_abab_f:.8f}")
    print(f"    Analytical: {eri_abab_a:.8f}")
    print(f"    Rel error:  {abs(eri_abab_f - eri_abab_a)/abs(eri_abab_a):.2e}")
    
    errors = [
        abs(eri_aaaa_f - eri_aaaa_a)/abs(eri_aaaa_a),
        abs(eri_aabb_f - eri_aabb_a)/abs(eri_aabb_a),
        abs(eri_abab_f - eri_abab_a)/abs(eri_abab_a),
    ]
    passed = all(e < 1e-3 for e in errors)
    return passed


def test_eri_direct_comparison():
    print("\n-----------------------------")
    print("Test 3: Fourier vs Direct Integration")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    grid = BeckeGrid.from_atom(1, center, n_rad=40, n_ang=194)
    
    gto = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
    density = gto * gto
    
    kv, k_weights = create_spherical_k_grid(12, 12, 12, k_max=8.0)
    
    t0 = time.time()
    eri_fourier = compute_eri_fourier(density, density, grid, grid, kv, k_weights)
    t_fourier = time.time() - t0
    
    print(f"  Computing direct integral ")
    t0 = time.time()
    eri_direct = compute_eri_direct(density, density, grid, grid)
    t_direct = time.time() - t0
    
    eri_analytical = eri_ssss_analytical(alpha, alpha, alpha, alpha,
                                         center, center, center, center)
    
    print(f"\n  Grid: {grid.size} points")
    print(f"\n  Results:")
    print(f"    Fourier:    {eri_fourier:.8f}  (time: {t_fourier*1000:.1f} ms)")
    print(f"    Direct:     {eri_direct:.8f}  (time: {t_direct*1000:.1f} ms)")
    print(f"    Analytical: {eri_analytical:.8f}")
    
    print(f"\n  Errors vs analytical:")
    print(f"    Fourier: {abs(eri_fourier - eri_analytical)/abs(eri_analytical):.2e}")
    print(f"    Direct:  {abs(eri_direct - eri_analytical)/abs(eri_analytical):.2e}")
    
    print(f"\n  Speedup (Direct/Fourier): {t_direct/t_fourier:.1f}x")
    
    passed = abs(eri_fourier - eri_analytical)/abs(eri_analytical) < 1e-3
    return passed


def test_eri_different_exponents():
    print("\n-----------------------------")
    print("Test 4: ERIs with Different Exponents")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    
    grid = BeckeGrid.from_atom(1, center, n_rad=80, n_ang=434)
    kv, k_weights = create_spherical_k_grid(15, 15, 15, k_max=12.0)
    
    test_cases = [
        (0.5, 0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0, 2.0),
        (0.5, 1.0, 0.5, 1.0),
        (1.0, 2.0, 1.0, 2.0),
    ]
    
    print(f"  Grid: {grid.size} points, k-grid: {len(kv)} points")
    print(f"\n  {'α_a':>6} {'α_b':>6} {'α_c':>6} {'α_d':>6}   {'Fourier':>12} {'Analytical':>12} {'Rel Err':>10}")
    print("  " + "-"*70)
    
    all_passed = True
    for alpha_a, alpha_b, alpha_c, alpha_d in test_cases:
        gto_a = evaluate_primitive_gto(grid.points, center, alpha_a, (0, 0, 0))
        gto_b = evaluate_primitive_gto(grid.points, center, alpha_b, (0, 0, 0))
        gto_c = evaluate_primitive_gto(grid.points, center, alpha_c, (0, 0, 0))
        gto_d = evaluate_primitive_gto(grid.points, center, alpha_d, (0, 0, 0))
        
        density_ab = gto_a * gto_b
        density_cd = gto_c * gto_d
        
        eri_f = compute_eri_fourier(density_ab, density_cd, grid, grid, kv, k_weights)
        eri_a = eri_ssss_analytical(alpha_a, alpha_b, alpha_c, alpha_d,
                                    center, center, center, center)
        
        rel_err = abs(eri_f - eri_a) / abs(eri_a)
        if rel_err > 1e-3:
            all_passed = False
        
        print(f"  {alpha_a:>6.1f} {alpha_b:>6.1f} {alpha_c:>6.1f} {alpha_d:>6.1f}   "
              f"{eri_f:>12.6f} {eri_a:>12.6f} {rel_err:>10.2e}")
    
    return all_passed


def test_convergence():
    print("\n-----------------------------")
    print("Test 5: Convergence Study")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    eri_analytical = eri_ssss_analytical(alpha, alpha, alpha, alpha,
                                         center, center, center, center)
    
    print(f"  Analytical ERI: {eri_analytical:.10f}")
    print(f"\n  Convergence with real-space grid:")
    print(f"  {'n_rad':>6} {'n_ang':>6} {'Points':>8} {'ERI':>14} {'Rel Error':>12}")
    print("  " + "-"*50)
    
    kv, k_weights = create_spherical_k_grid(15, 15, 15, k_max=10.0)
    
    for n_rad, n_ang in [(40, 194), (60, 302), (80, 434), (100, 590)]:
        grid = BeckeGrid.from_atom(1, center, n_rad=n_rad, n_ang=n_ang)
        gto = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
        density = gto * gto
        
        eri_f = compute_eri_fourier(density, density, grid, grid, kv, k_weights)
        rel_err = abs(eri_f - eri_analytical) / abs(eri_analytical)
        
        print(f"  {n_rad:>6} {n_ang:>6} {grid.size:>8} {eri_f:>14.8f} {rel_err:>12.2e}")
    
    print(f"\n  Convergence with k-space grid:")
    print(f"  {'n_k':>6} {'k_max':>6} {'k-pts':>8} {'ERI':>14} {'Rel Error':>12}")
    print("  " + "-"*50)
    
    grid = BeckeGrid.from_atom(1, center, n_rad=80, n_ang=434)
    gto = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
    density = gto * gto
    
    for n_k, k_max in [(8, 6), (10, 8), (12, 10), (15, 12), (18, 15)]:
        kv, k_weights = create_spherical_k_grid(n_k, n_k, n_k, k_max=k_max)
        eri_f = compute_eri_fourier(density, density, grid, grid, kv, k_weights)
        rel_err = abs(eri_f - eri_analytical) / abs(eri_analytical)
        
        print(f"  {n_k:>6} {k_max:>6} {len(kv):>8} {eri_f:>14.8f} {rel_err:>12.2e}")
    
    return True


def main():
    print("Two-Electron Integrals via Fourier Space")
    
    test1 = test_eri_same_center()
    test2 = test_eri_two_centers()
    test3 = test_eri_direct_comparison()
    test4 = test_eri_different_exponents()
    test5 = test_convergence()
    

    
    all_passed = test1 and test2 and test3 and test4 and test5
    

    
    return all_passed


if __name__ == "__main__":
    main()
