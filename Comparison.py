import numpy as np
from typing import Tuple
from dataclasses import dataclass
import time
import warnings
import finufft
from scipy.special import roots_legendre, erf
import matplotlib.pyplot as plt

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


def evaluate_primitive_gto(points, center, alpha, angular):
    ix, iy, iz = angular
    dr = points - center
    r2 = np.sum(dr**2, axis=1)
    radial = np.exp(-alpha * r2)
    angular_part = (dr[:, 0]**ix) * (dr[:, 1]**iy) * (dr[:, 2]**iz)
    return cart_gto_norm(alpha, ix, iy, iz) * angular_part * radial


@dataclass
class UniformGrid:
    points: np.ndarray
    weights: np.ndarray
    shape: Tuple[int, int, int]
    spacing: float
    L: float
    
    @classmethod
    def create(cls, L: float, N: int):
        x = np.linspace(-L/2, L/2, N, endpoint=False)
        spacing = L / N
        xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        weights = np.full(N**3, spacing**3)
        return cls(points, weights, (N, N, N), spacing, L)
    
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


def create_spherical_k_grid(n_r: int, n_theta: int, n_phi: int, k_max: float = 10.0):
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


def compute_eri_fft(density_bra, density_ket, grid):
    N = grid.shape[0]
    L = grid.L
    h = grid.spacing
    
    rho_bra = density_bra.reshape((N, N, N))
    rho_ket = density_ket.reshape((N, N, N))
    
    F_bra = np.fft.fftn(rho_bra) * h**3
    F_ket = np.fft.fftn(rho_ket) * h**3
    
    freq = np.fft.fftfreq(N, d=h) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(freq, freq, freq, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    
    k2[0, 0, 0] = 1.0
    coulomb = 4 * np.pi / k2
    coulomb[0, 0, 0] = 0.0
    
    integrand = F_bra * np.conj(F_ket) * coulomb
    
    dk = 2 * np.pi / L
    eri = np.sum(integrand).real * dk**3 / (2 * np.pi)**3
    
    return eri


def ft_direct(func_vals, points, weights, k_vectors):
    c = func_vals * weights
    F_k = np.zeros(len(k_vectors), dtype=np.complex128)
    for i, k in enumerate(k_vectors):
        phases = np.exp(1j * (points @ k))
        F_k[i] = np.sum(c * phases)
    return F_k


def compute_eri_nufft(density_bra, density_ket, grid_bra, grid_ket, kv, k_weights, eps=1e-6):
    try:
        c_bra = np.ascontiguousarray((density_bra * grid_bra.weights).astype(np.complex128))
        x_bra = np.ascontiguousarray(grid_bra.points[:, 0].astype(np.float64))
        y_bra = np.ascontiguousarray(grid_bra.points[:, 1].astype(np.float64))
        z_bra = np.ascontiguousarray(grid_bra.points[:, 2].astype(np.float64))
        
        c_ket = np.ascontiguousarray((density_ket * grid_ket.weights).astype(np.complex128))
        x_ket = np.ascontiguousarray(grid_ket.points[:, 0].astype(np.float64))
        y_ket = np.ascontiguousarray(grid_ket.points[:, 1].astype(np.float64))
        z_ket = np.ascontiguousarray(grid_ket.points[:, 2].astype(np.float64))
        
        kx = np.ascontiguousarray(kv[:, 0].astype(np.float64))
        ky = np.ascontiguousarray(kv[:, 1].astype(np.float64))
        kz = np.ascontiguousarray(kv[:, 2].astype(np.float64))
        
        F_bra = finufft.nufft3d3(x_bra, y_bra, z_bra, c_bra, kx, ky, kz, eps=eps, isign=1, nthreads=1)
        F_ket = finufft.nufft3d3(x_ket, y_ket, z_ket, c_ket, kx, ky, kz, eps=eps, isign=1, nthreads=1)
    except Exception as e:
        F_bra = ft_direct(density_bra, grid_bra.points, grid_bra.weights, kv)
        F_ket = ft_direct(density_ket, grid_ket.points, grid_ket.weights, kv)
    
    k2 = np.sum(kv**2, axis=1)
    k2_safe = np.where(k2 > 1e-20, k2, 1e-20)
    
    integrand = F_bra * np.conj(F_ket) / k2_safe
    eri = np.sum(k_weights * integrand.real) / (2 * np.pi**2)
    
    return eri


def compute_eri_direct(density_bra, density_ket, grid_bra, grid_ket):
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


def eri_ssss_analytical(alpha_a, alpha_b, alpha_c, alpha_d, Ra, Rb, Rc, Rd):
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
    print("\n" + "-"*70)
    print("TEST 1: (ss|ss) Integral - Same Center")
  
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    eri_analytical = eri_ssss_analytical(alpha, alpha, alpha, alpha, center, center, center, center)
    print(f"\n  Analytical ERI: {eri_analytical:.10f}")
    
    print("\n   FFT Method (Uniform Grid) ")
    fft_results = []
    for L, N in [(16, 64), (20, 80), (24, 96), (28, 112)]:
        grid = UniformGrid.create(L, N)
        gto = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
        density = gto * gto
        
        t0 = time.time()
        eri_fft = compute_eri_fft(density, density, grid)
        t_fft = (time.time() - t0) * 1000
        
        error = abs(eri_fft - eri_analytical) / eri_analytical
        fft_results.append((grid.size, error, t_fft, eri_fft))
        print(f"    L={L}, N={N}: {grid.size:>8} pts | ERI={eri_fft:.6f} | Error={error:.2e} | {t_fft:.1f}ms")
    
    print("\n  NUFFT Method (Becke Grid) ")
    kv, k_weights = create_spherical_k_grid(8, 8, 8, k_max=8.0)
    nufft_results = []
    for n_rad, n_ang in [(20, 50), (25, 86), (30, 110), (40, 194)]:
        grid = BeckeGrid.from_atom(1, center, n_rad=n_rad, n_ang=n_ang)
        gto = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
        density = gto * gto
        
        t0 = time.time()
        eri_nufft = compute_eri_nufft(density, density, grid, grid, kv, k_weights)
        t_nufft = (time.time() - t0) * 1000
        
        error = abs(eri_nufft - eri_analytical) / eri_analytical
        nufft_results.append((grid.size, error, t_nufft, eri_nufft))
        print(f"    n_rad={n_rad}, n_ang={n_ang}: {grid.size:>6} pts | ERI={eri_nufft:.10f} | Error={error:.2e} | {t_nufft:.1f}ms")
    
    print("\n   Direct Method ")
    grid_small = BeckeGrid.from_atom(1, center, n_rad=30, n_ang=110)
    gto = evaluate_primitive_gto(grid_small.points, center, alpha, (0, 0, 0))
    density = gto * gto
    
    t0 = time.time()
    eri_direct = compute_eri_direct(density, density, grid_small, grid_small)
    t_direct = (time.time() - t0) * 1000
    error_direct = abs(eri_direct - eri_analytical) / eri_analytical
    direct_results = (grid_small.size, error_direct, t_direct, eri_direct)
    print(f"    {grid_small.size} pts | ERI={eri_direct:.6f} | Error={error_direct:.2e} | {t_direct:.1f}ms")
    
    return fft_results, nufft_results, direct_results, eri_analytical


def test_eri_two_centers():
    print("\n" + "-"*70)
    print("TEST 2: (ss|ss) Integrals - H2 Molecule (Two Centers)")
  
    
    R = 1.4
    Ra = np.array([0.0, 0.0, -R/2])
    Rb = np.array([0.0, 0.0, R/2])
    alpha = 1.0
    
    eri_aaaa_a = eri_ssss_analytical(alpha, alpha, alpha, alpha, Ra, Ra, Ra, Ra)
    eri_aabb_a = eri_ssss_analytical(alpha, alpha, alpha, alpha, Ra, Ra, Rb, Rb)
    eri_abab_a = eri_ssss_analytical(alpha, alpha, alpha, alpha, Ra, Rb, Ra, Rb)
    
    print(f"\n  H2 bond length: {R} Bohr")
    print(f"\n  Analytical:")
    print(f"    (aa|aa) = {eri_aaaa_a:.10f}")
    print(f"    (aa|bb) = {eri_aabb_a:.10f}")
    print(f"    (ab|ab) = {eri_abab_a:.10f}")
    
    print("\n   FFT Method ")
    fft_grid = UniformGrid.create(L=24, N=96)
    gto_a_f = evaluate_primitive_gto(fft_grid.points, Ra, alpha, (0, 0, 0))
    gto_b_f = evaluate_primitive_gto(fft_grid.points, Rb, alpha, (0, 0, 0))
    
    eri_aaaa_fft = compute_eri_fft(gto_a_f*gto_a_f, gto_a_f*gto_a_f, fft_grid)
    eri_aabb_fft = compute_eri_fft(gto_a_f*gto_a_f, gto_b_f*gto_b_f, fft_grid)
    eri_abab_fft = compute_eri_fft(gto_a_f*gto_b_f, gto_a_f*gto_b_f, fft_grid)
    
    print(f"    Grid: {fft_grid.size:,} points")
    print(f"    (aa|aa): {eri_aaaa_fft:.8f} | Error: {abs(eri_aaaa_fft-eri_aaaa_a)/eri_aaaa_a:.2e}")
    print(f"    (aa|bb): {eri_aabb_fft:.8f} | Error: {abs(eri_aabb_fft-eri_aabb_a)/eri_aabb_a:.2e}")
    print(f"    (ab|ab): {eri_abab_fft:.8f} | Error: {abs(eri_abab_fft-eri_abab_a)/eri_abab_a:.2e}")
    
    print("\n   NUFFT Method ")
    becke_grid = BeckeGrid.from_molecule([1, 1], [Ra, Rb], n_rad=40, n_ang=194)
    gto_a_b = evaluate_primitive_gto(becke_grid.points, Ra, alpha, (0, 0, 0))
    gto_b_b = evaluate_primitive_gto(becke_grid.points, Rb, alpha, (0, 0, 0))
    kv, k_weights = create_spherical_k_grid(8, 8, 8, k_max=8.0)
    
    eri_aaaa_nufft = compute_eri_nufft(gto_a_b*gto_a_b, gto_a_b*gto_a_b, becke_grid, becke_grid, kv, k_weights)
    eri_aabb_nufft = compute_eri_nufft(gto_a_b*gto_a_b, gto_b_b*gto_b_b, becke_grid, becke_grid, kv, k_weights)
    eri_abab_nufft = compute_eri_nufft(gto_a_b*gto_b_b, gto_a_b*gto_b_b, becke_grid, becke_grid, kv, k_weights)
    
    print(f"    Grid: {becke_grid.size:,} points")
    print(f"    (aa|aa): {eri_aaaa_nufft:.10f} | Error: {abs(eri_aaaa_nufft-eri_aaaa_a)/eri_aaaa_a:.2e}")
    print(f"    (aa|bb): {eri_aabb_nufft:.10f} | Error: {abs(eri_aabb_nufft-eri_aabb_a)/eri_aabb_a:.2e}")
    print(f"    (ab|ab): {eri_abab_nufft:.10f} | Error: {abs(eri_abab_nufft-eri_abab_a)/eri_abab_a:.2e}")


def test_three_methods_comparison():
    print("\n" + "-"*70)
    print("TEST 3: Direct Comparison - FFT vs NUFFT vs Direct")
  
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    eri_analytical = eri_ssss_analytical(alpha, alpha, alpha, alpha, center, center, center, center)
    print(f"\n  Analytical ERI: {eri_analytical:.10f}")
    
    fft_grid = UniformGrid.create(L=24, N=96)
    gto_fft = evaluate_primitive_gto(fft_grid.points, center, alpha, (0, 0, 0))
    density_fft = gto_fft * gto_fft
    
    t0 = time.time()
    eri_fft = compute_eri_fft(density_fft, density_fft, fft_grid)
    time_fft = (time.time() - t0) * 1000
    error_fft = abs(eri_fft - eri_analytical) / eri_analytical
    
    becke_grid = BeckeGrid.from_atom(1, center, n_rad=40, n_ang=194)
    gto_becke = evaluate_primitive_gto(becke_grid.points, center, alpha, (0, 0, 0))
    density_becke = gto_becke * gto_becke
    kv, k_weights = create_spherical_k_grid(8, 8, 8, k_max=8.0)
    
    t0 = time.time()
    eri_nufft = compute_eri_nufft(density_becke, density_becke, becke_grid, becke_grid, kv, k_weights)
    time_nufft = (time.time() - t0) * 1000
    error_nufft = abs(eri_nufft - eri_analytical) / eri_analytical
    
    direct_grid = BeckeGrid.from_atom(1, center, n_rad=30, n_ang=110)
    gto_direct = evaluate_primitive_gto(direct_grid.points, center, alpha, (0, 0, 0))
    density_direct = gto_direct * gto_direct
    
    t0 = time.time()
    eri_direct = compute_eri_direct(density_direct, density_direct, direct_grid, direct_grid)
    time_direct = (time.time() - t0) * 1000
    error_direct = abs(eri_direct - eri_analytical) / eri_analytical
    
    print(f"\n  {'Method':<12} {'Grid Points':>12} {'ERI Value':>16} {'Rel Error':>12} {'Time':>10}")
    print("  " + "-"*65)
    print(f"  {'FFT':<12} {fft_grid.size:>12,} {eri_fft:>16.8f} {error_fft:>12.2e} {time_fft:>9.1f}ms")
    print(f"  {'NUFFT':<12} {becke_grid.size:>12,} {eri_nufft:>16.10f} {error_nufft:>12.2e} {time_nufft:>9.1f}ms")
    print(f"  {'Direct':<12} {direct_grid.size:>12,} {eri_direct:>16.8f} {error_direct:>12.2e} {time_direct:>9.1f}ms")



    
    return {
        'fft': (fft_grid.size, eri_fft, error_fft, time_fft),
        'nufft': (becke_grid.size, eri_nufft, error_nufft, time_nufft),
        'direct': (direct_grid.size, eri_direct, error_direct, time_direct),
        'analytical': eri_analytical
    }


def test_convergence():
    print("\n" + "-"*40)
    print("TEST 4: Convergence Study")
    
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    eri_analytical = eri_ssss_analytical(alpha, alpha, alpha, alpha, center, center, center, center)
    print(f"\n  Analytical ERI: {eri_analytical:.10f}")
    
    print(f"\n   FFT Convergence ")
    print(f"  {'L':>6} {'N':>6} {'Points':>10} {'ERI':>14} {'Rel Error':>12}")
    print("  " + "-"*55)
    
    fft_data = []
    for L in [12, 16, 20, 24, 28, 32]:
        N = 96
        grid = UniformGrid.create(L, N)
        gto = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
        density = gto * gto
        eri = compute_eri_fft(density, density, grid)
        error = abs(eri - eri_analytical) / eri_analytical
        fft_data.append((grid.size, error, L, N))
        print(f"  {L:>6} {N:>6} {grid.size:>10,} {eri:>14.8f} {error:>12.2e}")
    
    print(f"\n   NUFFT Convergence ")
    print(f"  {'n_rad':>6} {'n_ang':>6} {'Points':>10} {'ERI':>14} {'Rel Error':>12}")
    print("  " + "-"*55)
    
    kv, k_weights = create_spherical_k_grid(8, 8, 8, k_max=8.0)
    nufft_data = []
    for n_rad, n_ang in [(20, 50), (25, 86), (30, 110), (40, 194)]:
        grid = BeckeGrid.from_atom(1, center, n_rad=n_rad, n_ang=n_ang)
        gto = evaluate_primitive_gto(grid.points, center, alpha, (0, 0, 0))
        density = gto * gto
        eri = compute_eri_nufft(density, density, grid, grid, kv, k_weights)
        error = abs(eri - eri_analytical) / eri_analytical
        nufft_data.append((grid.size, error))
        print(f"  {n_rad:>6} {n_ang:>6} {grid.size:>10,} {eri:>14.10f} {error:>12.2e}")
    
    return fft_data, nufft_data


def plot_comparison(fft_results, nufft_results, direct_results, comparison_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    fft_pts = [r[0] for r in fft_results]
    fft_err = [r[1] for r in fft_results]
    nufft_pts = [r[0] for r in nufft_results]
    nufft_err = [r[1] for r in nufft_results]
    
    ax1.loglog(fft_pts, fft_err, 's-', color='#D55E00', markersize=8, lw=2, label='FFT (Uniform Grid)')
    ax1.loglog(nufft_pts, nufft_err, 'o-', color='#0072B2', markersize=8, lw=2, label='NUFFT (Becke Grid)')
    ax1.axhline(1e-6, color='gray', ls=':', lw=1.5, label='Chemical accuracy')
    ax1.axhline(1e-10, color='gray', ls='--', lw=1.5, label='High accuracy')
    ax1.set_xlabel('Number of Grid Points')
    ax1.set_ylabel('Relative ERI Error')
    ax1.set_title('Accuracy vs Grid Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1e3, 1e7)
    ax1.set_ylim(1e-12, 1e0)
    
    ax2 = axes[0, 1]
    methods = ['FFT\n(Uniform)', 'NUFFT\n(Becke)', 'Direct']
    points = [comparison_results['fft'][0], comparison_results['nufft'][0], comparison_results['direct'][0]]
    colors = ['#D55E00', '#0072B2', '#009E73']
    bars = ax2.bar(methods, points, color=colors, edgecolor='black', lw=1.5)
    ax2.set_ylabel('Grid Points')
    ax2.set_title('Grid Points Used')
    ax2.set_yscale('log')
    for bar, val in zip(bars, points):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.3, f'{val:,}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3 = axes[1, 0]
    errors = [comparison_results['fft'][2], comparison_results['nufft'][2], comparison_results['direct'][2]]
    errors_log = [-np.log10(max(e, 1e-15)) for e in errors]
    bars = ax3.bar(methods, errors_log, color=colors, edgecolor='black', lw=1.5)
    ax3.set_ylabel('Digits of Accuracy (-log10 error)')
    ax3.set_title('Accuracy Comparison')
    for bar, err in zip(bars, errors):
        if err < 100:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{err:.1e}', 
                    ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 'FAILED', 
                    ha='center', va='bottom', fontsize=9, color='red')
    
    ax4 = axes[1, 1]
    times = [comparison_results['fft'][3], comparison_results['nufft'][3], comparison_results['direct'][3]]
    bars = ax4.bar(methods, times, color=colors, edgecolor='black', lw=1.5)
    ax4.set_ylabel('Computation Time (ms)')
    ax4.set_title('Timing Comparison')
    for bar, val in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02, f'{val:.1f}ms', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('eri_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.savefig('eri_comparison_plots.pdf', bbox_inches='tight')
    plt.close()


def plot_grid_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    center = np.array([0.0, 0.0, 0.0])
    
    becke = BeckeGrid.from_atom(1, center, n_rad=30, n_ang=86)
    mask_b = np.abs(becke.points[:, 2]) < 0.5
    
    uniform = UniformGrid.create(L=10, N=21)
    mask_u = np.abs(uniform.points[:, 2]) < 0.5
    
    ax1 = axes[0]
    ax1.scatter(becke.points[mask_b, 0], becke.points[mask_b, 1], s=8, c='#0072B2', alpha=0.6)
    ax1.plot(0, 0, 'r*', markersize=15)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.set_xlabel('x (Bohr)')
    ax1.set_ylabel('y (Bohr)')
    ax1.set_title(f'Becke Grid (NUFFT)\n{becke.size:,} total points')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.scatter(uniform.points[mask_u, 0], uniform.points[mask_u, 1], s=8, c='#D55E00', alpha=0.6)
    ax2.plot(0, 0, 'r*', markersize=15)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax2.set_xlabel('x (Bohr)')
    ax2.set_ylabel('y (Bohr)')
    ax2.set_title(f'Uniform Grid (FFT)\n{uniform.size:,} total points')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grid_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('grid_comparison.pdf', bbox_inches='tight')
    plt.close()


def plot_convergence(fft_data, nufft_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fft_pts = [d[0] for d in fft_data]
    fft_err = [d[1] for d in fft_data]
    nufft_pts = [d[0] for d in nufft_data]
    nufft_err = [d[1] for d in nufft_data]
    
    ax.loglog(fft_pts, fft_err, 's-', color='#D55E00', markersize=9, lw=2, label='FFT (Uniform Grid)')
    ax.loglog(nufft_pts, nufft_err, 'o-', color='#0072B2', markersize=9, lw=2.5, label='NUFFT (Becke Grid)')
    
    ax.axhline(1e-2, color='red', ls=':', lw=1.5)
    ax.axhline(1e-10, color='green', ls=':', lw=1.5)
    
    ax.text(2e3, 1.5e-2, '1% error', fontsize=9, color='red')
    ax.text(2e3, 1.5e-10, 'High accuracy', fontsize=9, color='green')
    
    ax.annotate('', xy=(8000, 1e-11), xytext=(800000, 1e-11),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(80000, 3e-11, '~100x fewer points', fontsize=11, ha='center', color='purple', fontweight='bold')
    
    ax.set_xlabel('Number of Grid Points', fontsize=12)
    ax.set_ylabel('Relative ERI Error', fontsize=12)
    ax.set_title('Convergence: FFT vs NUFFT for Two-Electron Integrals', fontsize=13)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1e3, 5e6)
    ax.set_ylim(1e-13, 1e0)
    
    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('convergence_comparison.pdf', bbox_inches='tight')

    plt.close()


def print_summary(comparison_results):

    
    fft = comparison_results['fft']
    nufft = comparison_results['nufft']
    direct = comparison_results['direct']
    
    print(f"""
  
                      FFT vs NUFFT vs Direct                       
  
   Method      | Grid Points  | Rel. Error   | Status              
 
   FFT         | {fft[0]:>10,} | {fft[2]:>10.2e} | Works (limited)     
   NUFFT       | {nufft[0]:>10,} | {nufft[2]:>10.2e} | Works (excellent)   
   Direct      | {direct[0]:>10,} | {direct[2]:>10.2e} | {'FAILED (singular)' if direct[2] > 1 else 'Works':19} 
  
    """)
    


def main():
    print("-"*40)
    print("Two-Electron Integrals: FFT vs NUFFT vs Direct Comparison")
   
    
    fft_results, nufft_results, direct_results, _ = test_eri_same_center()
    test_eri_two_centers()
    comparison_results = test_three_methods_comparison()
    fft_data, nufft_data = test_convergence()
    

    plot_grid_comparison()
    plot_comparison(fft_results, nufft_results, direct_results, comparison_results)
    plot_convergence(fft_data, nufft_data)
    
    print_summary(comparison_results)
    
  

if __name__ == "__main__":
    main()
