import numpy as np
from scipy.special import gamma
from typing import List, Tuple
from dataclasses import dataclass


def gaussian_int(n: int, alpha: float) -> float:
    assert n >= 0
    assert alpha > 0

    n1 = (n + 1) * 0.5
    return gamma(n1) / (2.0 * alpha**n1)



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


def gto_norm(l: int, alpha: float) -> float:
    assert l >= 0 
    assert alpha > 0
    norm = gaussian_int(l * 2 + 2, 2 * alpha) ** (-0.5)
    norm *= ((2 * l + 1) / (4 * np.pi)) ** 0.5
    return norm


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
    points = np.asarray(points)
    center = np.asarray(center)
    
    assert points.ndim == 2 and points.shape[1] == 3, "points must be (N, 3)"
    assert center.shape == (3,), "center must be (3,)"
    
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
    exponents = np.asarray(exponents)
    coefficients = np.asarray(coefficients)
    ix, iy, iz = angular
    
    result = np.zeros(len(points))
    
    for alpha, coef in zip(exponents, coefficients):
        norm_coef = coef * cart_gto_norm(alpha, ix, iy, iz)
        primitive = evaluate_primitive_gto(
            points, center, alpha, angular, normalize=False
        )
        result += norm_coef * primitive
    
    return result


def evaluate_gto_shell(
    points: np.ndarray,
    center: np.ndarray,
    exponents: np.ndarray,
    coefficients: np.ndarray,
    l: int
) -> np.ndarray:
    n_points = len(points)
    n_components = n_cart(l)
    
    result = np.zeros((n_components, n_points))
    
    for i, angular in enumerate(iter_cart_xyz(l)):
        result[i] = evaluate_contracted_gto(
            points, center, exponents, coefficients, angular
        )
    
    return result


def test_double_factorial():
    print("\nTest 0: Double Factorial Implementation")
    print("-------------------")
    
    test_cases = [
        (0, 1),
        (1, 1),
        (2, 3),
        (3, 15),
        (4, 105),
    ]
    
    all_passed = True
    print("\n  Testing (2n-1)!! values:")
    for n, expected in test_cases:
        result = double_factorial(n)
        status = "Confirmed" if result == expected else "NOT Confirmed "
        if result != expected:
            all_passed = False
        print(f"    {status} n={n}: (2×{n}-1)!! = {result}")
    
    return all_passed


def test_normalization():
    print("\n-------------------")
    print("Test 1: GTO Normalization (Cartesian)")
    print()
    
    n = 50
    L = 10.0
    x = np.linspace(-L, L, n)
    dx = x[1] - x[0]
    
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    center = np.array([0.0, 0.0, 0.0])
    
    print(f"Grid: {n}³ = {n**3} points, spacing = {dx:.4f} Bohr")
    print(f"Integration volume element: dV = {dx**3:.6f}")
    
    test_cases = [
        (1.0, (0, 0, 0), "s (0,0,0)"),
        (0.5, (0, 0, 0), "s (α=0.5)"),
        (1.0, (1, 0, 0), "px (1,0,0)"),
        (1.0, (0, 1, 0), "py (0,1,0)"),
        (1.0, (0, 0, 1), "pz (0,0,1)"),
        (1.0, (2, 0, 0), "dxx (2,0,0)"),
        (1.0, (1, 1, 0), "dxy (1,1,0)"),
        (1.0, (1, 0, 1), "dxz (1,0,1)"),
        (1.0, (0, 2, 0), "dyy (0,2,0)"),
        (1.0, (0, 1, 1), "dyz (0,1,1)"),
        (1.0, (0, 0, 2), "dzz (0,0,2)"),
    ]
    
    print("\nTesting ∫|χ(r)|² dr = 1:")
    print("-------------------")
    
    print("\n  Normalization constants:")
    for alpha, angular, name in test_cases[:7]:
        norm = cart_gto_norm(alpha, *angular)
        print(f"    {name:15s}: N = {norm:.6f}")
    
    print("\n  Integration results:")
    all_passed = True
    for alpha, angular, name in test_cases:
        gto_vals = evaluate_primitive_gto(points, center, alpha, angular)
        integral = np.sum(gto_vals**2) * dx**3
        error = abs(integral - 1.0)
        status = "Confirmed " if error < 0.01 else "NOT Confirmed"

        if error >= 0.01:
            all_passed = False
        print(f"    {status} {name:15s}: ∫|χ|² dr = {integral:.6f} (error: {error:.2e})")
    
    return all_passed


def test_gaussian_decay():
    print("\n-------------------")
    print("Test 2: Gaussian Decay Behavior")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    r_values = np.linspace(0, 5, 50)
    points = np.column_stack([r_values, np.zeros_like(r_values), np.zeros_like(r_values)])
    
    gto_vals = evaluate_primitive_gto(points, center, alpha, (0, 0, 0))
    
    N = cart_gto_norm(alpha, 0, 0, 0)
    expected = N * np.exp(-alpha * r_values**2)
    
    max_error = np.max(np.abs(gto_vals - expected))
    
    print("Comparing s-type GTO vs analytical formula along x-axis:")
    print(f"  Normalization constant N = {N:.6f}")
    print(f"  Max absolute error: {max_error:.2e}")
    
    print("\n  GTO values at selected distances:")
    for r_idx in [0, 10, 20, 30, 40]:
        print(f"    r = {r_values[r_idx]:.1f}: χ(r) = {gto_vals[r_idx]:.6e}")
    
    return max_error < 1e-10


def test_angular_nodes():
    print("\n-------------------")
    print("Test 3: Angular Nodal Structure")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    alpha = 1.0
    
    points_yz = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 2.0],
    ])
    px_vals = evaluate_primitive_gto(points_yz, center, alpha, (1, 0, 0))
    px_node_test = np.allclose(px_vals, 0.0)
    print(f"  px orbital at x=0 plane: {px_vals}")
    print(f"   Nodal plane correct" if px_node_test else "  ✗ Nodal plane incorrect")
    
    points_pm = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    px_pm = evaluate_primitive_gto(points_pm, center, alpha, (1, 0, 0))
    sign_change = px_pm[0] * px_pm[1] < 0
    print(f"\n  px at (1,0,0) and (-1,0,0): {px_pm}")
    print(f"   Sign change correct" if sign_change else "  ✗ Sign change incorrect")
    
    points_quad = np.array([
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
    ])
    dxy_vals = evaluate_primitive_gto(points_quad, center, alpha, (1, 1, 0))
    quadrant_symmetry = (
        np.isclose(dxy_vals[0], dxy_vals[2]) and
        np.isclose(dxy_vals[1], dxy_vals[3]) and
        dxy_vals[0] * dxy_vals[1] < 0
    )
    print(f"\n  dxy at 4 quadrants: {dxy_vals}")
    print(f"   Quadrant symmetry correct" if quadrant_symmetry else "  ✗ Quadrant symmetry incorrect")
    
    return px_node_test and sign_change and quadrant_symmetry


def test_contraction():
    print("\n-------------------")
    print("Test 4: Contracted GTO (STO-3G Hydrogen 1s)")
    print()
    
    exponents = np.array([3.42525091, 0.62391373, 0.16885540])
    coefficients = np.array([0.15432897, 0.53532814, 0.44463454])
    center = np.array([0.0, 0.0, 0.0])
    
    print("STO-3G H 1s parameters:")
    print(f"  Exponents:     {exponents}")
    print(f"  Coefficients:  {coefficients}")
    
    n = 50
    L = 5.0
    x = np.linspace(-L, L, n)
    dx = x[1] - x[0]
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    cgto_vals = evaluate_contracted_gto(
        points, center, exponents, coefficients, (0, 0, 0)
    )
    
    integral = np.sum(cgto_vals**2) * dx**3
    print(f"\n  Numerical ∫|χ|² dr = {integral:.6f}")
    print("  Expected: ~1.0 (some error due to finite grid)")
    
    max_idx = np.argmax(cgto_vals)
    max_point = points[max_idx]
    print(f"\n  Maximum value at: {max_point}")
    print(f"   Centered at origin" if np.linalg.norm(max_point) < dx else "   Not centered")
    
    return True


def test_shell_evaluation():
    print("\n-------------------")
    print("Test 5: Shell Evaluation (all d-type orbitals)")
    print()
    
    center = np.array([0.0, 0.0, 0.0])
    exponents = np.array([1.0])
    coefficients = np.array([1.0])
    l = 2
    
    n = 40
    L = 8.0
    x = np.linspace(-L, L, n)
    dx = x[1] - x[0]
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    shell_vals = evaluate_gto_shell(points, center, exponents, coefficients, l)
    
    print(f"  d-shell has {n_cart(l)} Cartesian components:")
    for i, angular in enumerate(iter_cart_xyz(l)):
        integral = np.sum(shell_vals[i]**2) * dx**3
        print(f"    Component {i} {angular}: ∫|χ|² dr = {integral:.4f}")
    
    print("\n  Checking orthogonality ⟨χᵢ|χⱼ⟩:")
    n_comp = n_cart(l)
    orthogonal = True
    for i in range(n_comp):
        for j in range(i+1, n_comp):
            overlap = np.sum(shell_vals[i] * shell_vals[j]) * dx**3
            if abs(overlap) > 0.01:
                orthogonal = False
            print(f"    ⟨{i}|{j}⟩ = {overlap:.6f}")
    
    print(f"   Components are orthogonal" if orthogonal else "   Components not orthogonal")
    
    return orthogonal


def main():
    print("GTO Evaluation on Grid Points")
    
    test0 = test_double_factorial()
    test1 = test_normalization()
    test2 = test_gaussian_decay()
    test3 = test_angular_nodes()
    test4 = test_contraction()
    test5 = test_shell_evaluation()
    
    
    all_passed = test0 and test1 and test2 and test3 and test4 and test5

    
    return all_passed


if __name__ == "__main__":
    main()
