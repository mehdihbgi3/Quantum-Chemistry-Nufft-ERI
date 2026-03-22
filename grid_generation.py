import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import warnings

from grid.onedgrid import GaussLegendre, GaussChebyshev
from grid.rtransform import BeckeRTransform
from grid.atomgrid import AtomGrid
from grid.molgrid import MolGrid
from grid.becke import BeckeWeights


@dataclass
class UniformGrid:
    points: np.ndarray
    weights: np.ndarray
    shape: Tuple[int, int, int]
    origin: np.ndarray
    spacing: float
    
    @classmethod
    def from_box(
        cls,
        box_min: np.ndarray,
        box_max: np.ndarray,
        spacing: float
    ) -> 'UniformGrid':
        box_min = np.asarray(box_min)
        box_max = np.asarray(box_max)
        
        n_points = np.ceil((box_max - box_min) / spacing).astype(int) + 1
        nx, ny, nz = n_points
        
        x = np.linspace(box_min[0], box_max[0], nx)
        y = np.linspace(box_min[1], box_max[1], ny)
        z = np.linspace(box_min[2], box_max[2], nz)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        dV = spacing ** 3
        weights = np.full(len(points), dV)
        
        return cls(
            points=points,
            weights=weights,
            shape=(nx, ny, nz),
            origin=box_min.copy(),
            spacing=spacing
        )
    
    @classmethod
    def from_center(
        cls,
        center: np.ndarray,
        extent: float,
        spacing: float
    ) -> 'UniformGrid':
        center = np.asarray(center)
        box_min = center - extent
        box_max = center + extent
        return cls.from_box(box_min, box_max, spacing)
    
    @classmethod
    def from_molecule(
        cls,
        coordinates: np.ndarray,
        spacing: float,
        padding: float = 5.0
    ) -> 'UniformGrid':
        coordinates = np.asarray(coordinates)
        box_min = coordinates.min(axis=0) - padding
        box_max = coordinates.max(axis=0) + padding
        return cls.from_box(box_min, box_max, spacing)
    
    @property
    def size(self) -> int:
        return len(self.points)
    
    def integrate(self, values: np.ndarray) -> float:
        return np.sum(values * self.weights)


@dataclass  
class BeckeGrid:
    points: np.ndarray
    weights: np.ndarray
    molgrid: MolGrid
    
    @classmethod
    def from_molecule(
        cls,
        atnums: np.ndarray,
        atcoords: np.ndarray,
        n_rad: int = 75,
        n_ang: int = 302,
        preset: Optional[str] = None
    ) -> 'BeckeGrid':
        atnums = np.asarray(atnums)
        atcoords = np.asarray(atcoords)
        
        oned = GaussLegendre(npoints=n_rad)
        rgrid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if preset is not None:
                molgrid = MolGrid.from_preset(
                    atnums=atnums,
                    atcoords=atcoords,
                    rgrid=rgrid,
                    preset=preset,
                    aim_weights=BeckeWeights(),
                    store=True
                )
            else:
                molgrid = MolGrid.from_size(
                    atnums=atnums,
                    atcoords=atcoords,
                    rgrid=rgrid,
                    size=n_ang,
                    aim_weights=BeckeWeights(),
                    store=True
                )
        
        return cls(
            points=molgrid.points,
            weights=molgrid.weights,
            molgrid=molgrid
        )
    
    @classmethod
    def from_atom(
        cls,
        atnum: int,
        center: np.ndarray,
        n_rad: int = 75,
        n_ang: int = 302
    ) -> 'BeckeGrid':
        return cls.from_molecule(
            atnums=np.array([atnum]),
            atcoords=np.array([center]),
            n_rad=n_rad,
            n_ang=n_ang
        )
    
    @property
    def size(self) -> int:
        return len(self.points)
    
    def integrate(self, values: np.ndarray) -> float:
        return np.sum(values * self.weights)


def test_uniform_grid_basic():
    print("\nTest 1: Uniform Grid - Basic Functionality")
    print()
    
    grid = UniformGrid.from_center(
        center=np.array([0.0, 0.0, 0.0]),
        extent=5.0,
        spacing=0.5
    )
    
    print("  Grid properties:")
    print(f"    Shape: {grid.shape}")
    print(f"    Size: {grid.size} points")
    print(f"    Origin: {grid.origin}")
    print(f"    Spacing: {grid.spacing}")
    print(f"    Points shape: {grid.points.shape}")
    print(f"    Weights shape: {grid.weights.shape}")
    print(f"    Weight value: {grid.weights[0]:.6f}")
    
    expected_volume = (2 * 5.0) ** 3
    grid_volume = grid.weights.sum()
    print("\n  Volume check:")
    print(f"    Expected volume: {expected_volume:.1f}")
    print(f"    Grid volume (Σ weights): {grid_volume:.1f}")
    
    volume_ok = abs(grid_volume - expected_volume) / expected_volume < 0.1
    print(f"     Volume correct" if volume_ok else "     Volume incorrect")
    
    return volume_ok


def test_uniform_grid_integration():
    print("\n---------------------------")
    print("Test 2: Uniform Grid - Integration Accuracy")
    print()
    
    grid = UniformGrid.from_center(
        center=np.array([0.0, 0.0, 0.0]),
        extent=8.0,
        spacing=0.2
    )
    
    print(f"  Grid: {grid.shape}, {grid.size} points")
    
    alpha = 1.0
    r2 = np.sum(grid.points**2, axis=1)
    gaussian = np.exp(-alpha * r2)
    integral = grid.integrate(gaussian)
    expected = (np.pi / alpha) ** 1.5
    error1 = abs(integral - expected) / expected
    
    print("\n  Test: ∫ exp(-r²) d³r")
    print(f"    Numerical: {integral:.6f}")
    print(f"    Analytical: {expected:.6f}")
    print(f"    Relative error: {error1:.2e}")
    
    norm_gaussian = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2)
    integral2 = grid.integrate(norm_gaussian)
    error2 = abs(integral2 - 1.0)
    
    print("\n  Test: ∫ (α/π)^(3/2) exp(-α r²) d³r")
    print(f"    Numerical: {integral2:.6f}")
    print(f"    Expected: 1.0")
    print(f"    Absolute error: {error2:.2e}")
    
    passed = error1 < 0.01 and error2 < 0.01
    print(f"\n  {' Confirmed' if passed else ' NOT Confirmed'}")
    
    return passed


def test_becke_grid_basic():
    print("\n---------------------------")
    print("Test 3: Becke Grid - Basic Functionality")
    print()
    
    grid = BeckeGrid.from_atom(
        atnum=1,
        center=np.array([0.0, 0.0, 0.0]),
        n_rad=50,
        n_ang=194
    )
    
    print("  Grid properties:")
    print(f"    Size: {grid.size} points")
    print(f"    Points shape: {grid.points.shape}")
    print(f"    Weights shape: {grid.weights.shape}")
    print(f"    Min weight: {grid.weights.min():.2e}")
    print(f"    Max weight: {grid.weights.max():.2e}")
    
    r = np.linalg.norm(grid.points, axis=1)
    print("\n  Radial distribution:")
    print(f"    Min r: {r.min():.4f}")
    print(f"    Max r: {r.max():.4f}")
    print(f"    Mean r: {r.mean():.4f}")
    
    return True


def test_becke_grid_integration():
    print("\n---------------------------")
    print("Test 4: Becke Grid - Integration Accuracy")
    print()
    
    grid = BeckeGrid.from_atom(
        atnum=1,
        center=np.array([0.0, 0.0, 0.0]),
        n_rad=100,
        n_ang=302
    )
    
    print(f"  Grid: {grid.size} points")
    
    alpha = 1.0
    r2 = np.sum(grid.points**2, axis=1)
    norm_gaussian = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2)
    integral = grid.integrate(norm_gaussian)
    error = abs(integral - 1.0)
    
    print("\n  Test: ∫ (α/π)^(3/2) exp(-α r²) d³r")
    print(f"    Numerical: {integral:.10f}")
    print(f"    Expected: 1.0")
    print(f"    Absolute error: {error:.2e}")
    
    print("\n  Testing with various exponents:")
    all_good = True
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        norm_gaussian = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2)
        integral = grid.integrate(norm_gaussian)
        error = abs(integral - 1.0)
        status = "Confirmed" if error < 1e-6 else "NOT Confirmed"
        if error >= 1e-6:
            all_good = False
        print(f"    {status} α = {alpha}: integral = {integral:.8f}, error = {error:.2e}")
    
    return all_good


def test_becke_grid_molecule():
    print("\n---------------------------")
    print("Test 5: Becke Grid - H2 Molecule")
    print()
    
    atnums = np.array([1, 1])
    atcoords = np.array([
        [0.0, 0.0, -0.7],
        [0.0, 0.0,  0.7]
    ])
    
    grid = BeckeGrid.from_molecule(
        atnums=atnums,
        atcoords=atcoords,
        n_rad=75,
        n_ang=302
    )
    
    print("  H2 molecule:")
    print("    Bond length: 1.4 Bohr")
    print(f"    Grid size: {grid.size} points")
    
    alpha = 1.0
    r1 = np.linalg.norm(grid.points - atcoords[0], axis=1)
    r2 = np.linalg.norm(grid.points - atcoords[1], axis=1)
    
    density = (alpha / np.pi) ** 1.5 * (np.exp(-alpha * r1**2) + np.exp(-alpha * r2**2))
    integral = grid.integrate(density)
    expected = 2.0
    error = abs(integral - expected)
    
    print("\n  Test: Two Gaussians centered at H atoms")
    print(f"    Numerical: {integral:.8f}")
    print(f"    Expected: {expected:.1f}")
    print(f"    Error: {error:.2e}")
    
    passed = error < 1e-5
    print(f"\n  {' Confirmed' if passed else ' Not Confirmed'}")
    
    return passed


def test_grid_comparison():
    print("\n---------------------------")
    print("Test 6: Grid Comparison - Uniform vs Becke")
    print()
    
    alpha = 1.0
    center = np.array([0.0, 0.0, 0.0])
    target_error = 1e-5
    
    print(f"  Target: Integrate (α/π)^(3/2) exp(-α r²) to error < {target_error}")
    print(f"  α = {alpha}")
    
    print("\n  Uniform grid search:")
    uniform_size = None
    for spacing in [1.0, 0.5, 0.4, 0.3, 0.25, 0.2]:
        grid = UniformGrid.from_center(center, extent=8.0, spacing=spacing)
        r2 = np.sum(grid.points**2, axis=1)
        density = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2)
        integral = grid.integrate(density)
        error = abs(integral - 1.0)
        status = "Confirmed " if error < target_error else "Not Confirmed "
        print(f"    {status} spacing={spacing:.2f}: {grid.size:8d} points, error={error:.2e}")
        if error < target_error and uniform_size is None:
            uniform_size = grid.size
    
    print("\n  Becke grid search:")
    becke_size = None
    for n_rad in [20, 30, 40, 50, 60, 75]:
        grid = BeckeGrid.from_atom(atnum=1, center=center, n_rad=n_rad, n_ang=194)
        r2 = np.sum(grid.points**2, axis=1)
        density = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2)
        integral = grid.integrate(density)
        error = abs(integral - 1.0)
        status = "Confirmed" if error < target_error else "Not Confirmed "
        print(f"    {status} n_rad={n_rad:3d}: {grid.size:8d} points, error={error:.2e}")
        if error < target_error and becke_size is None:
            becke_size = grid.size
    
    print("\n  Summary:")
    print(f"    Uniform grid needed: ~{uniform_size:,} points")
    print(f"    Becke grid needed:   ~{becke_size:,} points")
    print(f"    Efficiency ratio:    {uniform_size/becke_size:.1f}x")
    
    return True


def main():
    print("Grid Generation (Uniform and Becke-type)")
    print("---------------------------")
    
    test1 = test_uniform_grid_basic()
    test2 = test_uniform_grid_integration()
    test3 = test_becke_grid_basic()
    test4 = test_becke_grid_integration()
    test5 = test_becke_grid_molecule()
    test6 = test_grid_comparison()
    
    all_passed = test1 and test2 and test3 and test4 and test5 and test6
    
    
    return all_passed


if __name__ == "__main__":
    main()
