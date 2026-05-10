"""
doping_monotonicity.py
======================
Tc monotonicity analysis along doping paths.

A doping path is a parameterized family M(x) for x in [0, x_max],
connected by doping morphisms at increasing concentration. If Tc were
a strict 2-functor to (R_>=0, <=), then Tc(x) would be monotone in x.
In reality, cuprate Tc(x) is dome-shaped: this module quantifies the
deviation from strict functoriality and computes the lax 2-cell data.

Results:
  1. DopingPath: a sequence of (concentration, Tc) points
  2. Monotonicity analysis: where does Tc increase vs decrease?
  3. Lax deviation: |Tc(x2) - Tc(x1)| when Tc drops along a morphism
  4. Dome fitting: parabolic fit to extract optimal doping x*
  5. Canonical doping curves for cuprate, pnictide, and hydride families

LH & Claude 2026
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from materials_2cat import (
    Material, Morphism, TwoMorphism, MaterialsCategory,
    ProcessType, POINT_GROUPS,
    doping_morphism, identity_morphism,
    TcFunctor,
)


@dataclass
class DopingPoint:
    """A single point on a doping path."""
    concentration: float
    tc: float
    material: Optional[Material] = None
    material_id: str = ""


@dataclass
class DopingPath:
    """
    A parameterized doping path M(x) for x in [x_min, x_max].

    The path is a sequence of doping morphisms:
      M(x_0) -> M(x_1) -> ... -> M(x_n)
    where x_0 < x_1 < ... < x_n and each arrow is a doping morphism.
    """
    family_name: str
    parent_formula: str
    dopant: str
    host: str
    points: List[DopingPoint] = field(default_factory=list)
    point_group: str = ""
    pairing_symmetry: str = ""

    def add_point(self, x: float, tc: float,
                  material: Material = None, mid: str = ""):
        self.points.append(DopingPoint(x, tc, material, mid))
        self.points.sort(key=lambda p: p.concentration)

    @property
    def concentrations(self) -> np.ndarray:
        return np.array([p.concentration for p in self.points])

    @property
    def tc_values(self) -> np.ndarray:
        return np.array([p.tc for p in self.points])

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def tc_max(self) -> float:
        return float(max(p.tc for p in self.points)) if self.points else 0.0

    @property
    def x_optimal(self) -> float:
        """Concentration at which Tc is maximized."""
        if not self.points:
            return 0.0
        best = max(self.points, key=lambda p: p.tc)
        return best.concentration


# ---------------------------------------------------------------------------
# Monotonicity analysis
# ---------------------------------------------------------------------------

@dataclass
class MonotonicityResult:
    """Result of monotonicity analysis on a doping path."""
    path: DopingPath
    n_increasing: int = 0
    n_decreasing: int = 0
    n_constant: int = 0
    total_increase: float = 0.0
    total_decrease: float = 0.0
    max_single_drop: float = 0.0
    max_single_rise: float = 0.0
    monotone_up_to: float = 0.0
    lax_deviation: float = 0.0
    dome_peak_x: float = 0.0
    dome_peak_tc: float = 0.0
    segments: List[Dict] = field(default_factory=list)


def analyze_monotonicity(path: DopingPath, tol: float = 0.5) -> MonotonicityResult:
    """
    Analyze Tc monotonicity along a doping path.

    Parameters
    ----------
    path : DopingPath
    tol : tolerance in K for treating Tc changes as "constant"

    Returns
    -------
    MonotonicityResult with segment-by-segment analysis
    """
    result = MonotonicityResult(path=path)

    if path.n_points < 2:
        return result

    xs = path.concentrations
    tcs = path.tc_values

    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        tc1, tc2 = tcs[i], tcs[i + 1]
        delta_tc = tc2 - tc1
        delta_x = x2 - x1

        seg = {
            'x_start': float(x1),
            'x_end': float(x2),
            'tc_start': float(tc1),
            'tc_end': float(tc2),
            'delta_tc': float(delta_tc),
            'slope': float(delta_tc / delta_x) if delta_x > 0 else 0.0,
        }

        if abs(delta_tc) < tol:
            seg['direction'] = 'constant'
            result.n_constant += 1
        elif delta_tc > 0:
            seg['direction'] = 'increasing'
            result.n_increasing += 1
            result.total_increase += delta_tc
            result.max_single_rise = max(result.max_single_rise, delta_tc)
        else:
            seg['direction'] = 'decreasing'
            result.n_decreasing += 1
            result.total_decrease += abs(delta_tc)
            result.max_single_drop = max(result.max_single_drop, abs(delta_tc))

        result.segments.append(seg)

    # Monotone-up-to: largest x such that Tc is non-decreasing from x=0
    result.monotone_up_to = xs[0]
    running_max_tc = tcs[0]
    for i in range(1, len(xs)):
        if tcs[i] >= running_max_tc - tol:
            result.monotone_up_to = xs[i]
            running_max_tc = max(running_max_tc, tcs[i])
        else:
            break

    # Lax deviation: max drop below a previous Tc value along the path
    peak_so_far = tcs[0]
    result.lax_deviation = 0.0
    for i in range(1, len(tcs)):
        peak_so_far = max(peak_so_far, tcs[i])
        drop = peak_so_far - tcs[i]
        result.lax_deviation = max(result.lax_deviation, drop)

    # Dome peak
    peak_idx = int(np.argmax(tcs))
    result.dome_peak_x = float(xs[peak_idx])
    result.dome_peak_tc = float(tcs[peak_idx])

    return result


# ---------------------------------------------------------------------------
# Dome fitting
# ---------------------------------------------------------------------------

def fit_dome(path: DopingPath) -> Dict:
    """
    Fit a parabolic dome Tc(x) = a*(x - x0)^2 + Tc_max to the doping path.

    Returns dict with: x0, Tc_max, a (curvature), r_squared, residuals.
    """
    if path.n_points < 3:
        return {'x0': path.x_optimal, 'Tc_max': path.tc_max,
                'a': 0.0, 'r_squared': 0.0, 'fit_ok': False}

    xs = path.concentrations
    tcs = path.tc_values

    # Fit quadratic: Tc = a*x^2 + b*x + c
    coeffs = np.polyfit(xs, tcs, 2)
    a, b, c = coeffs

    # Vertex of parabola: x0 = -b/(2a), Tc_max = c - b^2/(4a)
    if abs(a) > 1e-12:
        x0 = -b / (2 * a)
        tc_max = c - b**2 / (4 * a)
    else:
        x0 = path.x_optimal
        tc_max = path.tc_max

    # R-squared
    tc_pred = np.polyval(coeffs, xs)
    ss_res = np.sum((tcs - tc_pred)**2)
    ss_tot = np.sum((tcs - np.mean(tcs))**2)
    r_sq = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        'x0': float(x0),
        'Tc_max': float(tc_max),
        'a': float(a),
        'b': float(b),
        'c': float(c),
        'r_squared': float(r_sq),
        'residuals': (tcs - tc_pred).tolist(),
        'fit_ok': a < 0,
    }


# ---------------------------------------------------------------------------
# Lax 2-cell computation
# ---------------------------------------------------------------------------

def compute_lax_cells(path: DopingPath) -> List[Dict]:
    """
    Compute the lax 2-cells needed to make Tc a lax 2-functor
    along the doping path.

    For each pair of consecutive doping morphisms d_i: M(x_i) -> M(x_{i+1})
    where Tc drops, the strict 2-functor condition fails. The lax 2-cell
    epsilon_i has magnitude |Tc(x_{i+1}) - Tc(x_i)| and represents the
    "prediction error" of the strict framework.
    """
    cells = []
    if path.n_points < 2:
        return cells

    xs = path.concentrations
    tcs = path.tc_values

    for i in range(len(xs) - 1):
        delta_tc = tcs[i + 1] - tcs[i]
        if delta_tc < 0:
            cells.append({
                'index': i,
                'x_start': float(xs[i]),
                'x_end': float(xs[i + 1]),
                'tc_start': float(tcs[i]),
                'tc_end': float(tcs[i + 1]),
                'magnitude': float(abs(delta_tc)),
                'relative': float(abs(delta_tc) / max(tcs[i], 0.01)),
            })

    return cells


# ---------------------------------------------------------------------------
# Canonical doping paths (experimental data)
# ---------------------------------------------------------------------------

def canonical_cuprate_paths() -> List[DopingPath]:
    """
    Canonical cuprate doping paths from experimental literature.

    Returns DopingPath objects with measured Tc(x) data for:
    - La_{2-x}Sr_xCuO4 (LSCO)
    - La_{2-x}Ba_xCuO4 (LBCO)
    - YBa2Cu3O_{7-delta} (YBCO, oxygen doping)
    - Bi2Sr2CaCu2O_{8+delta} (BSCCO, oxygen doping)
    """
    paths = []

    # LSCO: La_{2-x}Sr_xCuO4
    lsco = DopingPath(
        family_name="LSCO",
        parent_formula="La2CuO4",
        dopant="Sr", host="La",
        point_group="D4h",
        pairing_symmetry="d-wave",
    )
    # Experimental Tc(x) from Takagi et al. (1989) and reviews
    for x, tc in [
        (0.00, 0.0), (0.05, 10.0), (0.07, 20.0), (0.10, 30.0),
        (0.12, 34.0), (0.15, 38.0), (0.18, 35.0), (0.20, 28.0),
        (0.22, 20.0), (0.25, 12.0), (0.30, 0.0),
    ]:
        lsco.add_point(x, tc)
    paths.append(lsco)

    # LBCO: La_{2-x}Ba_xCuO4
    lbco = DopingPath(
        family_name="LBCO",
        parent_formula="La2CuO4",
        dopant="Ba", host="La",
        point_group="D4h",
        pairing_symmetry="d-wave",
    )
    for x, tc in [
        (0.00, 0.0), (0.05, 8.0), (0.08, 18.0), (0.10, 26.0),
        (0.12, 30.0), (0.125, 4.0),
        (0.13, 22.0), (0.15, 25.0), (0.18, 15.0), (0.20, 5.0),
    ]:
        lbco.add_point(x, tc)
    paths.append(lbco)

    # YBCO: YBa2Cu3O_{7-delta}, parameterized by hole doping p
    ybco = DopingPath(
        family_name="YBCO",
        parent_formula="YBa2Cu3O7",
        dopant="O_holes", host="chain",
        point_group="D2h",
        pairing_symmetry="d-wave",
    )
    for p, tc in [
        (0.05, 10.0), (0.08, 40.0), (0.10, 60.0), (0.12, 80.0),
        (0.16, 92.0), (0.19, 88.0), (0.22, 70.0), (0.25, 40.0),
        (0.27, 10.0),
    ]:
        ybco.add_point(p, tc)
    paths.append(ybco)

    # BSCCO-2212: Bi2Sr2CaCu2O_{8+delta}
    bscco = DopingPath(
        family_name="BSCCO-2212",
        parent_formula="Bi2Sr2CaCu2O8",
        dopant="O_excess", host="BiO",
        point_group="D4h",
        pairing_symmetry="d-wave",
    )
    for p, tc in [
        (0.08, 30.0), (0.10, 55.0), (0.12, 72.0), (0.14, 82.0),
        (0.16, 85.0), (0.18, 80.0), (0.20, 65.0), (0.22, 40.0),
    ]:
        bscco.add_point(p, tc)
    paths.append(bscco)

    return paths


def canonical_pnictide_paths() -> List[DopingPath]:
    """Iron pnictide doping paths."""
    paths = []

    # Ba(Fe_{1-x}Co_x)2As2 (Ba-122)
    ba122 = DopingPath(
        family_name="Ba122-Co",
        parent_formula="BaFe2As2",
        dopant="Co", host="Fe",
        point_group="D4h",
        pairing_symmetry="s+-wave",
    )
    for x, tc in [
        (0.00, 0.0), (0.03, 10.0), (0.05, 18.0), (0.07, 24.0),
        (0.08, 25.0), (0.10, 22.0), (0.12, 15.0), (0.15, 5.0),
        (0.20, 0.0),
    ]:
        ba122.add_point(x, tc)
    paths.append(ba122)

    return paths


def canonical_hydride_paths() -> List[DopingPath]:
    """Hydride pressure paths (modeled as doping for analysis)."""
    paths = []

    # LaH10 under pressure (GPa used as "concentration" parameter)
    lah10 = DopingPath(
        family_name="LaH10",
        parent_formula="LaH10",
        dopant="pressure", host="lattice",
        point_group="Oh",
        pairing_symmetry="s-wave",
    )
    for p, tc in [
        (130.0, 200.0), (150.0, 250.0), (170.0, 260.0),
        (190.0, 250.0), (210.0, 220.0), (250.0, 170.0),
    ]:
        lah10.add_point(p, tc)
    paths.append(lah10)

    return paths


# ---------------------------------------------------------------------------
# Build doping paths in the 2-category
# ---------------------------------------------------------------------------

def build_doping_category(path: DopingPath) -> Tuple[MaterialsCategory, TcFunctor]:
    """
    Build a MaterialsCategory from a DopingPath, with doping morphisms
    connecting consecutive points.
    """
    cat = MaterialsCategory()
    pg = POINT_GROUPS.get(path.point_group, POINT_GROUPS['C1'])

    materials = []
    for i, pt in enumerate(path.points):
        mid = f"{path.family_name}_x{pt.concentration:.3f}"
        mat = Material(
            material_id=mid,
            composition={path.host: 2 - pt.concentration,
                         path.dopant: pt.concentration},
            point_group=pg,
            properties={'Tc': pt.tc},
        )
        cat.add_material(mat)
        materials.append(mat)

    for i in range(len(materials) - 1):
        m = doping_morphism(
            materials[i], materials[i + 1],
            path.dopant, path.host,
            path.points[i + 1].concentration,
        )
        cat.add_morphism(m)

    tc_func = TcFunctor(cat)
    return cat, tc_func


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def full_doping_analysis(verbose: bool = True) -> Dict:
    """
    Run the complete doping monotonicity analysis on all canonical paths.

    Returns a summary dict with per-path results.
    """
    all_paths = (
        canonical_cuprate_paths()
        + canonical_pnictide_paths()
        + canonical_hydride_paths()
    )

    results = {}
    for path in all_paths:
        mono = analyze_monotonicity(path)
        dome = fit_dome(path)
        lax = compute_lax_cells(path)

        cat, tc_func = build_doping_category(path)
        class_inv = tc_func.class_invariant(
            set(cat.materials.keys())
        )

        results[path.family_name] = {
            'path': path,
            'monotonicity': mono,
            'dome_fit': dome,
            'lax_cells': lax,
            'class_invariant': class_inv,
            'category': cat,
        }

        if verbose:
            _print_path_analysis(path, mono, dome, lax, class_inv)

    if verbose:
        _print_summary(results)

    return results


def _print_path_analysis(path, mono, dome, lax, class_inv):
    """Print analysis for a single doping path."""
    print(f"\n{'=' * 60}")
    print(f"  {path.family_name}: {path.parent_formula}")
    print(f"  Dopant: {path.dopant} -> {path.host}")
    print(f"  Point group: {path.point_group}, "
          f"Pairing: {path.pairing_symmetry}")
    print(f"{'=' * 60}")

    print(f"\n  Doping path ({path.n_points} points):")
    print(f"    x range: [{path.concentrations[0]:.3f}, "
          f"{path.concentrations[-1]:.3f}]")
    print(f"    Tc range: [{min(path.tc_values):.1f}, "
          f"{max(path.tc_values):.1f}] K")
    print(f"    Class invariant I(C) = {class_inv:.1f} K")

    print(f"\n  Monotonicity analysis:")
    n_total = mono.n_increasing + mono.n_decreasing + mono.n_constant
    print(f"    Segments: {n_total} total")
    print(f"      Increasing: {mono.n_increasing}")
    print(f"      Decreasing: {mono.n_decreasing}")
    print(f"      Constant:   {mono.n_constant}")
    print(f"    Total rise:   {mono.total_increase:7.1f} K")
    print(f"    Total drop:   {mono.total_decrease:7.1f} K")
    print(f"    Max single drop: {mono.max_single_drop:.1f} K")

    is_mono = mono.n_decreasing == 0
    print(f"\n  Strict 2-functor: {'COMPATIBLE' if is_mono else 'FAILS'}")
    if not is_mono:
        print(f"    Monotone up to x = {mono.monotone_up_to:.3f}")
        print(f"    Lax deviation: {mono.lax_deviation:.1f} K "
              f"({mono.lax_deviation / max(mono.dome_peak_tc, 0.01) * 100:.1f}%"
              f" of Tc_max)")

    if dome['fit_ok']:
        print(f"\n  Dome fit (parabolic):")
        print(f"    x* = {dome['x0']:.3f}  "
              f"(optimal doping)")
        print(f"    Tc_max = {dome['Tc_max']:.1f} K")
        print(f"    Curvature a = {dome['a']:.1f}")
        print(f"    R^2 = {dome['r_squared']:.4f}")

    if lax:
        print(f"\n  Lax 2-cells ({len(lax)} needed):")
        for cell in lax:
            print(f"    x=[{cell['x_start']:.3f},{cell['x_end']:.3f}]: "
                  f"Tc drops {cell['tc_start']:.1f} -> {cell['tc_end']:.1f} K "
                  f"(|eps| = {cell['magnitude']:.1f} K, "
                  f"{cell['relative'] * 100:.1f}%)")


def _print_summary(results):
    """Print cross-family summary."""
    print(f"\n{'=' * 60}")
    print(f"  CROSS-FAMILY SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  {'Family':<16} {'Tc_max':>7} {'x*':>7} {'I(C)':>7} "
          f"{'Mono?':>6} {'Lax_dev':>8} {'#Lax':>5}")
    print(f"  {'-' * 60}")

    for name, r in sorted(results.items(),
                          key=lambda kv: -kv[1]['monotonicity'].dome_peak_tc):
        mono = r['monotonicity']
        is_mono = mono.n_decreasing == 0
        print(f"  {name:<16} {mono.dome_peak_tc:>7.1f} "
              f"{mono.dome_peak_x:>7.3f} "
              f"{r['class_invariant']:>7.1f} "
              f"{'yes' if is_mono else 'NO':>6} "
              f"{mono.lax_deviation:>8.1f} "
              f"{len(r['lax_cells']):>5}")

    n_strict = sum(1 for r in results.values()
                   if r['monotonicity'].n_decreasing == 0)
    n_lax = sum(1 for r in results.values()
                if r['monotonicity'].n_decreasing > 0)
    total_cells = sum(len(r['lax_cells']) for r in results.values())
    max_dev = max(r['monotonicity'].lax_deviation
                  for r in results.values())

    print(f"\n  Strict 2-functor compatible: {n_strict}/{len(results)} paths")
    print(f"  Require lax extension: {n_lax}/{len(results)} paths")
    print(f"  Total lax 2-cells needed: {total_cells}")
    print(f"  Maximum lax deviation: {max_dev:.1f} K")
    print(f"\n  Conclusion: Tc is NOT a strict 2-functor along doping paths.")
    print(f"  The lax 2-functor Tc^lax with comparison 2-cells of max")
    print(f"  magnitude {max_dev:.1f} K provides the correct framework.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print("  Tc Monotonicity Analysis Along Doping Paths")
    print("=" * 60)

    results = full_doping_analysis(verbose=True)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
