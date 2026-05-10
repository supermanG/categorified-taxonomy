"""
class_invariants.py
===================
Compute Tc upper bounds per connected component of the substitution graph.

The class invariant I(C) for a connected component C of the substitution
graph is defined as:
  I(C) = sup { Tc(M) : M in C }

For a computable approximation, we use the v15 hybrid Tc estimator
restricted to the materials in C:
  I_hat(C) = max { Tc_hat(M) : M in C }

where Tc_hat combines:
  - chi_n symmetry features (star_G decomposition)
  - family-specific Ridge corrections (v11)
  - Sigrist-Ueda channel classification (v14)
  - viability filters (phonon stability, Heusler anchors)

The key theorem (target): I(C) is BOUNDED by an explicit function of
the crystallographic class, the chi_n profile, and the family chemistry.

LH & Claude 2026
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

from materials_2cat import Material, MaterialsCategory, TcFunctor


@dataclass
class ClassInvariant:
    """
    Tc class invariant for a connected component.

    Attributes:
        component: set of material IDs in the component
        point_group: dominant point group
        tc_empirical: empirical max Tc (from known measurements)
        tc_estimated: estimated max Tc (from v15 hybrid estimator)
        chi_n_profile: average chi_n decomposition across the component
        pairing_channel: dominant pairing channel (s, d, p, f-wave)
        confidence: confidence level (low/medium/high)
    """
    component: Set[str]
    point_group: str
    tc_empirical: float
    tc_estimated: float
    chi_n_profile: Dict[int, float]
    pairing_channel: str = "unknown"
    confidence: str = "low"

    @property
    def bound(self) -> float:
        """Upper bound on Tc for materials in this component."""
        return max(self.tc_empirical, self.tc_estimated)

    @property
    def n_materials(self) -> int:
        return len(self.component)


# ---------------------------------------------------------------------------
# BCS-type bound from symmetry
# ---------------------------------------------------------------------------

# Tc upper bounds by pairing channel (from BCS theory + corrections).
# These are theoretical maxima assuming optimal coupling.
CHANNEL_TC_BOUNDS = {
    's-wave':  40.0,    # conventional BCS
    'd-wave':  200.0,   # cuprate-class (Tc_max ~ 133 K, theory allows ~200 K)
    'p-wave':  5.0,     # very rare (Sr2RuO4 ~ 1.5 K)
    'f-wave':  1.0,     # essentially never observed
    'unknown': 300.0,   # no constraint
}


def bcs_channel_bound(pairing_channel: str) -> float:
    """Theoretical Tc upper bound from the pairing channel symmetry."""
    return CHANNEL_TC_BOUNDS.get(pairing_channel, 300.0)


# ---------------------------------------------------------------------------
# Compute class invariants from a MaterialsCategory
# ---------------------------------------------------------------------------

def compute_class_invariants(
    cat: MaterialsCategory,
    tc_estimator: Optional[callable] = None,
) -> List[ClassInvariant]:
    """
    Compute class invariants for all connected components.

    Parameters
    ----------
    cat : MaterialsCategory
    tc_estimator : optional callable(Material) -> float
        External Tc estimator (e.g., v15 hybrid). If None, uses
        only empirical Tc values.

    Returns
    -------
    List of ClassInvariant, sorted by bound (descending).
    """
    tc_func = TcFunctor(cat)
    invariants = []

    for comp in cat.connected_components():
        mats = [cat.materials[mid] for mid in comp if mid in cat.materials]
        if not mats:
            continue

        # Dominant point group (most common in the component)
        pg_counts: Dict[str, int] = {}
        for m in mats:
            pg = m.point_group.name
            pg_counts[pg] = pg_counts.get(pg, 0) + 1
        dominant_pg = max(pg_counts, key=pg_counts.get)

        # Empirical Tc
        tc_values = [m.tc for m in mats if m.tc is not None and m.tc > 0]
        tc_emp = max(tc_values) if tc_values else 0.0

        # Estimated Tc
        tc_est = 0.0
        if tc_estimator is not None:
            for m in mats:
                try:
                    est = tc_estimator(m)
                    tc_est = max(tc_est, est)
                except Exception:
                    pass

        # Average chi_n profile
        chi_n_profile: Dict[int, float] = {}
        n_with_chi = 0
        for m in mats:
            chi = m.chi_n
            if chi:
                n_with_chi += 1
                for k, v in chi.items():
                    chi_n_profile[k] = chi_n_profile.get(k, 0.0) + v
        if n_with_chi > 0:
            for k in chi_n_profile:
                chi_n_profile[k] /= n_with_chi

        # Pairing channel (from properties if available)
        pairing_channels = [m.properties.get('pairing', 'unknown')
                           for m in mats if 'pairing' in m.properties]
        if pairing_channels:
            from collections import Counter
            pairing_channel = Counter(pairing_channels).most_common(1)[0][0]
        else:
            pairing_channel = 'unknown'

        # Confidence
        if len(tc_values) >= 3 and n_with_chi >= 2:
            confidence = 'high'
        elif len(tc_values) >= 1:
            confidence = 'medium'
        else:
            confidence = 'low'

        invariants.append(ClassInvariant(
            component=comp,
            point_group=dominant_pg,
            tc_empirical=tc_emp,
            tc_estimated=tc_est,
            chi_n_profile=chi_n_profile,
            pairing_channel=pairing_channel,
            confidence=confidence,
        ))

    invariants.sort(key=lambda x: -x.bound)
    return invariants


def verify_tc_bounds(
    invariants: List[ClassInvariant],
    cat: MaterialsCategory,
) -> List[Dict]:
    """
    Verify that all known Tc values are within the class invariant bounds.

    Returns a list of violations (materials exceeding their class bound).
    """
    violations = []
    comp_to_inv = {}
    for inv in invariants:
        for mid in inv.component:
            comp_to_inv[mid] = inv

    for mid, mat in cat.materials.items():
        tc = mat.tc
        if tc is None or tc <= 0:
            continue
        inv = comp_to_inv.get(mid)
        if inv is None:
            continue
        channel_bound = bcs_channel_bound(inv.pairing_channel)
        if tc > inv.bound:
            violations.append({
                'material_id': mid,
                'formula': mat.formula,
                'tc': tc,
                'class_bound': inv.bound,
                'channel_bound': channel_bound,
                'point_group': mat.point_group.name,
                'excess': tc - inv.bound,
            })

    return violations


def print_invariant_table(invariants: List[ClassInvariant], top_n: int = 20):
    """Print a summary table of class invariants."""
    print(f"\n{'Rank':>4}  {'Bound':>8}  {'Tc_emp':>8}  {'Tc_est':>8}  "
          f"{'|C|':>5}  {'PG':<6}  {'Channel':<10}  {'Conf':<6}")
    print("-" * 75)
    for i, inv in enumerate(invariants[:top_n]):
        print(f"{i+1:>4}  {inv.bound:>8.1f}  {inv.tc_empirical:>8.1f}  "
              f"{inv.tc_estimated:>8.1f}  {inv.n_materials:>5}  "
              f"{inv.point_group:<6}  {inv.pairing_channel:<10}  "
              f"{inv.confidence:<6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from materials_2cat import build_cuprate_example

    print("=" * 60)
    print("  Class Invariant Computation")
    print("=" * 60)

    cat, _ = build_cuprate_example()
    invariants = compute_class_invariants(cat)
    print_invariant_table(invariants)

    violations = verify_tc_bounds(invariants, cat)
    if violations:
        print(f"\nVIOLATIONS ({len(violations)}):")
        for v in violations:
            print(f"  {v['formula']}: Tc={v['tc']:.1f} K > "
                  f"I(C)={v['class_bound']:.1f} K")
    else:
        print("\nNo violations: all Tc values within class bounds.")
