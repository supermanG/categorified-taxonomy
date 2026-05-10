"""
cross_validate.py
=================
Cross-validate the Tc class invariants against the v14 candidate pool.

This module:
1. Loads the v14 candidate pool from the rtsc repo
2. Builds the substitution graph
3. Computes class invariants per connected component
4. Checks consistency: does any candidate have predicted Tc exceeding
   the class invariant of its component?
5. Reports inconsistencies for human review

LH & Claude 2026
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from materials_2cat import Material, MaterialsCategory, POINT_GROUPS
from substitution_graph import build_substitution_graph, component_analysis
from class_invariants import compute_class_invariants, verify_tc_bounds


def load_v14_candidates(
    rtsc_path: str = None,
    cache_path: str = None,
) -> List[Material]:
    """
    Load the v14 candidate pool from the rtsc repo.

    Attempts to load from:
    1. Cached JSON (if cache_path exists)
    2. JARVIS dft_3d dataset (if rtsc_path points to the data dir)
    3. A small built-in test set (fallback)
    """
    # Try cached JSON
    if cache_path and os.path.exists(cache_path):
        import json
        with open(cache_path, 'r') as f:
            data = json.load(f)
        materials = []
        for entry in data:
            pg_name = entry.get('point_group', 'C1')
            pg = POINT_GROUPS.get(pg_name, POINT_GROUPS['C1'])
            materials.append(Material(
                material_id=entry.get('jid', entry.get('id', str(len(materials)))),
                composition=entry.get('composition', {}),
                point_group=pg,
                spacegroup=entry.get('spacegroup', 1),
                properties=entry.get('properties', {}),
            ))
        return materials

    # Try JARVIS
    if rtsc_path:
        jarvis_cache = os.path.join(rtsc_path, 'data', 'jarvis',
                                     'dft_3d_2021_cache.json')
        if os.path.exists(jarvis_cache):
            import json
            with open(jarvis_cache, 'r') as f:
                data = json.load(f)
            return _jarvis_to_materials(data)

    # Fallback: built-in test set
    return _builtin_test_set()


def _jarvis_to_materials(data: list) -> List[Material]:
    """Convert JARVIS JSON entries to Material objects."""
    materials = []
    for entry in data:
        comp = {}
        formula = entry.get('formula', '')
        atoms = entry.get('atoms', {})

        if isinstance(atoms, dict) and 'elements' in atoms:
            for el in atoms['elements']:
                comp[el] = comp.get(el, 0) + 1

        pg_name = entry.get('point_group', 'C1')
        pg = POINT_GROUPS.get(pg_name, POINT_GROUPS['C1'])

        props = {}
        for key in ['Tc', 'optB88vdW_bandgap', 'bulk_modulus_kv',
                     'ehull', 'formation_energy_peratom']:
            if key in entry and entry[key] is not None:
                try:
                    props[key] = float(entry[key])
                except (ValueError, TypeError):
                    pass

        if entry.get('Tc') is not None:
            try:
                props['Tc'] = float(entry['Tc'])
            except (ValueError, TypeError):
                pass

        materials.append(Material(
            material_id=entry.get('jid', str(len(materials))),
            composition=comp,
            point_group=pg,
            spacegroup=entry.get('spacegroup_number', 1),
            properties=props,
        ))
    return materials


def _builtin_test_set() -> List[Material]:
    """Small built-in test set for development/testing."""
    from materials_2cat import build_cuprate_example
    cat, _ = build_cuprate_example()
    return list(cat.materials.values())


# ---------------------------------------------------------------------------
# Cross-validation pipeline
# ---------------------------------------------------------------------------

def cross_validate(
    materials: List[Material],
    verbose: bool = True,
) -> Dict:
    """
    Run the full cross-validation pipeline.

    Returns a dict with:
      - n_materials: total materials
      - n_components: number of connected components
      - n_with_tc: materials with known Tc > 0
      - invariants: list of ClassInvariant objects
      - violations: list of violation dicts
      - consistency_rate: fraction of materials consistent with bounds
    """
    if verbose:
        print(f"\nLoaded {len(materials)} materials")

    # Build substitution graph
    cat = build_substitution_graph(materials, same_point_group=True)
    if verbose:
        print(f"Substitution graph: {cat.n_objects} nodes, "
              f"{cat.n_morphisms} edges, {cat.n_components} components")

    # Compute class invariants
    invariants = compute_class_invariants(cat)
    if verbose:
        print(f"Class invariants computed for {len(invariants)} components")

    # Check for violations
    violations = verify_tc_bounds(invariants, cat)
    n_with_tc = sum(1 for m in materials if m.tc and m.tc > 0)
    consistency = 1.0 - len(violations) / max(n_with_tc, 1)

    if verbose:
        print(f"\nResults:")
        print(f"  Materials with Tc > 0: {n_with_tc}")
        print(f"  Violations: {len(violations)}")
        print(f"  Consistency rate: {consistency:.1%}")
        if violations:
            print(f"\n  Top violations:")
            for v in sorted(violations, key=lambda x: -x['excess'])[:10]:
                print(f"    {v['formula']} ({v['point_group']}): "
                      f"Tc={v['tc']:.1f} K > I(C)={v['class_bound']:.1f} K "
                      f"(+{v['excess']:.1f} K)")

    return {
        'n_materials': len(materials),
        'n_components': cat.n_components,
        'n_with_tc': n_with_tc,
        'invariants': invariants,
        'violations': violations,
        'consistency_rate': consistency,
        'category': cat,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print("  Cross-Validation: Tc Class Invariants vs v14")
    print("=" * 60)

    # Try to find rtsc data
    rtsc_candidates = [
        r'C:\Users\superman\rtsc',
        os.path.expanduser('~/rtsc'),
    ]
    rtsc_path = None
    for p in rtsc_candidates:
        if os.path.exists(p):
            rtsc_path = p
            break

    materials = load_v14_candidates(rtsc_path=rtsc_path)
    results = cross_validate(materials, verbose=True)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
