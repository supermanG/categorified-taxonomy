#!/usr/bin/env python3
"""
build_full_graph.py
===================
Build the full substitution graph from JARVIS dft_3d + supercon data.

This is the main computational script that:
1. Loads JARVIS materials
2. Builds the substitution graph
3. Computes connected components per crystallographic class
4. Computes Tc class invariants
5. Reports statistics and identifies key structural classes

LH & Claude 2026
"""

import sys
import os
import time
import json
from collections import Counter
from typing import List, Dict

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from materials_2cat import Material, MaterialsCategory, POINT_GROUPS, TcFunctor
from substitution_graph import (
    build_substitution_graph, component_analysis, print_component_table,
    are_substitutable, SUBSTITUTION_FAMILIES,
)
from class_invariants import (
    compute_class_invariants, verify_tc_bounds, print_invariant_table,
    bcs_channel_bound,
)
from jarvis_loader import load_all_jarvis, spacegroup_to_point_group


def main():
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 70)
    print("  Full-Scale Substitution Graph Construction")
    print("  (JARVIS dft_3d + supercon_3d)")
    print("=" * 70)

    # -- Locate data --
    data_dirs = [
        r'C:\Users\superman\rtsc\data\jarvis',
        os.path.expanduser('~/rtsc/data/jarvis'),
    ]
    data_dir = None
    for d in data_dirs:
        if os.path.exists(d):
            data_dir = d
            break

    if data_dir is None:
        print("ERROR: JARVIS data directory not found.")
        print("Expected at: C:\\Users\\superman\\rtsc\\data\\jarvis\\")
        return

    # -- Load data --
    t0 = time.time()
    materials, supercon = load_all_jarvis(
        data_dir, max_materials=None, metals_only=True
    )
    t_load = time.time() - t0
    print(f"\nLoad time: {t_load:.1f}s")

    n_with_tc = sum(1 for m in materials if m.tc and m.tc > 0)
    print(f"Total materials: {len(materials)}")
    print(f"With Tc > 0: {n_with_tc}")

    # -- Build substitution graph --
    print("\n" + "-" * 70)
    print("Building substitution graph (same point group)...")
    t1 = time.time()
    cat = build_substitution_graph(materials, same_point_group=True)
    t_graph = time.time() - t1
    print(f"Graph: {cat.n_objects} nodes, {cat.n_morphisms} edges, "
          f"{cat.n_components} components ({t_graph:.1f}s)")

    # -- Component analysis --
    print("\n" + "-" * 70)
    print("Analyzing connected components...")
    results = component_analysis(cat)

    print(f"\nTotal components: {len(results)}")
    print(f"Components with Tc > 0: "
          f"{sum(1 for r in results if r['class_invariant'] > 0)}")
    print(f"Components with |C| >= 5: "
          f"{sum(1 for r in results if r['n_materials'] >= 5)}")
    print(f"Components with |C| >= 10: "
          f"{sum(1 for r in results if r['n_materials'] >= 10)}")

    # -- Top components by class invariant --
    print("\n" + "-" * 70)
    print("TOP COMPONENTS BY CLASS INVARIANT (Tc upper bound):")
    print_component_table(results, top_n=30)

    # -- Class invariants --
    print("\n" + "-" * 70)
    print("Computing class invariants...")
    invariants = compute_class_invariants(cat)
    print_invariant_table(invariants, top_n=20)

    # -- Violations check --
    violations = verify_tc_bounds(invariants, cat)
    print(f"\nViolation check: {len(violations)} violations out of "
          f"{n_with_tc} materials with Tc")
    if violations:
        print("  (violations indicate materials whose Tc exceeds their "
              "component's known max; this should not happen by construction)")

    # -- Per-point-group statistics --
    print("\n" + "-" * 70)
    print("PER-POINT-GROUP SUMMARY:")
    print(f"{'PG':>6}  {'n_mat':>6}  {'n_comp':>6}  {'n_Tc':>5}  "
          f"{'max_Tc':>8}  {'avg_comp':>8}  {'max_comp':>8}")
    print("-" * 65)

    pg_stats: Dict[str, Dict] = {}
    for r in results:
        for pg in r['point_groups']:
            if pg not in pg_stats:
                pg_stats[pg] = {
                    'n_mat': 0, 'n_comp': 0, 'n_tc': 0,
                    'max_tc': 0.0, 'comp_sizes': [],
                }
            pg_stats[pg]['n_mat'] += r['n_materials']
            pg_stats[pg]['n_comp'] += 1
            pg_stats[pg]['n_tc'] += r['n_with_tc']
            pg_stats[pg]['max_tc'] = max(
                pg_stats[pg]['max_tc'], r['class_invariant']
            )
            pg_stats[pg]['comp_sizes'].append(r['n_materials'])

    for pg in sorted(pg_stats.keys(),
                     key=lambda x: -pg_stats[x]['max_tc']):
        s = pg_stats[pg]
        avg_comp = sum(s['comp_sizes']) / max(len(s['comp_sizes']), 1)
        max_comp = max(s['comp_sizes']) if s['comp_sizes'] else 0
        print(f"{pg:>6}  {s['n_mat']:>6}  {s['n_comp']:>6}  "
              f"{s['n_tc']:>5}  {s['max_tc']:>8.2f}  "
              f"{avg_comp:>8.1f}  {max_comp:>8}")

    # -- Key findings --
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)

    top_tc = results[0] if results else None
    if top_tc:
        print(f"\n  Highest class invariant: I(C) = {top_tc['class_invariant']:.2f} K")
        print(f"  Component size: {top_tc['n_materials']} materials")
        print(f"  Point groups: {top_tc['point_groups']}")
        print(f"  Formulas (first 5): {top_tc['formulas'][:5]}")

    large_comps = [r for r in results if r['n_materials'] >= 10]
    print(f"\n  Large components (|C| >= 10): {len(large_comps)}")

    sc_comps = [r for r in results
                if r['class_invariant'] > 0 and r['n_materials'] >= 3]
    print(f"  SC components (Tc > 0, |C| >= 3): {len(sc_comps)}")

    # -- Save results --
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'n_materials': len(materials),
        'n_with_tc': n_with_tc,
        'n_components': len(results),
        'n_violations': len(violations),
        'top_components': [
            {
                'rank': i + 1,
                'class_invariant': r['class_invariant'],
                'n_materials': r['n_materials'],
                'n_with_tc': r['n_with_tc'],
                'point_groups': list(r['point_groups']),
                'formulas': r['formulas'][:10],
                'tc_spread': r['tc_spread'],
            }
            for i, r in enumerate(results[:50])
        ],
        'pg_stats': {
            pg: {
                'n_materials': s['n_mat'],
                'n_components': s['n_comp'],
                'n_with_tc': s['n_tc'],
                'max_tc': s['max_tc'],
            }
            for pg, s in pg_stats.items()
        },
    }

    summary_path = os.path.join(output_dir, 'substitution_graph_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    print("\n" + "=" * 70)
    print(f"  Total time: {time.time() - t0:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
