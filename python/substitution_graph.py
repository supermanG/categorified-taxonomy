"""
substitution_graph.py
=====================
Build and analyze the substitution graph for crystalline materials.

The substitution graph is the 1-skeleton of the 2-category M restricted
to substitution and doping morphisms. Its connected components define
the "structural classes" within which Tc is bounded by the class
invariant.

Key operations:
  - Build the graph from a materials database (JARVIS, ICSD, etc.)
  - Compute connected components per crystallographic class
  - Extract Tc bounds per component
  - Enumerate substitution paths between materials

LH & Claude 2026
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from materials_2cat import (
    Material, MaterialsCategory, Morphism, ProcessType,
    PointGroup, POINT_GROUPS,
    substitution_morphism, doping_morphism,
    TcFunctor,
)


# ---------------------------------------------------------------------------
# Element substitution rules
# ---------------------------------------------------------------------------

SUBSTITUTION_FAMILIES = {
    'alkali':       ['Li', 'Na', 'K', 'Rb', 'Cs'],
    'alkaline':     ['Be', 'Mg', 'Ca', 'Sr', 'Ba'],
    'transition_3d': ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'],
    'transition_4d': ['Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag'],
    'transition_5d': ['Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au'],
    'rare_earth':   ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
                     'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y'],
    'pnictide':     ['N', 'P', 'As', 'Sb', 'Bi'],
    'chalcogen':    ['O', 'S', 'Se', 'Te'],
    'halogen':      ['F', 'Cl', 'Br', 'I'],
    'post_trans':   ['Al', 'Ga', 'In', 'Tl'],
    'carbon_group': ['C', 'Si', 'Ge', 'Sn', 'Pb'],
}

ELEMENT_TO_FAMILY: Dict[str, str] = {}
for family, elements in SUBSTITUTION_FAMILIES.items():
    for el in elements:
        ELEMENT_TO_FAMILY[el] = family


def are_substitutable(el_a: str, el_b: str) -> bool:
    """Check if two elements are substitutable (same chemical family)."""
    if el_a == el_b:
        return False
    fam_a = ELEMENT_TO_FAMILY.get(el_a)
    fam_b = ELEMENT_TO_FAMILY.get(el_b)
    if fam_a is None or fam_b is None:
        return False
    return fam_a == fam_b


def substitutable_pairs(comp: Dict[str, float]) -> List[Tuple[str, str]]:
    """List all substitutable element pairs in a composition."""
    elements = list(comp.keys())
    pairs = []
    for el in elements:
        fam = ELEMENT_TO_FAMILY.get(el)
        if fam is None:
            continue
        for other in SUBSTITUTION_FAMILIES.get(fam, []):
            if other != el:
                pairs.append((el, other))
    return pairs


# ---------------------------------------------------------------------------
# Build substitution graph from materials list
# ---------------------------------------------------------------------------

def _composition_signature(comp: Dict[str, float],
                           ignore_element: str = None) -> str:
    """
    Composition signature invariant under single-element substitution.

    Replace the substituted element with a wildcard '*', keeping
    stoichiometry. Materials with matching signatures are connected
    by a substitution morphism.
    """
    parts = []
    for el in sorted(comp.keys()):
        count = comp[el]
        label = '*' if el == ignore_element else el
        if count == int(count):
            parts.append(f"{label}{int(count)}")
        else:
            parts.append(f"{label}{count:.2f}")
    return "_".join(parts)


def build_substitution_graph(
    materials: List[Material],
    same_point_group: bool = True,
    same_spacegroup: bool = False,
) -> MaterialsCategory:
    """
    Build the substitution graph from a list of materials.

    Two materials A, B are connected by a substitution morphism if:
    1. They share the same point group (if same_point_group=True)
    2. They share the same spacegroup (if same_spacegroup=True)
    3. Their compositions differ by exactly one element substitution
       within the same chemical family

    Returns a MaterialsCategory with all identified morphisms.
    """
    cat = MaterialsCategory()
    for mat in materials:
        cat.add_material(mat)

    sig_index: Dict[str, List[Material]] = defaultdict(list)

    for mat in materials:
        for el in mat.composition:
            sig = _composition_signature(mat.composition, ignore_element=el)
            key = sig
            if same_point_group:
                key = f"{mat.point_group.name}|{sig}"
            if same_spacegroup:
                key = f"{mat.spacegroup}|{sig}"
            sig_index[key].append((mat, el))

    n_morphisms = 0
    for key, group in sig_index.items():
        for i in range(len(group)):
            mat_i, el_i = group[i]
            for j in range(i + 1, len(group)):
                mat_j, el_j = group[j]
                if el_i == el_j:
                    continue
                if not are_substitutable(el_i, el_j):
                    continue
                morph = substitution_morphism(mat_i, mat_j, el_i, el_j)
                cat.add_morphism(morph)
                n_morphisms += 1

    return cat


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def component_analysis(cat: MaterialsCategory) -> List[Dict]:
    """
    Analyze connected components of the substitution graph.

    Returns a list of dicts, one per component, with:
      - materials: list of material_ids
      - point_groups: set of point group names
      - tc_max: maximum known Tc in the component
      - tc_min: minimum known Tc > 0
      - tc_spread: max - min
      - formulas: list of chemical formulas
    """
    tc_func = TcFunctor(cat)
    results = []

    for comp in cat.connected_components():
        mats = [cat.materials[mid] for mid in comp if mid in cat.materials]
        pgs = set(m.point_group.name for m in mats)
        tc_values = [m.tc for m in mats if m.tc is not None and m.tc > 0]

        results.append({
            'materials': list(comp),
            'point_groups': pgs,
            'tc_max': max(tc_values) if tc_values else 0.0,
            'tc_min': min(tc_values) if tc_values else 0.0,
            'tc_spread': (max(tc_values) - min(tc_values))
                          if len(tc_values) >= 2 else 0.0,
            'formulas': [m.formula for m in mats],
            'n_materials': len(mats),
            'n_with_tc': len(tc_values),
            'class_invariant': tc_func.class_invariant(comp),
        })

    results.sort(key=lambda x: -x['class_invariant'])
    return results


def print_component_table(results: List[Dict], top_n: int = 20):
    """Print a summary table of the top components."""
    print(f"\n{'Rank':>4}  {'I(C)':>8}  {'|C|':>5}  {'n_Tc':>5}  "
          f"{'Spread':>8}  {'PGs':<12}  {'Representative formulas'}")
    print("-" * 90)
    for i, r in enumerate(results[:top_n]):
        formulas = r['formulas'][:3]
        formula_str = ", ".join(formulas)
        if len(r['formulas']) > 3:
            formula_str += f" (+{len(r['formulas'])-3})"
        pgs = ", ".join(sorted(r['point_groups']))
        print(f"{i+1:>4}  {r['class_invariant']:>8.1f}  {r['n_materials']:>5}  "
              f"{r['n_with_tc']:>5}  {r['tc_spread']:>8.1f}  "
              f"{pgs:<12}  {formula_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from materials_2cat import build_cuprate_example

    print("=" * 60)
    print("  Substitution Graph Analysis")
    print("=" * 60)

    cat, tc = build_cuprate_example()
    results = component_analysis(cat)
    print_component_table(results)

    print("\nSubstitution families used:")
    for fam, elements in sorted(SUBSTITUTION_FAMILIES.items()):
        print(f"  {fam}: {elements}")
