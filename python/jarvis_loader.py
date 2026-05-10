"""
jarvis_loader.py
================
Load JARVIS dft_3d and supercon datasets into the 2-categorical framework.

Ingests:
  - JARVIS dft_3d (55K+ materials with structure, bandgap, etc.)
  - JARVIS supercon_3d (1058 materials with Tc, lambda, omega_log)

Produces Material objects suitable for build_substitution_graph().

LH & Claude 2026
"""

from __future__ import annotations

import json
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

from materials_2cat import Material, POINT_GROUPS, PointGroup, CrystalSystem


# ---------------------------------------------------------------------------
# Spacegroup -> point group mapping
# ---------------------------------------------------------------------------

SPG_TO_PG = {}

_MAPPING = {
    'C1':  range(1, 2),
    'Ci':  range(2, 3),
    'C2':  range(3, 5),
    'Cs':  range(6, 10),
    'C2h': range(10, 16),
    'D2':  range(16, 25),
    'C2v': range(25, 47),
    'D2h': range(47, 75),
    'C4':  range(75, 81),
    'S4':  range(81, 83),
    'C4h': range(83, 89),
    'D4':  range(89, 99),
    'C4v': range(99, 111),
    'D2d': range(111, 123),
    'D4h': range(123, 143),
    'C3':  range(143, 147),
    'S6':  range(147, 149),
    'D3':  range(149, 156),
    'C3v': range(156, 162),
    'D3d': range(162, 168),
    'C6':  range(168, 174),
    'C3h': range(174, 175),
    'C6h': range(175, 177),
    'D6':  range(177, 183),
    'C6v': range(183, 187),
    'D3h': range(187, 191),
    'D6h': range(191, 195),
    'T':   range(195, 200),
    'Th':  range(200, 207),
    'O':   range(207, 215),
    'Td':  range(215, 221),
    'Oh':  range(221, 231),
}

for pg_name, spg_range in _MAPPING.items():
    for spg in spg_range:
        SPG_TO_PG[spg] = pg_name


def spacegroup_to_point_group(spg_number: int) -> PointGroup:
    """Map spacegroup number (1-230) to crystallographic point group."""
    pg_name = SPG_TO_PG.get(spg_number, 'C1')
    return POINT_GROUPS.get(pg_name, POINT_GROUPS['C1'])


# ---------------------------------------------------------------------------
# Formula parsing
# ---------------------------------------------------------------------------

def parse_formula(formula: str) -> Dict[str, float]:
    """Parse a chemical formula string into element:count dict."""
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    comp = {}
    for element, count_str in matches:
        if not element:
            continue
        count = float(count_str) if count_str else 1.0
        comp[element] = comp.get(element, 0) + count
    return comp


# ---------------------------------------------------------------------------
# Load JARVIS dft_3d
# ---------------------------------------------------------------------------

def load_jarvis_dft3d(
    cache_path: str,
    max_materials: int = None,
    min_bandgap: float = None,
    max_bandgap: float = None,
    require_stable: bool = False,
) -> List[Material]:
    """
    Load materials from JARVIS dft_3d cache JSON.

    Parameters
    ----------
    cache_path : path to dft_3d_2021_cache.json
    max_materials : limit number of materials loaded
    min_bandgap : filter by minimum bandgap (eV)
    max_bandgap : filter by maximum bandgap (eV)
    require_stable : if True, only include materials with ehull < 0.1 eV/atom

    Returns
    -------
    List of Material objects
    """
    print(f"Loading JARVIS dft_3d from {cache_path}...")
    with open(cache_path, 'r') as f:
        data = json.load(f)

    if max_materials:
        data = data[:max_materials]

    materials = []
    skipped = Counter()

    for entry in data:
        jid = entry.get('jid', '')
        formula = entry.get('formula', '')
        try:
            spg = int(entry.get('spg_number', 1))
        except (ValueError, TypeError):
            spg = 1

        if not formula:
            skipped['no_formula'] += 1
            continue

        # Composition
        atoms = entry.get('atoms', {})
        if isinstance(atoms, dict) and 'elements' in atoms:
            elements = atoms['elements']
            comp = Counter(elements)
            comp = dict(comp)
        else:
            comp = parse_formula(formula)

        if not comp:
            skipped['no_composition'] += 1
            continue

        # Point group from spacegroup
        pg = spacegroup_to_point_group(spg)

        # Properties
        props = {}
        bandgap = entry.get('optb88vdw_bandgap')
        if bandgap is not None:
            try:
                props['bandgap'] = float(bandgap)
            except (ValueError, TypeError):
                pass

        ehull = entry.get('ehull')
        if ehull is not None:
            try:
                props['ehull'] = float(ehull)
            except (ValueError, TypeError):
                pass

        form_e = entry.get('formation_energy_peratom')
        if form_e is not None:
            try:
                props['formation_energy'] = float(form_e)
            except (ValueError, TypeError):
                pass

        bulk_mod = entry.get('bulk_modulus_kv')
        if bulk_mod is not None:
            try:
                props['bulk_modulus'] = float(bulk_mod)
            except (ValueError, TypeError):
                pass

        # Filters
        if min_bandgap is not None and props.get('bandgap', 0) < min_bandgap:
            skipped['bandgap_low'] += 1
            continue
        if max_bandgap is not None and props.get('bandgap', 999) > max_bandgap:
            skipped['bandgap_high'] += 1
            continue
        if require_stable and props.get('ehull', 999) > 0.1:
            skipped['unstable'] += 1
            continue

        # Lattice parameters
        lat_params = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
        if isinstance(atoms, dict):
            abc = atoms.get('abc', [1, 1, 1])
            angles = atoms.get('angles', [90, 90, 90])
            if len(abc) >= 3 and len(angles) >= 3:
                lat_params = tuple(abc[:3]) + tuple(angles[:3])

        materials.append(Material(
            material_id=jid,
            composition={str(k): float(v) for k, v in comp.items()},
            point_group=pg,
            spacegroup=spg,
            lattice_params=lat_params,
            properties=props,
        ))

    print(f"  Loaded {len(materials)} materials "
          f"(skipped: {dict(skipped) if skipped else 'none'})")
    return materials


# ---------------------------------------------------------------------------
# Load JARVIS supercon
# ---------------------------------------------------------------------------

def load_jarvis_supercon(
    supercon_path: str,
) -> Dict[str, Dict]:
    """
    Load JARVIS supercon_3d dataset.

    Returns a dict mapping jid -> {Tc, lambda, omega_log, ...}
    """
    print(f"Loading JARVIS supercon from {supercon_path}...")
    with open(supercon_path, 'r') as f:
        data = json.load(f)

    supercon = {}
    for entry in data:
        jid = entry.get('jid', '')
        tc = entry.get('Tc')
        lamb = entry.get('lamb')
        wlog = entry.get('wlog')

        if not jid:
            continue

        supercon[jid] = {
            'Tc': float(tc) if tc is not None else None,
            'lambda': float(lamb) if lamb is not None else None,
            'omega_log': float(wlog) if wlog is not None else None,
            'stability': entry.get('stability', 'unknown'),
        }

    print(f"  Loaded {len(supercon)} superconductor records")

    # Statistics
    tc_values = [v['Tc'] for v in supercon.values()
                 if v['Tc'] is not None and v['Tc'] > 0]
    if tc_values:
        print(f"  Tc range: [{min(tc_values):.2f}, {max(tc_values):.2f}] K")
        print(f"  Tc > 10 K: {sum(1 for t in tc_values if t > 10)}")
        print(f"  Tc > 30 K: {sum(1 for t in tc_values if t > 30)}")

    return supercon


# ---------------------------------------------------------------------------
# Merge: annotate materials with Tc from supercon
# ---------------------------------------------------------------------------

def annotate_with_tc(
    materials: List[Material],
    supercon: Dict[str, Dict],
) -> int:
    """
    Annotate Material objects with Tc from the supercon dataset.

    Returns the number of materials annotated.
    """
    n_annotated = 0
    for mat in materials:
        sc = supercon.get(mat.material_id)
        if sc and sc.get('Tc') is not None:
            mat.properties['Tc'] = sc['Tc']
            if sc.get('lambda') is not None:
                mat.properties['lambda'] = sc['lambda']
            if sc.get('omega_log') is not None:
                mat.properties['omega_log'] = sc['omega_log']
            n_annotated += 1
    print(f"  Annotated {n_annotated} materials with Tc")
    return n_annotated


# ---------------------------------------------------------------------------
# Main: build materials list from JARVIS
# ---------------------------------------------------------------------------

def load_all_jarvis(
    data_dir: str,
    max_materials: int = None,
    metals_only: bool = True,
) -> Tuple[List[Material], Dict[str, Dict]]:
    """
    Load and merge JARVIS dft_3d + supercon datasets.

    Parameters
    ----------
    data_dir : directory containing dft_3d_2021_cache.json and supercon_3d.json
    max_materials : limit on dft_3d materials
    metals_only : if True, filter to bandgap < 0.5 eV (metallic/near-metallic)

    Returns
    -------
    materials : List[Material] with Tc annotations where available
    supercon : Dict[str, Dict] raw supercon data
    """
    dft3d_path = os.path.join(data_dir, 'dft_3d_2021_cache.json')
    supercon_path = os.path.join(data_dir, 'supercon_3d.json')

    if not os.path.exists(dft3d_path):
        raise FileNotFoundError(f"dft_3d cache not found at {dft3d_path}")

    materials = load_jarvis_dft3d(
        dft3d_path,
        max_materials=max_materials,
        max_bandgap=0.5 if metals_only else None,
    )

    supercon = {}
    if os.path.exists(supercon_path):
        supercon = load_jarvis_supercon(supercon_path)
        annotate_with_tc(materials, supercon)

    # Summary statistics
    pg_counts = Counter(m.point_group.name for m in materials)
    print(f"\n  Point group distribution (top 10):")
    for pg, count in pg_counts.most_common(10):
        print(f"    {pg:6s}: {count:6d}")

    tc_mats = [m for m in materials if m.tc and m.tc > 0]
    print(f"\n  Materials with Tc > 0: {len(tc_mats)}")

    return materials, supercon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print("  JARVIS Data Loader")
    print("=" * 60)

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
        print("JARVIS data directory not found. Using built-in test set.")
        from materials_2cat import build_cuprate_example
        cat, tc = build_cuprate_example()
        print(cat.summary())
    else:
        materials, supercon = load_all_jarvis(
            data_dir, max_materials=5000, metals_only=True
        )
        print(f"\nTotal materials loaded: {len(materials)}")
