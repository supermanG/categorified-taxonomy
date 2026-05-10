"""
Tests for the 2-categorical materials taxonomy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from materials_2cat import (
    Material, MaterialsCategory, Morphism, TwoMorphism,
    ProcessType, PointGroup, POINT_GROUPS,
    identity_morphism, compose_morphisms,
    substitution_morphism, doping_morphism, pressure_morphism,
    commutativity_witness,
    identity_2morphism, vertical_compose,
    TcFunctor,
    build_cuprate_example,
)


def test_material_creation():
    m = Material(
        material_id="test_001",
        composition={'La': 2, 'Cu': 1, 'O': 4},
        point_group=POINT_GROUPS['D4h'],
    )
    assert m.formula == "Cu1La2O4" or "Cu" in m.formula
    assert m.material_id == "test_001"
    assert m.point_group.name == "D4h"
    assert m.tc is None
    print("  PASS: material creation")


def test_identity_morphism():
    m = Material("id_test", {'Cu': 1}, POINT_GROUPS['C1'])
    f = identity_morphism(m)
    assert f.is_identity
    assert f.is_invertible
    assert f.source == f.target == m
    print("  PASS: identity morphism")


def test_composition():
    a = Material("A", {'Cu': 1}, POINT_GROUPS['D4h'])
    b = Material("B", {'Ni': 1}, POINT_GROUPS['D4h'])
    c = Material("C", {'Zn': 1}, POINT_GROUPS['D4h'])

    f = substitution_morphism(a, b, 'Cu', 'Ni')
    g = substitution_morphism(b, c, 'Ni', 'Zn')
    fg = compose_morphisms(f, g)

    assert fg.source == a
    assert fg.target == c

    # Identity composition
    id_a = identity_morphism(a)
    id_f = compose_morphisms(id_a, f)
    assert id_f.target == b

    f_id = compose_morphisms(f, identity_morphism(b))
    assert f_id.target == b

    print("  PASS: morphism composition")


def test_inverse():
    a = Material("A", {'Cu': 1}, POINT_GROUPS['D4h'])
    b = Material("B", {'Ni': 1}, POINT_GROUPS['D4h'])

    f = substitution_morphism(a, b, 'Cu', 'Ni')
    assert f.is_invertible
    f_inv = f.inverse()
    assert f_inv is not None
    assert f_inv.source == b
    assert f_inv.target == a

    # Pressure is invertible
    p = pressure_morphism(a, a, 10.0)
    assert p.is_invertible
    p_inv = p.inverse()
    assert p_inv.parameters['pressure_GPa'] == -10.0

    # Doping is not invertible
    d = doping_morphism(a, b, 'Ni', 'Cu', 0.1)
    assert not d.is_invertible

    print("  PASS: morphism inverse")


def test_2morphism():
    a = Material("A", {'Cu': 1}, POINT_GROUPS['D4h'])
    b = Material("B", {'Ni': 1}, POINT_GROUPS['D4h'])

    f = substitution_morphism(a, b, 'Cu', 'Ni')
    g = substitution_morphism(a, b, 'Cu', 'Ni')

    alpha = TwoMorphism(f, g, witness_type="test")
    assert alpha.domain == a
    assert alpha.codomain == b

    # Identity 2-morphism
    id_alpha = identity_2morphism(f)
    assert id_alpha.source_morphism == id_alpha.target_morphism

    print("  PASS: 2-morphisms")


def test_category():
    cat = MaterialsCategory()
    a = Material("A", {'Cu': 1}, POINT_GROUPS['D4h'], properties={'Tc': 10.0})
    b = Material("B", {'Ni': 1}, POINT_GROUPS['D4h'], properties={'Tc': 20.0})
    c = Material("C", {'Zn': 1}, POINT_GROUPS['Oh'], properties={'Tc': 5.0})

    cat.add_material(a)
    cat.add_material(b)
    cat.add_material(c)

    f = substitution_morphism(a, b, 'Cu', 'Ni')
    cat.add_morphism(f)

    assert cat.n_objects == 3
    assert cat.n_morphisms == 1

    # Connected components
    comps = cat.connected_components()
    assert len(comps) == 2  # {A, B} and {C}

    # Sub-category
    sub = cat.sub_category_by_point_group('D4h')
    assert sub.n_objects == 2

    print("  PASS: category operations")


def test_tc_functor():
    cat, tc = build_cuprate_example()

    # Tc on objects
    assert tc.on_object(cat.materials['YBCO']) == 92.0
    assert tc.on_object(cat.materials['Hg1223']) == 133.0
    assert tc.on_object(cat.materials['La2CuO4']) == 0.0

    # Class invariants
    invariants = tc.all_class_invariants()
    assert len(invariants) > 0

    # The cuprate component should have I(C) >= 133 (Hg-1223)
    for comp, inv in invariants.items():
        if 'Hg1223' in comp:
            assert inv >= 133.0
            break

    # Functoriality
    for morph in cat.morphisms:
        ok, msg = tc.verify_functoriality(morph)
        assert ok, f"Functoriality violation: {msg}"

    print("  PASS: Tc functor")


def test_cuprate_example():
    cat, tc = build_cuprate_example()

    assert cat.n_objects == 6
    assert cat.n_morphisms >= 5
    assert len(cat.two_morphisms) >= 1

    # Verify connected component structure
    comps = cat.connected_components()
    # YBCO (D2h) is separate from the D4h cuprates
    d4h_mats = cat.materials_in_class('D4h')
    d2h_mats = cat.materials_in_class('D2h')
    assert len(d4h_mats) >= 4
    assert len(d2h_mats) >= 1

    print("  PASS: cuprate example")


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 40)
    print("  Testing materials_2cat")
    print("=" * 40)

    test_material_creation()
    test_identity_morphism()
    test_composition()
    test_inverse()
    test_2morphism()
    test_category()
    test_tc_functor()
    test_cuprate_example()

    print("\nAll tests passed.")
