# Categorified Taxonomy

2-categorical materials taxonomy with Tc as a 2-functor.

## Overview

This project lifts the materials-prediction setting to a **2-category**
where:

- **Objects** are materials (points in the moduli of crystalline structures)
- **1-morphisms** are physical processes (substitution, doping, pressure, strain)
- **2-morphisms** are universal natural transformations (commutativity
  squares for "doping then substitution = substitution then doping")

The superconducting critical temperature Tc becomes a **2-functor**,
giving universally-natural statements like "same Tc up to substitution
within a structural class".

## Key results (target)

1. On each connected component of the substitution graph within a
   crystallographic class, Tc is bounded above by an explicit class
   invariant.
2. The class invariant is computable from symmetry constraints + chi_n +
   family chemistry (reduces to the v15 hybrid Tc estimator at the
   1-categorical level).

## Repository structure

```
python/
  materials_2cat.py       -- 2-category definition (objects, morphisms, 2-morphisms)
  substitution_graph.py   -- connected components of the substitution graph
  class_invariants.py     -- Tc upper bounds per connected component
  cross_validate.py       -- validation against v14 candidate pool
  tests/                  -- unit tests
latex/
  main.tex                -- paper skeleton
lean/
  (optional Lean 4 formalization)
data/
  (materialsdata, cached JARVIS joins)
```

## Dependencies

- Python 3.10+
- numpy, scipy, networkx, pandas
- jarvis-tools (for materials data access)
- rtsc repo (v14 candidate pool, StarGAlgebra)

## References

- Leinster, "Higher Operads, Higher Categories" (2004)
- Loday-Vallette, "Algebraic Operads" (2012)
- Hoyos et al., "Tensor-group-sym" (2025)
- Horesh, "Hierarchical star_G" (2026)

## Related work (same project family)

- [rtsc](https://github.com/supermanG/rtsc) (parent project, star_G spectroscopy)
- [Ritt-Kolchin](https://github.com/supermanG/Ritt-Kolchin) (Tc upper bounds)
- [G-spacegroup](https://github.com/supermanG/G-spacegroup) (full spacegroup spectroscopy)

## License

See LICENSE file.

## Authors

LH & Claude 2026
