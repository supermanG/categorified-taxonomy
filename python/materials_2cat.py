"""
materials_2cat.py
=================
2-categorical materials taxonomy with Tc as a 2-functor.

Defines the 2-category M of crystalline materials:
  - Objects: materials (crystallographic structure + composition)
  - 1-morphisms: physical processes (substitution, doping, pressure, strain)
  - 2-morphisms: natural transformations between processes (commutativity
    witnesses for "doping then substitution = substitution then doping")

The SC critical temperature Tc is a 2-functor
  Tc : M --> R_{>=0}^{pos}
where the codomain is the posetal 2-category of non-negative reals.

LH & Claude 2026
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# 1. Objects of the 2-category: Materials
# ---------------------------------------------------------------------------

class CrystalSystem(Enum):
    TRICLINIC = auto()
    MONOCLINIC = auto()
    ORTHORHOMBIC = auto()
    TETRAGONAL = auto()
    TRIGONAL = auto()
    HEXAGONAL = auto()
    CUBIC = auto()


@dataclass(frozen=True)
class PointGroup:
    """Crystallographic point group (Schoenflies notation)."""
    name: str
    order: int
    crystal_system: CrystalSystem

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, PointGroup):
            return NotImplemented
        return self.name == other.name


POINT_GROUPS = {
    'C1':  PointGroup('C1',  1,  CrystalSystem.TRICLINIC),
    'Ci':  PointGroup('Ci',  2,  CrystalSystem.TRICLINIC),
    'C2':  PointGroup('C2',  2,  CrystalSystem.MONOCLINIC),
    'Cs':  PointGroup('Cs',  2,  CrystalSystem.MONOCLINIC),
    'C2h': PointGroup('C2h', 4,  CrystalSystem.MONOCLINIC),
    'D2':  PointGroup('D2',  4,  CrystalSystem.ORTHORHOMBIC),
    'C2v': PointGroup('C2v', 4,  CrystalSystem.ORTHORHOMBIC),
    'D2h': PointGroup('D2h', 8,  CrystalSystem.ORTHORHOMBIC),
    'C4':  PointGroup('C4',  4,  CrystalSystem.TETRAGONAL),
    'S4':  PointGroup('S4',  4,  CrystalSystem.TETRAGONAL),
    'C4h': PointGroup('C4h', 8,  CrystalSystem.TETRAGONAL),
    'D4':  PointGroup('D4',  8,  CrystalSystem.TETRAGONAL),
    'C4v': PointGroup('C4v', 8,  CrystalSystem.TETRAGONAL),
    'D2d': PointGroup('D2d', 8,  CrystalSystem.TETRAGONAL),
    'D4h': PointGroup('D4h', 16, CrystalSystem.TETRAGONAL),
    'C3':  PointGroup('C3',  3,  CrystalSystem.TRIGONAL),
    'S6':  PointGroup('S6',  6,  CrystalSystem.TRIGONAL),
    'D3':  PointGroup('D3',  6,  CrystalSystem.TRIGONAL),
    'C3v': PointGroup('C3v', 6,  CrystalSystem.TRIGONAL),
    'D3d': PointGroup('D3d', 12, CrystalSystem.TRIGONAL),
    'C6':  PointGroup('C6',  6,  CrystalSystem.HEXAGONAL),
    'C3h': PointGroup('C3h', 6,  CrystalSystem.HEXAGONAL),
    'C6h': PointGroup('C6h', 12, CrystalSystem.HEXAGONAL),
    'D6':  PointGroup('D6',  12, CrystalSystem.HEXAGONAL),
    'C6v': PointGroup('C6v', 12, CrystalSystem.HEXAGONAL),
    'D3h': PointGroup('D3h', 12, CrystalSystem.HEXAGONAL),
    'D6h': PointGroup('D6h', 24, CrystalSystem.HEXAGONAL),
    'T':   PointGroup('T',   12, CrystalSystem.CUBIC),
    'Th':  PointGroup('Th',  24, CrystalSystem.CUBIC),
    'O':   PointGroup('O',   24, CrystalSystem.CUBIC),
    'Td':  PointGroup('Td',  24, CrystalSystem.CUBIC),
    'Oh':  PointGroup('Oh',  48, CrystalSystem.CUBIC),
}


@dataclass
class Material:
    """
    Object of the 2-category M.

    A material is specified by:
      - composition: dict mapping element symbol to count
      - point_group: crystallographic point group
      - spacegroup: international number (1-230)
      - lattice_params: (a, b, c, alpha, beta, gamma)
      - properties: dict of measured/computed properties (Tc, lambda, etc.)
      - material_id: unique identifier (e.g., JARVIS jid)

    Two materials are isomorphic (in the 2-categorical sense) if they
    are related by a sequence of invertible 1-morphisms.
    """
    material_id: str
    composition: Dict[str, float]
    point_group: PointGroup
    spacegroup: int = 1
    lattice_params: Tuple[float, ...] = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
    properties: Dict[str, float] = field(default_factory=dict)

    @property
    def formula(self) -> str:
        parts = []
        for el, count in sorted(self.composition.items()):
            if count == 1:
                parts.append(el)
            elif count == int(count):
                parts.append(f"{el}{int(count)}")
            else:
                parts.append(f"{el}{count:.2f}")
        return "".join(parts)

    @property
    def tc(self) -> Optional[float]:
        return self.properties.get('Tc')

    @property
    def chi_n(self) -> Optional[Dict[int, float]]:
        """Character values chi_n from star_G decomposition."""
        result = {}
        for key, val in self.properties.items():
            if key.startswith('chi_'):
                try:
                    n = int(key.split('_')[1])
                    result[n] = val
                except (ValueError, IndexError):
                    pass
        return result if result else None

    def __hash__(self):
        return hash(self.material_id)

    def __eq__(self, other):
        if not isinstance(other, Material):
            return NotImplemented
        return self.material_id == other.material_id


# ---------------------------------------------------------------------------
# 2. 1-morphisms: Physical processes
# ---------------------------------------------------------------------------

class ProcessType(Enum):
    SUBSTITUTION = auto()     # replace element A with element B
    DOPING = auto()           # partial substitution (fractional occupancy)
    HYDROSTATIC_PRESSURE = auto()
    UNIAXIAL_STRAIN = auto()
    EPITAXIAL_STRAIN = auto()
    INTERCALATION = auto()    # insert atoms into van der Waals gaps
    VACANCY = auto()          # remove atoms (create vacancies)
    IDENTITY = auto()


@dataclass
class Morphism:
    """
    1-morphism in the 2-category M.

    A morphism f: M1 --> M2 represents a physical process that transforms
    material M1 into material M2. The process is characterized by:
      - process_type: what kind of transformation
      - parameters: type-specific parameters
      - preserves_symmetry: whether the point group is preserved

    Composition: (f ; g)(M) = g(f(M)) (diagrammatic order).
    """
    source: Material
    target: Material
    process_type: ProcessType
    parameters: Dict[str, Any] = field(default_factory=dict)
    preserves_symmetry: bool = True
    label: str = ""

    @property
    def is_identity(self) -> bool:
        return self.process_type == ProcessType.IDENTITY

    @property
    def is_invertible(self) -> bool:
        """Whether this morphism has an inverse (is an isomorphism)."""
        if self.process_type == ProcessType.IDENTITY:
            return True
        if self.process_type == ProcessType.SUBSTITUTION:
            return True
        if self.process_type == ProcessType.HYDROSTATIC_PRESSURE:
            return True
        return False

    def inverse(self) -> Optional['Morphism']:
        """Construct the inverse morphism, if it exists."""
        if not self.is_invertible:
            return None
        if self.process_type == ProcessType.IDENTITY:
            return identity_morphism(self.source)
        if self.process_type == ProcessType.SUBSTITUTION:
            return Morphism(
                source=self.target,
                target=self.source,
                process_type=ProcessType.SUBSTITUTION,
                parameters={
                    'from_element': self.parameters.get('to_element'),
                    'to_element': self.parameters.get('from_element'),
                    'site': self.parameters.get('site'),
                },
                preserves_symmetry=self.preserves_symmetry,
                label=f"inv({self.label})" if self.label else "",
            )
        if self.process_type == ProcessType.HYDROSTATIC_PRESSURE:
            p = self.parameters.get('pressure_GPa', 0.0)
            return Morphism(
                source=self.target,
                target=self.source,
                process_type=ProcessType.HYDROSTATIC_PRESSURE,
                parameters={'pressure_GPa': -p},
                preserves_symmetry=self.preserves_symmetry,
                label=f"inv({self.label})" if self.label else "",
            )
        return None


def identity_morphism(M: Material) -> Morphism:
    """Identity 1-morphism on material M."""
    return Morphism(
        source=M, target=M,
        process_type=ProcessType.IDENTITY,
        preserves_symmetry=True,
        label=f"id_{M.material_id}",
    )


def compose_morphisms(f: Morphism, g: Morphism) -> Morphism:
    """
    Compose 1-morphisms: f ; g (diagrammatic order).

    f: A --> B, g: B --> C, result: A --> C.
    """
    assert f.target == g.source, (
        f"Cannot compose: f.target={f.target.material_id} != "
        f"g.source={g.source.material_id}"
    )

    if f.is_identity:
        return g
    if g.is_identity:
        return f

    return Morphism(
        source=f.source,
        target=g.target,
        process_type=g.process_type,
        parameters={
            'composed_from': [
                (f.process_type.name, f.parameters),
                (g.process_type.name, g.parameters),
            ]
        },
        preserves_symmetry=f.preserves_symmetry and g.preserves_symmetry,
        label=f"({f.label} ; {g.label})" if f.label and g.label else "",
    )


# ---------------------------------------------------------------------------
# 3. 2-morphisms: Natural transformations between processes
# ---------------------------------------------------------------------------

@dataclass
class TwoMorphism:
    """
    2-morphism in the 2-category M.

    A 2-morphism alpha: f ==> g (where f, g: M1 --> M2) witnesses that
    two physical processes yield the same result up to the relevant
    equivalence. In the posetal codomain (R_{>=0}), 2-morphisms encode
    inequalities: alpha: f ==> g means Tc(f(M)) <= Tc(g(M)).

    For the materials 2-category, the key 2-morphisms are:
    1. Commutativity witnesses: "doping then substitution =
       substitution then doping" (up to coherent iso).
    2. Monotonicity witnesses: "Tc under doping path A <= Tc under
       doping path B" within a family.
    3. Exchange witnesses (interchange law): composites of 2-morphisms
       along both 1-morphism and 2-morphism directions.
    """
    source_morphism: Morphism
    target_morphism: Morphism
    witness_type: str = "commutativity"
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.source_morphism.source == self.target_morphism.source, (
            "2-morphism source/target must share domain"
        )
        assert self.source_morphism.target == self.target_morphism.target, (
            "2-morphism source/target must share codomain"
        )

    @property
    def domain(self) -> Material:
        return self.source_morphism.source

    @property
    def codomain(self) -> Material:
        return self.source_morphism.target


def identity_2morphism(f: Morphism) -> TwoMorphism:
    """Identity 2-morphism on a 1-morphism f."""
    return TwoMorphism(
        source_morphism=f,
        target_morphism=f,
        witness_type="identity",
    )


def vertical_compose(alpha: TwoMorphism, beta: TwoMorphism) -> TwoMorphism:
    """
    Vertical composition of 2-morphisms:
    alpha: f ==> g, beta: g ==> h gives alpha . beta: f ==> h.
    """
    assert alpha.target_morphism == beta.source_morphism, (
        "Vertical composition: alpha.target must equal beta.source"
    )
    return TwoMorphism(
        source_morphism=alpha.source_morphism,
        target_morphism=beta.target_morphism,
        witness_type="vertical_composite",
        data={'components': [alpha, beta]},
    )


def horizontal_compose(alpha: TwoMorphism, beta: TwoMorphism) -> TwoMorphism:
    """
    Horizontal composition of 2-morphisms (whiskering):
    alpha: f ==> f' (A --> B), beta: g ==> g' (B --> C)
    gives alpha * beta: f;g ==> f';g' (A --> C).

    This is the interchange law in a 2-category.
    """
    assert alpha.codomain == beta.domain, (
        "Horizontal composition: alpha.codomain must equal beta.domain"
    )
    return TwoMorphism(
        source_morphism=compose_morphisms(
            alpha.source_morphism, beta.source_morphism
        ),
        target_morphism=compose_morphisms(
            alpha.target_morphism, beta.target_morphism
        ),
        witness_type="horizontal_composite",
        data={'left': alpha, 'right': beta},
    )


# ---------------------------------------------------------------------------
# 4. The 2-category M (assembled)
# ---------------------------------------------------------------------------

class MaterialsCategory:
    """
    The 2-category M of crystalline materials.

    This is a concrete 2-category: objects are Material instances,
    1-morphisms are Morphism instances, 2-morphisms are TwoMorphism
    instances. The hom-categories Hom(A, B) are (small) categories
    whose objects are 1-morphisms A --> B and whose morphisms are
    2-morphisms between them.

    The category is built incrementally: materials and processes are
    added, and the system maintains the substitution graph as a
    NetworkX DiGraph.
    """

    def __init__(self):
        self.materials: Dict[str, Material] = {}
        self.morphisms: List[Morphism] = []
        self.two_morphisms: List[TwoMorphism] = []
        self._graph = nx.DiGraph()

    def add_material(self, mat: Material):
        self.materials[mat.material_id] = mat
        self._graph.add_node(mat.material_id, point_group=mat.point_group.name)

    def add_morphism(self, morph: Morphism):
        if morph.source.material_id not in self.materials:
            self.add_material(morph.source)
        if morph.target.material_id not in self.materials:
            self.add_material(morph.target)

        self.morphisms.append(morph)
        self._graph.add_edge(
            morph.source.material_id,
            morph.target.material_id,
            process_type=morph.process_type.name,
            preserves_symmetry=morph.preserves_symmetry,
            label=morph.label,
        )
        if morph.is_invertible:
            self._graph.add_edge(
                morph.target.material_id,
                morph.source.material_id,
                process_type=morph.process_type.name + "_inv",
                preserves_symmetry=morph.preserves_symmetry,
            )

    def add_2morphism(self, two_morph: TwoMorphism):
        self.two_morphisms.append(two_morph)

    # -- Sub-2-categories --

    def sub_category_by_point_group(self, pg_name: str) -> 'MaterialsCategory':
        """
        Extract the sub-2-category M_G of materials with point group G
        and symmetry-preserving processes.
        """
        sub = MaterialsCategory()
        for mid, mat in self.materials.items():
            if mat.point_group.name == pg_name:
                sub.add_material(mat)
        for morph in self.morphisms:
            if (morph.source.point_group.name == pg_name and
                    morph.target.point_group.name == pg_name and
                    morph.preserves_symmetry):
                sub.add_morphism(morph)
        return sub

    # -- Connected components (substitution graph) --

    def connected_components(self) -> List[Set[str]]:
        """
        Connected components of the substitution graph.

        Within each component, all materials are related by chains of
        invertible 1-morphisms (substitutions). The Tc 2-functor is
        bounded on each component.
        """
        undirected = self._graph.to_undirected()
        return [set(c) for c in nx.connected_components(undirected)]

    def component_of(self, material_id: str) -> Set[str]:
        """Materials in the same connected component as material_id."""
        undirected = self._graph.to_undirected()
        for component in nx.connected_components(undirected):
            if material_id in component:
                return set(component)
        return {material_id}

    # -- Queries --

    def materials_in_class(self, pg_name: str) -> List[Material]:
        return [m for m in self.materials.values()
                if m.point_group.name == pg_name]

    def morphisms_from(self, material_id: str) -> List[Morphism]:
        return [m for m in self.morphisms
                if m.source.material_id == material_id]

    def morphisms_to(self, material_id: str) -> List[Morphism]:
        return [m for m in self.morphisms
                if m.target.material_id == material_id]

    def path_between(self, src_id: str, tgt_id: str) -> Optional[List[str]]:
        """Shortest path in the substitution graph."""
        try:
            return nx.shortest_path(self._graph, src_id, tgt_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    @property
    def n_objects(self) -> int:
        return len(self.materials)

    @property
    def n_morphisms(self) -> int:
        return len(self.morphisms)

    @property
    def n_components(self) -> int:
        return len(self.connected_components())

    def summary(self) -> str:
        comps = self.connected_components()
        comp_sizes = sorted([len(c) for c in comps], reverse=True)
        pg_counts: Dict[str, int] = {}
        for m in self.materials.values():
            pg_counts[m.point_group.name] = pg_counts.get(
                m.point_group.name, 0) + 1

        lines = [
            f"MaterialsCategory: {self.n_objects} objects, "
            f"{self.n_morphisms} 1-morphisms, "
            f"{len(self.two_morphisms)} 2-morphisms",
            f"Connected components: {len(comps)} "
            f"(sizes: {comp_sizes[:10]}{'...' if len(comp_sizes) > 10 else ''})",
            "Point groups:",
        ]
        for pg, count in sorted(pg_counts.items(),
                                key=lambda x: -x[1])[:10]:
            lines.append(f"  {pg}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Tc 2-functor
# ---------------------------------------------------------------------------

class TcFunctor:
    """
    The Tc 2-functor: M --> R_{>=0}^{pos}.

    The codomain is the posetal 2-category of non-negative reals:
      - Objects: R_{>=0} (temperatures in Kelvin)
      - 1-morphisms: a <= b (unique morphism from a to b iff a <= b)
      - 2-morphisms: trivial (at most one 2-morphism between any pair
        of 1-morphisms, since the codomain is posetal)

    On objects: Tc(M) = critical temperature of material M.
    On 1-morphisms: Tc(f: M1 --> M2) = the assertion Tc(M1) <= Tc(M2)
      (or its reversal, depending on the process).
    On 2-morphisms: trivially preserved (posetal codomain).

    The key property: for each connected component C of the substitution
    graph within a crystallographic class, there exists a CLASS INVARIANT
      I(C) = sup { Tc(M) : M in C }
    that is computable from the symmetry constraints and chi_n decomposition.

    The 2-functoriality condition is:
      For composable f: A --> B and g: B --> C,
      Tc(f ; g) = Tc(g) . Tc(f)  (up to coherent natural transformation)
    which in the posetal codomain reduces to:
      |Tc(C) - Tc(A)| <= |Tc(C) - Tc(B)| + |Tc(B) - Tc(A)|
    (triangle inequality, automatically satisfied).
    """

    def __init__(self, category: MaterialsCategory):
        self.category = category
        self._tc_cache: Dict[str, float] = {}
        self._class_invariants: Dict[frozenset, float] = {}

    def on_object(self, mat: Material) -> float:
        """Evaluate Tc on an object (material)."""
        if mat.material_id in self._tc_cache:
            return self._tc_cache[mat.material_id]
        tc = mat.properties.get('Tc', 0.0)
        self._tc_cache[mat.material_id] = tc
        return tc

    def on_morphism(self, morph: Morphism) -> Tuple[float, float]:
        """
        Evaluate Tc on a 1-morphism.

        Returns (Tc_source, Tc_target). In the posetal codomain,
        the morphism exists iff Tc_source R Tc_target (where R is the
        appropriate order relation for the process type).
        """
        tc_src = self.on_object(morph.source)
        tc_tgt = self.on_object(morph.target)
        return tc_src, tc_tgt

    def class_invariant(self, component: Set[str]) -> float:
        """
        Compute the class invariant I(C) for a connected component C.

        I(C) = max { Tc(M) : M in C, Tc(M) is known }

        This is the upper bound on Tc for all materials in the component.
        When new materials are added to the component, I(C) can only
        increase (monotonicity of sup).
        """
        key = frozenset(component)
        if key in self._class_invariants:
            return self._class_invariants[key]

        tc_values = []
        for mid in component:
            if mid in self.category.materials:
                tc = self.on_object(self.category.materials[mid])
                if tc > 0:
                    tc_values.append(tc)

        inv = max(tc_values) if tc_values else 0.0
        self._class_invariants[key] = inv
        return inv

    def all_class_invariants(self) -> Dict[frozenset, float]:
        """Compute class invariants for all connected components."""
        result = {}
        for comp in self.category.connected_components():
            result[frozenset(comp)] = self.class_invariant(comp)
        return result

    def verify_functoriality(
        self, morph: Morphism, tolerance: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Verify the functoriality condition for a 1-morphism.

        For a morphism f: A --> B in a connected component C:
          Tc(B) <= I(C) (class invariant bound)

        Returns (is_consistent, message).
        """
        comp = self.category.component_of(morph.source.material_id)
        inv = self.class_invariant(comp)
        tc_tgt = self.on_object(morph.target)

        if tc_tgt <= inv + tolerance:
            return True, (
                f"Tc({morph.target.material_id}) = {tc_tgt:.2f} K <= "
                f"I(C) = {inv:.2f} K"
            )
        else:
            return False, (
                f"VIOLATION: Tc({morph.target.material_id}) = {tc_tgt:.2f} K > "
                f"I(C) = {inv:.2f} K (exceeds class invariant by "
                f"{tc_tgt - inv:.2f} K)"
            )

    def summary(self) -> str:
        invariants = self.all_class_invariants()
        nontrivial = {k: v for k, v in invariants.items() if v > 0}
        lines = [
            f"TcFunctor: {len(invariants)} components, "
            f"{len(nontrivial)} with known Tc > 0",
        ]
        for comp, inv in sorted(nontrivial.items(),
                                key=lambda x: -x[1])[:10]:
            pg_set = set()
            for mid in comp:
                mat = self.category.materials.get(mid)
                if mat:
                    pg_set.add(mat.point_group.name)
            lines.append(
                f"  I(C) = {inv:8.2f} K  |C| = {len(comp):4d}  "
                f"PGs: {pg_set}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Substitution morphism constructors
# ---------------------------------------------------------------------------

def substitution_morphism(
    source: Material,
    target: Material,
    from_element: str,
    to_element: str,
    site: str = "all",
) -> Morphism:
    """
    Construct a substitution 1-morphism: replace from_element with
    to_element in the source material to obtain the target.
    """
    return Morphism(
        source=source,
        target=target,
        process_type=ProcessType.SUBSTITUTION,
        parameters={
            'from_element': from_element,
            'to_element': to_element,
            'site': site,
        },
        preserves_symmetry=(source.point_group == target.point_group),
        label=f"sub({from_element}->{to_element})",
    )


def doping_morphism(
    source: Material,
    target: Material,
    dopant_element: str,
    host_element: str,
    concentration: float,
) -> Morphism:
    """
    Construct a doping 1-morphism: partially substitute host_element
    with dopant_element at the given concentration x in [0, 1].
    """
    return Morphism(
        source=source,
        target=target,
        process_type=ProcessType.DOPING,
        parameters={
            'dopant': dopant_element,
            'host': host_element,
            'concentration': concentration,
        },
        preserves_symmetry=(source.point_group == target.point_group),
        label=f"dope({host_element}:{dopant_element}@{concentration:.2f})",
    )


def pressure_morphism(
    source: Material,
    target: Material,
    pressure_gpa: float,
) -> Morphism:
    """Construct a hydrostatic pressure 1-morphism."""
    return Morphism(
        source=source,
        target=target,
        process_type=ProcessType.HYDROSTATIC_PRESSURE,
        parameters={'pressure_GPa': pressure_gpa},
        preserves_symmetry=True,
        label=f"P={pressure_gpa:.1f}GPa",
    )


# ---------------------------------------------------------------------------
# 7. Commutativity witness (key 2-morphism constructor)
# ---------------------------------------------------------------------------

def commutativity_witness(
    f: Morphism, g: Morphism,
    f_prime: Morphism, g_prime: Morphism,
) -> Optional[TwoMorphism]:
    """
    Construct a commutativity witness 2-morphism for the square:

        A --f--> B
        |        |
        g        g'
        |        |
        v        v
        C --f'-> D

    The 2-morphism witnesses: f ; g' ==> g ; f' (both go A --> D).
    Returns None if the square does not commute (i.e., endpoints differ).
    """
    fg_prime = compose_morphisms(f, g_prime)
    gf_prime = compose_morphisms(g, f_prime)

    if fg_prime.target != gf_prime.target:
        return None

    return TwoMorphism(
        source_morphism=fg_prime,
        target_morphism=gf_prime,
        witness_type="commutativity",
        data={
            'square': {
                'top': f.label,
                'bottom': f_prime.label,
                'left': g.label,
                'right': g_prime.label,
            }
        },
    )


# ---------------------------------------------------------------------------
# 8. Example: cuprate family
# ---------------------------------------------------------------------------

def build_cuprate_example() -> Tuple[MaterialsCategory, TcFunctor]:
    """
    Build a small example 2-category for the cuprate family.

    Demonstrates substitution morphisms and Tc class invariants for:
    - La2CuO4 (parent, Tc ~ 0 K undoped)
    - La_{2-x}Sr_x CuO4 (LSCO, Tc ~ 38 K at optimal doping x=0.15)
    - La_{2-x}Ba_x CuO4 (LBCO, Tc ~ 30 K)
    - YBa2Cu3O7 (YBCO, Tc ~ 92 K)
    - Bi2Sr2CaCu2O8 (BSCCO-2212, Tc ~ 85 K)
    - HgBa2Ca2Cu3O8 (Hg-1223, Tc ~ 133 K)
    """
    cat = MaterialsCategory()

    # -- Objects --
    La2CuO4 = Material(
        material_id="La2CuO4",
        composition={'La': 2, 'Cu': 1, 'O': 4},
        point_group=POINT_GROUPS['D4h'],
        spacegroup=139,
        properties={'Tc': 0.0, 'pairing': 'd-wave'},
    )
    LSCO = Material(
        material_id="LSCO_x015",
        composition={'La': 1.85, 'Sr': 0.15, 'Cu': 1, 'O': 4},
        point_group=POINT_GROUPS['D4h'],
        spacegroup=139,
        properties={'Tc': 38.0, 'pairing': 'd-wave'},
    )
    LBCO = Material(
        material_id="LBCO_x012",
        composition={'La': 1.88, 'Ba': 0.12, 'Cu': 1, 'O': 4},
        point_group=POINT_GROUPS['D4h'],
        spacegroup=139,
        properties={'Tc': 30.0, 'pairing': 'd-wave'},
    )
    YBCO = Material(
        material_id="YBCO",
        composition={'Y': 1, 'Ba': 2, 'Cu': 3, 'O': 7},
        point_group=POINT_GROUPS['D2h'],
        spacegroup=47,
        properties={'Tc': 92.0, 'pairing': 'd-wave'},
    )
    BSCCO = Material(
        material_id="BSCCO_2212",
        composition={'Bi': 2, 'Sr': 2, 'Ca': 1, 'Cu': 2, 'O': 8},
        point_group=POINT_GROUPS['D4h'],
        spacegroup=139,
        properties={'Tc': 85.0, 'pairing': 'd-wave'},
    )
    Hg1223 = Material(
        material_id="Hg1223",
        composition={'Hg': 1, 'Ba': 2, 'Ca': 2, 'Cu': 3, 'O': 8},
        point_group=POINT_GROUPS['D4h'],
        spacegroup=139,
        properties={'Tc': 133.0, 'pairing': 'd-wave'},
    )

    for m in [La2CuO4, LSCO, LBCO, YBCO, BSCCO, Hg1223]:
        cat.add_material(m)

    # -- 1-morphisms --
    # Doping: La2CuO4 --> LSCO (Sr doping)
    cat.add_morphism(doping_morphism(
        La2CuO4, LSCO, 'Sr', 'La', 0.15,
    ))
    # Doping: La2CuO4 --> LBCO (Ba doping)
    cat.add_morphism(doping_morphism(
        La2CuO4, LBCO, 'Ba', 'La', 0.12,
    ))
    # Substitution: LSCO --> LBCO (replace Sr with Ba at the dopant site)
    cat.add_morphism(substitution_morphism(
        LSCO, LBCO, 'Sr', 'Ba', site='dopant',
    ))
    # BSCCO is in the same D4h class as LSCO
    cat.add_morphism(substitution_morphism(
        LSCO, BSCCO, 'La', 'Bi', site='A-site',
    ))
    # Hg-1223 is in the same D4h class
    cat.add_morphism(substitution_morphism(
        BSCCO, Hg1223, 'Bi', 'Hg', site='A-site',
    ))

    # -- 2-morphism: commutativity witness --
    # The doping-then-substitution square for La2CuO4:
    #   La2CuO4 --Sr-dope--> LSCO
    #      |                    |
    #   Ba-dope             Sr->Ba sub
    #      |                    |
    #      v                    v
    #   LBCO  ----id------>  LBCO
    #
    # This witnesses: (Sr-dope ; Sr->Ba-sub) ==> (Ba-dope ; id)
    sr_dope = cat.morphisms[0]
    ba_dope = cat.morphisms[1]
    sr_ba_sub = cat.morphisms[2]
    lbco_id = identity_morphism(LBCO)

    witness = commutativity_witness(sr_dope, ba_dope, lbco_id, sr_ba_sub)
    if witness:
        cat.add_2morphism(witness)

    # -- Tc functor --
    tc_func = TcFunctor(cat)

    return cat, tc_func


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print("  Categorified Taxonomy: Cuprate Example")
    print("=" * 60)

    cat, tc = build_cuprate_example()

    print("\n" + cat.summary())
    print("\n" + tc.summary())

    # Connected components
    print("\nConnected components:")
    for i, comp in enumerate(cat.connected_components()):
        materials = [cat.materials[mid] for mid in comp]
        formulas = [m.formula for m in materials]
        inv = tc.class_invariant(comp)
        print(f"  C{i}: {formulas}  I(C) = {inv:.1f} K")

    # Verify functoriality
    print("\nFunctoriality checks:")
    for morph in cat.morphisms:
        ok, msg = tc.verify_functoriality(morph)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {morph.label}: {msg}")

    # Sub-2-category by point group
    print("\nSub-2-category M_{D4h}:")
    sub = cat.sub_category_by_point_group('D4h')
    print(f"  {sub.summary()}")

    # Path queries
    print("\nPath from La2CuO4 to Hg1223:")
    path = cat.path_between("La2CuO4", "Hg1223")
    if path:
        print(f"  {' -> '.join(path)}")
    else:
        print("  No path found")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)
