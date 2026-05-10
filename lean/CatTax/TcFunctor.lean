/-
  CatTax.TcFunctor
  =================
  The Tc 2-functor from the materials 2-category to (NNReal, <=).

  We prove:
  1. Tc as a functor is well-defined (maps to posetal category)
  2. Functoriality: preserves composition and identities
  3. Monotonicity along morphism chains

  Horesh 2026
-/

import CatTax.Basic
import Mathlib.Data.NNReal.Defs

open CategoryTheory

namespace CatTax

/-! ### Tc as a functor on a category of materials -/

/-- A `TcAssignment` on a category M assigns a non-negative real Tc value
    to each object, monotone along morphisms: if f : a -> b exists then
    Tc(a) <= Tc(b). -/
structure TcAssignment (M : Type*) [Category M] where
  tc : M → NNReal
  monotone : ∀ {a b : M}, (a ⟶ b) → tc a ≤ tc b

/-- A TcAssignment defines a functor M -> NNReal (posetal category). -/
def TcAssignment.toFunctor {M : Type*} [Category M]
    (T : TcAssignment M) : M ⥤ NNReal where
  obj := T.tc
  map f := ⟨T.monotone f⟩

/-- Tc preserves identities. -/
theorem tc_preserves_id {M : Type*} [Category M]
    (T : TcAssignment M) (a : M) :
    T.toFunctor.map (𝟙 a) = 𝟙 (T.tc a) := by
  simp [TcAssignment.toFunctor]

/-- Tc preserves composition. -/
theorem tc_preserves_comp {M : Type*} [Category M]
    (T : TcAssignment M) {a b c : M} (f : a ⟶ b) (g : b ⟶ c) :
    T.toFunctor.map (f ≫ g) = T.toFunctor.map f ≫ T.toFunctor.map g := by
  simp [TcAssignment.toFunctor]

/-! ### Monotonicity along morphism chains -/

/-- Two composable morphisms: Tc(source) <= Tc(target of composite). -/
theorem tc_chain_bound {M : Type*} [Category M]
    (T : TcAssignment M) {a b c : M}
    (f : a ⟶ b) (g : b ⟶ c) :
    T.tc a ≤ T.tc c :=
  le_trans (T.monotone f) (T.monotone g)

/-- Tc at any source is bounded by any upper bound on the target. -/
theorem tc_bounded_by_endpoint {M : Type*} [Category M]
    (T : TcAssignment M) {a b : M}
    (f : a ⟶ b) (bound : NNReal)
    (hb : T.tc b ≤ bound) :
    T.tc a ≤ bound :=
  le_trans (T.monotone f) hb

/-! ### 2-Functor version (universe-monomorphic) -/

/-- A Tc 2-functor from a strict 2-category to PosetalRGeq.
    Since PosetalRGeq lives in a fixed universe, we work with
    concrete Strict2Cat instances. -/
theorem tc_2functor_preserves_id (C : Strict2Cat.{0,0,0})
    (F : Strict2Functor C PosetalRGeq) (a : C.Obj) :
    F.onMor (C.id_mor a) = PosetalRGeq.id_mor (F.onObj a) :=
  F.preserves_id a

theorem tc_2functor_preserves_comp (C : Strict2Cat.{0,0,0})
    (F : Strict2Functor C PosetalRGeq) {a b c : C.Obj}
    (f : C.Mor a b) (g : C.Mor b c) :
    F.onMor (C.comp_mor f g) =
    PosetalRGeq.comp_mor (F.onMor f) (F.onMor g) :=
  F.preserves_comp f g

end CatTax
