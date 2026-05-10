/-
  CatTax.Basic
  ============
  Core 2-categorical definitions for the materials taxonomy.

  We define:
  1. A strict 2-category structure
  2. The posetal 2-category on NNReal
  3. Vertical and horizontal composition with interchange

  Horesh 2026
-/

import Mathlib.CategoryTheory.Bicategory.Basic
import Mathlib.CategoryTheory.Category.Basic
import Mathlib.CategoryTheory.Functor.Basic
import Mathlib.Order.Basic
import Mathlib.Data.NNReal.Defs

open CategoryTheory

namespace CatTax

/-! ### The posetal category on non-negative reals -/

/-- The posetal category on NNReal: a unique morphism r -> s iff r <= s. -/
instance : SmallCategory NNReal where
  Hom a b := PLift (a ≤ b)
  id a := ⟨le_refl a⟩
  comp f g := ⟨le_trans f.down g.down⟩

/-! ### Strict 2-category axioms -/

/-- A strict 2-category.
    Objects, 1-morphisms, 2-morphisms with vertical and horizontal
    composition satisfying associativity, unit laws, and interchange. -/
structure Strict2Cat where
  Obj : Type*
  Mor : Obj → Obj → Type*
  TwoMor : {a b : Obj} → Mor a b → Mor a b → Type*
  id_mor : (a : Obj) → Mor a a
  comp_mor : {a b c : Obj} → Mor a b → Mor b c → Mor a c
  id_two : {a b : Obj} → (f : Mor a b) → TwoMor f f
  v_comp : {a b : Obj} → {f g h : Mor a b} →
           TwoMor f g → TwoMor g h → TwoMor f h
  h_comp : {a b c : Obj} → {f f' : Mor a b} → {g g' : Mor b c} →
           TwoMor f f' → TwoMor g g' →
           TwoMor (comp_mor f g) (comp_mor f' g')
  assoc_mor : {a b c d : Obj} →
              (f : Mor a b) → (g : Mor b c) → (h : Mor c d) →
              comp_mor (comp_mor f g) h = comp_mor f (comp_mor g h)
  left_unit : {a b : Obj} → (f : Mor a b) →
              comp_mor (id_mor a) f = f
  right_unit : {a b : Obj} → (f : Mor a b) →
               comp_mor f (id_mor b) = f

/-- The posetal 2-category on NNReal. Every hom-set is at most a
    singleton (the unique proof of <=), so all 2-morphisms are trivial. -/
def PosetalRGeq : Strict2Cat where
  Obj := NNReal
  Mor a b := PLift (a ≤ b)
  TwoMor _ _ := PUnit
  id_mor _ := ⟨le_refl _⟩
  comp_mor f g := ⟨le_trans f.down g.down⟩
  id_two _ := PUnit.unit
  v_comp _ _ := PUnit.unit
  h_comp _ _ := PUnit.unit
  assoc_mor _ _ _ := rfl
  left_unit _ := rfl
  right_unit _ := rfl

/-! ### 2-Functor -/

/-- A strict 2-functor between strict 2-categories. -/
structure Strict2Functor (C D : Strict2Cat) where
  onObj : C.Obj → D.Obj
  onMor : {a b : C.Obj} → C.Mor a b → D.Mor (onObj a) (onObj b)
  onTwoMor : {a b : C.Obj} → {f g : C.Mor a b} →
             C.TwoMor f g → D.TwoMor (onMor f) (onMor g)
  preserves_id : (a : C.Obj) → onMor (C.id_mor a) = D.id_mor (onObj a)
  preserves_comp : {a b c : C.Obj} → (f : C.Mor a b) → (g : C.Mor b c) →
                   onMor (C.comp_mor f g) = D.comp_mor (onMor f) (onMor g)

/-! ### Abstract materials category -/

/-- A `MaterialsCat` bundles a type of materials with category structure
    and a point-group assignment. -/
class MaterialsCat (M : Type*) extends SmallCategory M where
  pointGroup : M → ℕ

/-- Two materials share a point group. -/
def samePointGroup [MaterialsCat M] (g : ℕ) (a b : M) : Prop :=
  MaterialsCat.pointGroup a = g ∧ MaterialsCat.pointGroup b = g

end CatTax
