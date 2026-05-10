import Lake
open Lake DSL

package «cattax»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

@[default_target]
lean_lib «CatTax» where
  roots := #[`CatTax.Basic, `CatTax.TcFunctor, `CatTax.ClassInvariant]
