# Changelog

Trixi.jl follows the interpretation of [semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file
for human readability.


## Unreleased

#### Added
#### Changed
#### Removed
#### Deprecated


## Changes in the v0.3 lifecycle

#### Added

- Support for automatic differentiation, e.g. `jacobian_ad_forward`

#### Changed

#### Removed

### Deprecated

- `calcflux` → `flux` (https://github.com/trixi-framework/Trixi.jl/pull/463)
- `flux_upwind` → `flux_godunov`
- Providing the keyword argument `solution_variables` of `SaveSolutionCallback`
  as `Symbol` is deprecated in favor of using functions like `cons2cons` and
  `cons2prim`
- `varnames_cons(equations)` → `varnames(cons2cons, equations)`
- `varnames_prim(equations)` → `varnames(cons2prim, equations)`
