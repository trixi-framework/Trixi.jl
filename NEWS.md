# Changelog

Trixi.jl follows the interpretation of [semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file
for human readability.


## Changes in the v0.3 lifecycle

#### Added

- Support for automatic differentiation, e.g. `jacobian_ad_forward`
- In-situ visualization and post hoc visualization with Plots.jl
- New systems of equations
  - multicomponent compressible Euler and MHD equations
  - acoustic perturbation equations
  - Lattice-Boltzmann equations
- Composable `FluxPlusDissipation` and `FluxLaxFriedrichs()`, `FluxHLL()` with adaptable
  wave speed estimates were added in [#493](https://github.com/trixi-framework/Trixi.jl/pull/493)
- New structured, curvilinear, conforming mesh type `StructuredMesh`
- New unstructured, curvilinear, conforming mesh type `UnstructuredMesh2D` in 2D
- New unstructured, curvilinear, adaptive (non-conforming) mesh type `P4estMesh` in 2D and 3D
- Experimental support for finite difference (FD) summation-by-parts (SBP) methods via
  [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl)
- New support for modal DG and SBP-DG methods on triangular and tetrahedral meshes via [StartUpDG.jl](https://github.com/jlchan/StartUpDG.jl)

#### Changed

- `flux_lax_friedrichs(u_ll, u_rr, orientation, equations::LatticeBoltzmannEquations2D)` and
  `flux_lax_friedrichs(u_ll, u_rr, orientation, equations::LatticeBoltzmannEquations3D)`
  were actually using the logic of `flux_godunov`. Thus, they were renamed accordingly
  in [#493](https://github.com/trixi-framework/Trixi.jl/pull/493). This is considered a bugfix
  (released in Trixi v0.3.22).
- The required Julia version is updated to v1.6.

#### Deprecated

- `calcflux` → `flux` ([#463](https://github.com/trixi-framework/Trixi.jl/pull/463))
- `flux_upwind` → `flux_godunov`
- `flux_hindenlang` → `flux_hindenlang_gassner`
- Providing the keyword argument `solution_variables` of `SaveSolutionCallback`
  as `Symbol` is deprecated in favor of using functions like `cons2cons` and
  `cons2prim`
- `varnames_cons(equations)` → `varnames(cons2cons, equations)`
- `varnames_prim(equations)` → `varnames(cons2prim, equations)`
- The old interface for nonconservative terms is deprecated. In particular, passing
  only a single two-point numerical flux for nonconservative is deprecated. The new
  interface is described in a tutorial. Now, a tuple of two numerical fluxes of the
  form `(conservative_flux, nonconservative_flux)` needs to be passed for
  nonconservative equations, see [#657](https://github.com/trixi-framework/Trixi.jl/pull/657).

#### Removed
