# Changelog

Trixi.jl follows the interpretation of [semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file
for human readability.


## Changes when updating to v0.4 from v0.3.x

#### Added

- Experimental support for artifial neural network-based indicators for shock capturing and
  adaptive mesh refinement ([#632](https://github.com/trixi-framework/Trixi.jl/pull/632))

#### Changed

- Implementation of acoustic perturbation equations now uses the conservative form, i.e. the
  perturbed pressure `p_prime` has been replaced with `p_prime_scaled = p_prime / c_mean^2`.

#### Deprecated

#### Removed

- Many initial/boundary conditions and source terms for typical setups were
  moved from `Trixi/src` to the example elixirs `Trixi/examples`. Thus, they
  are no longer available when `using Trixi`, e.g., the initial condition
  for the Kelvin Helmholtz instability.
- Some initial/boundary conditions and source terms for academic verification
  setups were removed, e.g., `initial_condition_linear_x` for the 2D linear
  advection equation.


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
