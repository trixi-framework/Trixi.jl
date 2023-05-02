# Changelog

Trixi.jl follows the interpretation of [semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file
for human readability.

## Changes in the v0.5 lifecycle

#### Added

- Experimental support for 3D parabolic diffusion terms has been added.

#### Changed

- The required Julia version is updated to v1.8 in Trixi.jl v0.5.13.

#### Deprecated

- The macro `@unpack` (re-exported originally from UnPack.jl) is deprecated and
  will be removed. Consider using Julia's standard destructuring syntax
  `(; a, b) = stuff` instead of `@unpack a, b = stuff`.
- The constructor `DGMultiMesh(dg; cells_per_dimension, kwargs...)` is deprecated
  and will be removed. The new constructor `DGMultiMesh(dg, cells_per_dimension; kwargs...)`
  does not have `cells_per_dimesion` as a keyword argument.

#### Removed


## Changes when updating to v0.5 from v0.4.x

#### Added

#### Changed

- Compile-time boolean indicators have been changed from `Val{true}`/`Val{false}`
  to `Trixi.True`/`Trixi.False`. This affects user code only if new equations
  with nonconservative terms are created. Change
  `Trixi.has_nonconservative_terms(::YourEquations) = Val{true}()` to
  `Trixi.has_nonconservative_terms(::YourEquations) = Trixi.True()`.
- The (non-exported) DGSEM function `split_form_kernel!` has been renamed to `flux_differencing_kernel!`
- Trixi.jl updated its dependency [P4est.jl](https://github.com/trixi-framework/P4est.jl/)
  from v0.3 to v0.4. The new bindings of the C library `p4est` have been
  generated using Clang.jl instead of CBinding.jl v0.9. This affects only user
  code that is interacting directly with `p4est`, e.g., because custom refinement
  functions have been passed to `p4est`. Please consult the
  [NEWS.md of P4est.jl](https://github.com/trixi-framework/P4est.jl/blob/main/NEWS.md)
  for further information.

#### Deprecated

- The signature of the `DGMultiMesh` constructors has changed - the `dg::DGMulti`
  argument now comes first.
- The undocumented and unused
  `DGMultiMesh(triangulateIO, rd::RefElemData{2, Tri}, boundary_dict::Dict{Symbol, Int})`
  constructor was removed.

#### Removed

- Everything deprecated in Trixi.jl v0.4.x has been removed.


## Changes in the v0.4 lifecycle

#### Added

- Implementation of linearized Euler equations in 2D
- Experimental support for upwind finite difference summation by parts (FDSBP)
  has been added in Trixi.jl v0.4.55. The first implementation requires a `TreeMesh` and comes
  with several examples in the `examples_dir()` of Trixi.jl.
- Experimental support for 2D parabolic diffusion terms has been added.
  * `LaplaceDiffusion2D` and `CompressibleNavierStokesDiffusion2D` can be used to add
  diffusion to systems. `LaplaceDiffusion2D` can be used to add scalar diffusion to each
  equation of a system, while `CompressibleNavierStokesDiffusion2D` can be used to add
  Navier-Stokes diffusion to `CompressibleEulerEquations2D`.
  * Parabolic boundary conditions can be imposed as well. For `LaplaceDiffusion2D`, both
  `Dirichlet` and `Neumann` conditions are supported. For `CompressibleNavierStokesDiffusion2D`,
  viscous no-slip velocity boundary conditions are supported, along with adiabatic and isothermal
  temperature boundary conditions. See the boundary condition container
  `BoundaryConditionNavierStokesWall` and boundary condition types `NoSlip`, `Adiabatic`, and
  `Isothermal` for more information.
  * `CompressibleNavierStokesDiffusion2D` can utilize both primitive variables (which are not
  guaranteed to provably dissipate entropy) and entropy variables (which provably dissipate
  entropy at the semi-discrete level).
  * Please check the `examples` directory for further information about the supported setups.
    Further documentation will be added later.
- Numerical fluxes `flux_shima_etal_turbo` and `flux_ranocha_turbo` that are
  equivalent to their non-`_turbo` counterparts but may enable specialized
  methods making use of SIMD instructions to increase runtime efficiency
- Support for (periodic and non-periodic) SBP operators of
  [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl)
  as approximation type in `DGMulti` solvers
- Initial support for MPI-based parallel simulations using non-conforming meshes of type `P4estMesh`
  in 2D and 3D including adaptive mesh refinement

#### Removed

- The `VertexMappedMesh` type is removed in favor of the `DGMultiMesh` type.
  The `VertexMappedMesh` constructor is deprecated.

#### Changed

- The required Julia version is updated to v1.7.
- The isentropic vortex setups contained a bug that was fixed in Trixi.jl v0.4.54.
  Moreover, the setup was made a bit more challenging. See
  https://github.com/trixi-framework/Trixi.jl/issues/1269 for further
  information.

#### Deprecated

- The `DGMultiMesh` constructor which uses a `rd::RefElemData` argument is deprecated in
  favor of the constructor which uses a `dg::DGMulti` argument instead.

## Changes when updating to v0.4 from v0.3.x

#### Added

- Experimental support for artificial neural network-based indicators for shock capturing and
  adaptive mesh refinement ([#632](https://github.com/trixi-framework/Trixi.jl/pull/632))
- Experimental support for direct-hybrid aeroacoustics simulations
  ([#712](https://github.com/trixi-framework/Trixi.jl/pull/712))
- Implementation of shallow water equations in 2D
- Experimental support for interactive visualization with [Makie.jl](https://makie.juliaplots.org/)

#### Changed

- Implementation of acoustic perturbation equations now uses the conservative form, i.e. the
  perturbed pressure `p_prime` has been replaced with `p_prime_scaled = p_prime / c_mean^2`.
- Removed the experimental `BoundaryConditionWall` and instead directly compute slip wall boundary
  condition flux term using the function `boundary_condition_slip_wall`.
- Renamed `advectionvelocity` in `LinearScalarAdvectionEquation` to `advection_velocity`.
- The signature of indicators used for adaptive mesh refinement (AMR) and shock capturing
  changed to generalize them to curved meshes.

#### Deprecated

#### Removed

- Many initial/boundary conditions and source terms for typical setups were
  moved from `Trixi/src` to the example elixirs `Trixi/examples`. Thus, they
  are no longer available when `using Trixi`, e.g., the initial condition
  for the Kelvin Helmholtz instability.
- Features deprecated in v0.3 were removed.


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
  (released in Trixi.jl v0.3.22).
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
