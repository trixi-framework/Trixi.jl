#src # Explicit time stepping

# For the time integration, [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) uses the package
# [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) from the SciML ecosystem.
# The interface to this package is the `solve(...)` function. It always requires an ODE problem and
# a time integration algorithm as input parameters.
# ````julia
# solve(ode, alg; kwargs...)
# ````
# In Trixi.jl, the ODE problem is created by `semidiscretize(semi, tspan)` for a semidiscretization
# `semi` and the time span `tspan`. In particular, [`semidiscretize`](@ref) returns an `ODEProblem`
# used by OrdinaryDiffEq.jl.

# OrdinaryDiffEq.jl provides many integration algorithms, which are summarized in
# the [documentation](https://diffeq.sciml.ai/stable/solvers/ode_solve/#Full-List-of-Methods).
# Particularly interesting for Trixi.jl are their
# [strong stability preserving (SSP) methods](https://diffeq.sciml.ai/stable/solvers/ode_solve/#Explicit-Strong-Stability-Preserving-Runge-Kutta-Methods-for-Hyperbolic-PDEs-(Conservation-Laws))
# and [low-storage methods](https://diffeq.sciml.ai/stable/solvers/ode_solve/#Low-Storage-Methods).
# There are some differences regarding the choice of the used time step.

# # [Error-based adaptive step sizes](@id adaptive_step_sizes)
# First, we treat time integration algorithms with adaptive step sizes, such as `SSPRK43`. It is used in
# some elixirs, like [`elixir_euler_colliding_flow.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_colliding_flow.jl)
# or [`elixir_euler_astro_jet_amr.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_astro_jet_amr.jl).

# Other error-based adaptive integration algorithms are for instance `RDPK3SpFSAL35`, `RDPK3Sp35`,
# `RDPK3SpFSAL49`, `RDPK3Sp49`, `RDPK3SpFSAL510`, `RDPK3Sp510`.

# They already contain an error-based adaptive step size control and heuristics to guess
# a starting step size. If this heuristic fails in your case, you can specify an appropriately
# small initial step size as keyword argument `dt=...` of `solve`.

# If you run Trixi in parallel with MPI you need to pass `internalnorm=ode_norm` and you should pass `unstable_check=ode_unstable_check`
# to enable MPI aware error-based adaptive step size control. These keyword arguments are also included in [`ode_default_options`](@ref).

# # CFL-based step size control
# The SciML ecosystem also provides time integration algorithms without adaptive time stepping on
# their own, such as `CarpenterKennedy2N54`. Moreover, you also can deactivate the automatic adaptivity
# of adaptive integration algorithms by passing `adaptive=false` in the `solve` function.

# These algorithms require another way of setting the step size. You have to pass `dt=...`
# in the `solve` function. Without other settings, the simulation uses this fixed time step.

# For hyperbolic PDEs, it is natural to use an adaptive CFL-based step size control. Here, the time
# step is proportional to a ratio of the local measure of mesh spacing $\Delta x_i$ for an element `i`
# and the maximum (local) wave speed $\lambda_{\max}$ related to the largest-magnitude eigenvalue of
# the flux Jacobian of the hyperbolic system.
# ```math
# \Delta t_n = \text{CFL} * \min_i \frac{\Delta x_i}{\lambda_{\max}(u_i^n)}
# ```
# We compute $\Delta x_i$ by scaling the element size by a factor of $1/(N+1)$, cf.
# [Gassner and Kopriva (2011)](https://doi.org/10.1137/100807211), Section 5.

# Trixi.jl provides such a CFL-based step size control. It is implemented as the callback
# [`StepsizeCallback`](@ref).
# ````julia
# stepsize_callback = StepsizeCallback(; cfl=1.0)
# ````
# A suitable CFL number depends on many parameters such as the chosen grid, the integration
# algorithm and the polynomial degree of the spatial DG discretization. So, the optimal number
# for an example is mostly determined experimentally.

# You can add this CFL-based step size control to your simulation like any other callback.
# ````julia
# callbacks = CallbackSet(stepsize_callback)
# alg = CarpenterKennedy2N54(williamson_condition=false)
# solve(ode, alg;
#       dt=1.0 # solve needs some value here but it will be overwritten by the stepsize_callback
#       callback=callbacks)
# ````

# You can find simple examples with a CFL-based step size control for instance in the elixirs
# [`elixir_advection_basic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_advection_basic.jl)
# or [`elixir_euler_source_terms.jl`](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_2d_dgsem/elixir_euler_source_terms.jl).

# ## Package versions

# These results were obtained using the following versions.

using InteractiveUtils
versioninfo()

using Pkg
Pkg.status(["Trixi", "OrdinaryDiffEq"],
           mode = PKGMODE_MANIFEST)
