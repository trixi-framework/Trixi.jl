#src # Custom semidiscretizations

# As described in the [overview section](@ref overview-semidiscretizations),
# semidiscretizations are high-level descriptions of spatial discretizations
# in Trixi.jl. Trixi.jl's main focus is on hyperbolic conservation
# laws represented in a [`SemidiscretizationHyperbolic`](@ref).
# Hyperbolic-parabolic problems based on the advection-diffusion equation or
# the compressible Navier-Stokes equations can be represented in a
# [`SemidiscretizationHyperbolicParabolic`](@ref). This is described in the
# [basic tutorial on parabolic terms](@ref parabolic_terms) and its extension to
# [custom parabolic terms](@ref adding_new_parabolic_terms).
# In this tutorial, we will describe how these semidiscretizations work and how
# they can be used to create custom semidiscretizations involving also other tasks.


# ## Overview of the right-hand side evaluation

# The semidiscretizations provided by Trixi.jl are set up to create `ODEProblem`s from
# [SciML ecosystem for ordinary differential equations](https://diffeq.sciml.ai/latest/).
# In particular, a spatial semidiscretization can be wrapped in an ODE problem
# using [`semidiscretize`](@ref), which returns an `ODEProblem`. This `ODEProblem`
# bundles an initial condition, a right-hand side (RHS) function, the time span,
# and possible parameters. The `ODEProblem`s created by Trixi.jl use the semidiscretization
# passed to [`semidiscretize`](@ref) as parameter.
# For a [`SemidiscretizationHyperbolic`](@ref), the `ODEProblem` wraps
# `Trixi.rhs!` as ODE RHS.
#


# ## Setting up a custom semidiscretization

# Required
# - `Trixi.rhs!(du_ode, u_ode, semi::SemidiscretizationEulerGravity, t)`
# - `Trixi.mesh_equations_solver_cache(semi::SemidiscretizationEulerGravity)`

# Basic
# - `Base.show(io::IO, parameters::SemidiscretizationEulerGravity)`
# - `Base.show(io::IO, ::MIME"text/plain", parameters::SemidiscretizationEulerGravity)`
# - `Base.ndims(semi::SemidiscretizationEulerGravity)`
# - `Base.real(semi::SemidiscretizationEulerGravity)`
# - `Trixi.compute_coefficients(t, semi::SemidiscretizationEulerGravity)`
# - `Trixi.compute_coefficients!(u_ode, t, semi::SemidiscretizationEulerGravity)`
# - `Trixi.calc_error_norms(func, u, t, analyzer, semi::SemidiscretizationEulerGravity, cache_analysis)`

# Advanced
# - `Trixi.nvariables(semi::SemidiscretizationHyperbolicParabolic)`
# - `Trixi.save_solution_file(u_ode, t, dt, iter, semi::SemidiscretizationEulerGravity, solution_callback, element_variables = Dict{Symbol, Any}(); system = "")`
# - `(amr_callback::AMRCallback)(u_ode, semi::SemidiscretizationEulerGravity, t, iter; kwargs...)`
# - `Trixi.semidiscretize(semi::SemidiscretizationHyperbolicParabolic, tspan; reset_threads = true)`
