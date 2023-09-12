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
# For a [`SemidiscretizationHyperbolicParabolic`](@ref),  Trixi.jl
# uses a `SplitODEProblem` combining `Trixi.rhs_parabolic!` for the
# (potentially) stiff part and `Trixi.rhs!` for the other part.


# ## Standard Trixi.jl setup

# In this tutorial, we will consider the linear advection equation
# with source term
# ```math
# \partial_t u(t,x) + \partial_x u(t,x) = -\exp(-t) sin\bigl(\pi (x - t) \bigr)
# ```
# with periodic boundary conditions in the domain `[-1, 1]` as a
# model problem.
# The initial condition is
# ```math
# u(0,x) = \sin(\pi x).
# ```
# The source term results in some damping and the analytical solution
# ```math
# u(t,x) = \exp(-t) \sin\bigl(\pi (x - t) \bigr).
# ```
# First, we discretize this equation using the standard functionality
# of Trixi.jl.

using Trixi, OrdinaryDiffEq, Plots

# The linear scalar advection equation is already implemented in
# Trixi.jl as [`LinearScalarAdvectionEquation1D`](@ref). We construct
# it with an advection velocity `1.0`.

equations = LinearScalarAdvectionEquation1D(1.0)

# Next, we use a standard [`DGSEM`](@ref) solver.

solver = DGSEM(polydeg = 3)

# We create a simple [`TreeMesh`](@ref) in 1D.

coordinates_min = (-1.0,)
coordinates_max = (+1.0,)
mesh = TreeMesh(coordinates_min, coordinates_max;
                initial_refinement_level = 4,
                n_cells_max = 10^4,
                periodicity = true)

# We wrap everything in in a semidiscretization and pass the source
# terms as a standard Julia function. Please note that Trixi.jl uses
# `SVector`s from
# [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
# to store the conserved variables `u`. Thus, the return value of the
# source terms must be wrapped in an `SVector` - even if we consider
# just a scalar problem.

function initial_condition(x, t, equations)
    return SVector(exp(-t) * sinpi(x[1] - t))
end

function source_terms_standard(u, x, t, equations)
    return -initial_condition(x, t, equations)
end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                    solver;
                                    source_terms = source_terms_standard)

# Now, we can create the `ODEProblem`, solve the resulting ODE
# using a time integration method from
# [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl),
# and visualize the numerical solution at the final time using
# [Plots.jl](https://github.com/JuliaPlots/Plots.jl).

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

sol = solve(ode, RDPK3SpFSAL49(); ode_default_options()...)

plot(sol; label = "numerical sol.")

# We can also plot the analytical solution for comparison.
# Since Trixi.jl uses `SVector`s for the variables, we take their `first`
# (and only) component to get the scalar value for manual plotting.

let
   x = range(-1.0, 1.0; length = 200)
   plot!(x, first.(initial_condition.(x, sol.t[end], equations)),
         label = "analytical sol.", linestyle = :dash, legend = :topleft)
end

# We can also add the initial condition to the plot.

plot!(sol.u[1], semi, label = "u0", linestyle = :dot, legend = :topleft)

# You can of course also use some
# [callbacks](https://trixi-framework.github.io/Trixi.jl/stable/callbacks/)
# provided by Trixi.jl as usual.

summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi; interval = analysis_interval)
alive_callback = AliveCallback(; analysis_interval)
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

sol = solve(ode, RDPK3SpFSAL49();
            ode_default_options()..., callback = callbacks)
summary_callback()


# ## Using a custom ODE right-hand side function

# TODO


# ## Setting up a custom semidiscretization

# TODO

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
