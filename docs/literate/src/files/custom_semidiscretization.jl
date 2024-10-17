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

# The semidiscretizations provided by Trixi.jl are set up to create `ODEProblem`s from the
# [SciML ecosystem for ordinary differential equations](https://diffeq.sciml.ai/latest/).
# In particular, a spatial semidiscretization can be wrapped in an ODE problem
# using [`semidiscretize`](@ref), which returns an `ODEProblem`. This `ODEProblem`
# bundles an initial condition, a right-hand side (RHS) function, the time span,
# and possible parameters. The `ODEProblem`s created by Trixi.jl use the semidiscretization
# passed to [`semidiscretize`](@ref) as a parameter.
# For a [`SemidiscretizationHyperbolic`](@ref), the `ODEProblem` wraps
# `Trixi.rhs!` as ODE RHS.
# For a [`SemidiscretizationHyperbolicParabolic`](@ref),  Trixi.jl
# uses a `SplitODEProblem` combining `Trixi.rhs_parabolic!` for the
# (potentially) stiff part and `Trixi.rhs!` for the other part.

# ## Standard Trixi.jl setup

# In this tutorial, we will consider the linear advection equation
# with source term
# ```math
# \partial_t u(t,x) + \partial_x u(t,x) = -\exp(-t) \sin\bigl(\pi (x - t) \bigr)
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

plot(sol; label = "numerical sol.", legend = :topright)

# We can also plot the analytical solution for comparison.
# Since Trixi.jl uses `SVector`s for the variables, we take their `first`
# (and only) component to get the scalar value for manual plotting.

let
    x = range(-1.0, 1.0; length = 200)
    plot!(x, first.(initial_condition.(x, sol.t[end], equations)),
          label = "analytical sol.", linestyle = :dash, legend = :topright)
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

# Next, we will solve the same problem but use our own ODE RHS function.
# To demonstrate this, we will artificially create a global variable
# containing the current time of the simulation.

const GLOBAL_TIME = Ref(0.0)

function source_terms_custom(u, x, t, equations)
    t = GLOBAL_TIME[]
    return -initial_condition(x, t, equations)
end

# Next, we create our own RHS function to update the global time of
# the simulation before calling the RHS function from Trixi.jl.

function rhs_source_custom!(du_ode, u_ode, semi, t)
    GLOBAL_TIME[] = t
    Trixi.rhs!(du_ode, u_ode, semi, t)
end

# Next, we create an `ODEProblem` manually copying over the data from
# the one we got from [`semidiscretize`](@ref) earlier.

ode_source_custom = ODEProblem(rhs_source_custom!,
                               ode.u0,
                               ode.tspan,
                               ode.p) # semi
sol_source_custom = solve(ode_source_custom, RDPK3SpFSAL49();
                          ode_default_options()...)

plot(sol_source_custom; label = "numerical sol.")
let
    x = range(-1.0, 1.0; length = 200)
    plot!(x, first.(initial_condition.(x, sol_source_custom.t[end], equations)),
          label = "analytical sol.", linestyle = :dash, legend = :topleft)
end
plot!(sol_source_custom.u[1], semi, label = "u0", linestyle = :dot, legend = :topleft)

# This also works with callbacks as usual.

summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi; interval = analysis_interval)
alive_callback = AliveCallback(; analysis_interval)
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

sol = solve(ode_source_custom, RDPK3SpFSAL49();
            ode_default_options()..., callback = callbacks)
summary_callback()

# ## Setting up a custom semidiscretization

# Using a global constant is of course not really nice from a software
# engineering point of view. Thus, it can often be useful to collect
# additional data in the parameters of the `ODEProblem`. Thus, it is
# time to create our own semidiscretization. Here, we create a small
# wrapper of a standard semidiscretization of Trixi.jl and the current
# global time of the simulation.

struct CustomSemidiscretization{Semi, T} <: Trixi.AbstractSemidiscretization
    semi::Semi
    t::T
end

semi_custom = CustomSemidiscretization(semi, Ref(0.0))

# To get pretty printing in the REPL, you can consider specializing
#
# - `Base.show(io::IO, parameters::CustomSemidiscretization)`
# - `Base.show(io::IO, ::MIME"text/plain", parameters::CustomSemidiscretization)`
#
# for your custom semidiscretiation.

# Next, we create our own source terms that use the global time stored
# in the custom semidiscretiation.

source_terms_custom_semi = let semi_custom = semi_custom
    function source_terms_custom_semi(u, x, t, equations)
        t = semi_custom.t[]
        return -initial_condition(x, t, equations)
    end
end

# We also create a custom ODE RHS to update the current global time
# stored in the custom semidiscretization. We unpack the standard
# semidiscretization created by Trixi.jl and pass it to `Trixi.rhs!`.

function rhs_semi_custom!(du_ode, u_ode, semi_custom, t)
    semi_custom.t[] = t
    Trixi.rhs!(du_ode, u_ode, semi_custom.semi, t)
end

# Finally, we set up an `ODEProblem` and solve it numerically.

ode_semi_custom = ODEProblem(rhs_semi_custom!,
                             ode.u0,
                             ode.tspan,
                             semi_custom)
sol_semi_custom = solve(ode_semi_custom, RDPK3SpFSAL49();
                        ode_default_options()...)

# If we want to make use of additional functionality provided by
# Trixi.jl, e.g., for plotting, we need to implement a few additional
# specializations. In this case, we forward everything to the standard
# semidiscretization provided by Trixi.jl wrapped in our custom
# semidiscretization.

Base.ndims(semi::CustomSemidiscretization) = ndims(semi.semi)
function Trixi.mesh_equations_solver_cache(semi::CustomSemidiscretization)
    Trixi.mesh_equations_solver_cache(semi.semi)
end

# Now, we can plot the numerical solution as usual.

plot(sol_semi_custom; label = "numerical sol.")
let
    x = range(-1.0, 1.0; length = 200)
    plot!(x, first.(initial_condition.(x, sol_semi_custom.t[end], equations)),
          label = "analytical sol.", linestyle = :dash, legend = :topleft)
end
plot!(sol_semi_custom.u[1], semi, label = "u0", linestyle = :dot, legend = :topleft)

# This also works with many callbacks as usual. However, the
# [`AnalysisCallback`](@ref) requires some special handling since it
# makes use of a performance counter contained in the standard
# semidiscretizations of Trixi.jl to report some
# [performance metrics](@ref performance-metrics).
# Here, we forward all accesses to the performance counter to the
# wrapped semidiscretization.

function Base.getproperty(semi::CustomSemidiscretization, s::Symbol)
    if s === :performance_counter
        wrapped_semi = getfield(semi, :semi)
        wrapped_semi.performance_counter
    else
        getfield(semi, s)
    end
end

# Moreover, the [`AnalysisCallback`](@ref) also performs some error
# calculations. We also need to forward them to the wrapped
# semidiscretization.

function Trixi.calc_error_norms(func, u, t, analyzer,
                                semi::CustomSemidiscretization,
                                cache_analysis)
    Trixi.calc_error_norms(func, u, t, analyzer,
                           semi.semi,
                           cache_analysis)
end

# Now, we can work with the callbacks used before as usual.

summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi_custom;
                                     interval = analysis_interval)
alive_callback = AliveCallback(; analysis_interval)
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

sol = solve(ode_semi_custom, RDPK3SpFSAL49();
            ode_default_options()..., callback = callbacks)
summary_callback()

# For even more advanced usage of custom semidiscretizations, you
# may look at the source code of the ones contained in Trixi.jl, e.g.,
# - [`SemidiscretizationHyperbolicParabolic`](@ref)
# - [`SemidiscretizationEulerGravity`](@ref)
# - [`SemidiscretizationEulerAcoustics`](@ref)
# - [`SemidiscretizationCoupled`](@ref)

# ## Package versions

# These results were obtained using the following versions.

using InteractiveUtils
versioninfo()

using Pkg
Pkg.status(["Trixi", "OrdinaryDiffEq", "Plots"],
           mode = PKGMODE_MANIFEST)
