using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation

diffusivity() = 1.0e-3
advection_velocity = (1.0, 0.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

# Define initial condition (copied from "examples/tree_1d_dgsem/elixir_advection_diffusion.jl")
function initial_condition_bubble(x, t, equation)
    return SVector((x[1]-pi) * (x[2]-pi) * (x[1]+pi) * (x[2]+pi))
end

initial_condition = initial_condition_bubble

boundary_conditions = Dict(:x_neg => BoundaryConditionDirichlet(initial_condition),
                           :y_neg => BoundaryConditionDirichlet(initial_condition),
                           :y_pos => BoundaryConditionDirichlet(initial_condition),
                           :x_pos => boundary_condition_do_nothing)

boundary_conditions_parabolic = Dict(:x_neg => BoundaryConditionDirichlet(initial_condition), 
                                     :x_pos => BoundaryConditionDirichlet(initial_condition), 
                                     :y_neg => BoundaryConditionDirichlet(initial_condition), 
                                     :y_pos => BoundaryConditionDirichlet(initial_condition))

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=2, surface_flux=flux_lax_friedrichs)

coordinates_min = (-pi, -pi) # minimum coordinates (min(x), min(y))
coordinates_max = ( pi,  pi) # maximum coordinates (max(x), max(y))

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=2, initial_refinement_level=2,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 periodicity=false)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), 
                                             initial_condition, solver, 
                                             boundary_conditions = (boundary_conditions, 
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, .5)
ode = semidiscretize(semi, tspan)

# u = sol.u[end]

# du = similar(u)
# Trixi.rhs_parabolic!(du, u, semi, 0.0)

# x, y = [semi.cache.elements.node_coordinates[i, :, :, :] for i in 1:2]
# for i in eachindex(x)
#     u[i] = initial_condition_bubble((x[i], y[i]), 0.0, equations)[1]
# end
# u = Trixi.wrap_array(u, semi)
# fill!(cache_parabolic.elements.surface_flux_values, NaN);
# dg = solver
# parabolic_scheme = semi.solver_parabolic
# t = 0.0
# (; cache, cache_parabolic, boundary_conditions_parabolic) = semi
# @unpack viscous_container = cache_parabolic
# @unpack u_transformed, gradients, flux_viscous = viscous_container

# Trixi.transform_variables!(u_transformed, u, mesh, equations_parabolic,
#                            dg, parabolic_scheme, cache, cache_parabolic)

# Trixi.calc_gradient!(gradients, u_transformed, t, mesh, equations_parabolic,
#                      boundary_conditions_parabolic, dg, cache, cache_parabolic)

# grad_x, grad_y = gradients
# @show any(isnan.(grad_x))
# @show any(isnan.(grad_y))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval=analysis_interval)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                      base_level=1,
                                      med_level=2, med_threshold=0.5,
                                      max_level=3, max_threshold=0.75)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=5)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, amr_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
time_int_tol = 1.0e-11
sol = solve(ode, dt = 1e-7, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks)

# Print the timer summary
summary_callback()
plot(sol)
# pd = PlotData2D(sol)
# plot!(getmesh(pd))

