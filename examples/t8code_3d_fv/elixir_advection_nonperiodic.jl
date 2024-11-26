using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the linear advection equation.

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

initial_condition = initial_condition_convergence_test

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:all => boundary_condition)
# Problem: T8codeMesh interface with parameter cmesh cannot distinguish between boundaries
# boundary_conditions = Dict(:x_neg => boundary_condition,
#                            :x_pos => boundary_condition,
#                            :y_neg => boundary_condition_periodic,
#                            :y_pos => boundary_condition_periodic)

solver = FV(order = 2, surface_flux = flux_lax_friedrichs)

# TODO: When using mesh construction as in elixir_advection_basic.jl boundary Symbol :all is not defined
initial_refinement_level = 5
cmesh = Trixi.cmesh_new_quad_3d(periodicity = (false, false, false))
mesh = T8codeMesh(cmesh, solver, initial_refinement_level = initial_refinement_level)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# Run the simulation.

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # Solve needs some value here but it will be overwritten by the stepsize_callback.
            save_everystep = false, callback = callbacks);
summary_callback()

# Note: Since the mesh must be finalizized by hand in the elixir, it is not defined anymore here.
# Moved allocation test to the elixirs for now.
let
    t = sol.t[end]
    u_ode = sol.u[end]
    du_ode = similar(u_ode)
    @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
end

# Finalize `T8codeMesh` to make sure MPI related objects in t8code are
# released before `MPI` finalizes.
!isinteractive() && finalize(mesh)
