using OrdinaryDiffEq
using Trixi

####################################################

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

initial_condition = initial_condition_gauss

solver = FV(order = 2, surface_flux = flux_lax_friedrichs)

initial_refinement_level = 4

# TODO: There are no other cmesh functions implemented yet in 3d.
cmesh = Trixi.cmesh_new_quad_3d(coordinates_min = (-5.0, -5.0, -5.0),
                                coordinates_max = (5.0, 5.0, 5.0),
                                periodicity = (true, true, true))
mesh = T8codeMesh(cmesh, solver, initial_refinement_level = initial_refinement_level)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 10,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.7)

callbacks = CallbackSet(summary_callback, save_solution, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),#Euler(),
            dt = 5.0e-2, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback()

# Note: Since the mesh must be finalizized by hand in the elixir, it is not defined anymore here.
# Moved allocation test to the elixirs for now.
using Test
let
    t = sol.t[end]
    u_ode = sol.u[end]
    du_ode = similar(u_ode)
    @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
end

# Finalize `T8codeMesh` to make sure MPI related objects in t8code are
# released before `MPI` finalizes.
!isinteractive() && finalize(mesh)
