
using OrdinaryDiffEq
using Trixi

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_convergence_test

# boundary_condition = BoundaryConditionDirichlet(initial_condition)
# boundary_conditions = Dict(:all => boundary_condition)

source_terms = source_terms_convergence_test

solver = FV(order = 2, extended_reconstruction_stencil = false,
            surface_flux = flux_lax_friedrichs)

# TODO: There are no other cmesh functions implemented yet in 3d.
cmesh = Trixi.cmesh_new_quad_3d(periodicity = (true, true, true))
mesh = T8codeMesh(cmesh, solver,
                  initial_refinement_level = 3)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms)
# boundary_conditions = boundary_conditions)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary

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
