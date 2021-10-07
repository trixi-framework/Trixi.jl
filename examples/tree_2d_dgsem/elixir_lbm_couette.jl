
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the Lattice-Boltzmann equations for the D2Q9 scheme

equations = LatticeBoltzmannEquations2D(Ma=0.05, Re=2000)

initial_condition = initial_condition_couette_unsteady
boundary_conditions = (
                       x_neg=boundary_condition_periodic,
                       x_pos=boundary_condition_periodic,
                       y_neg=boundary_condition_noslip_wall,
                       y_pos=boundary_condition_couette,
                      )

solver = DGSEM(polydeg=3, surface_flux=flux_godunov)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                periodicity=(true, false),
                n_cells_max=10_000,)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 40.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

# Custom solution variables: normalize velocities by reference speed `u0`
@inline function macroscopic_normalized(u, equations::LatticeBoltzmannEquations2D)
  macroscopic = cons2macroscopic(u, equations)
  rho, v1, v2, p = macroscopic

  # Use `typeof(macroscopic)` to avoid having to explicitly add `using StaticArrays`
  convert(typeof(macroscopic), (rho, v1/equations.u0, v2/equations.u0, p))
end
Trixi.varnames(::typeof(macroscopic_normalized), equations::LatticeBoltzmannEquations2D) = ("rho", "v1_normalized", "v2_normalized", "p")

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=macroscopic_normalized)

stepsize_callback = StepsizeCallback(cfl=1.0)

collision_callback = LBMCollisionCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        collision_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
