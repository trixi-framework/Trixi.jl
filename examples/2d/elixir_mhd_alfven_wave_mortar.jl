
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
gamma = 5/3
equations = IdealGlmMhdEquations2D(gamma)

initial_conditions = initial_conditions_convergence_test

surface_flux = flux_hll
volume_flux  = flux_derigs_etal
solver = DGSEM(3, surface_flux, VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0, 0)
coordinates_max = (1.4142135623730951, 1.4142135623730951)
refinement_patches = (
  (type="box", coordinates_min=(0.3535533905932738, 0.3535533905932738),
               coordinates_max=(1.0606601717798214, 1.0606601717798214)),
)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                refinement_patches=refinement_patches,
                n_cells_max=10_000,)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_conditions, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl=0.5)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)

analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal,
                                                               energy_magnetic, cross_helicity))

callbacks = CallbackSet(summary_callback, stepsize_callback, save_solution, analysis_callback, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
