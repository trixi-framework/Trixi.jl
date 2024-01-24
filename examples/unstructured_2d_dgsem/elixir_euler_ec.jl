
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(5 / 3)

initial_condition = initial_condition_weak_blast_wave

###############################################################################
# Get the DG approximation space

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 6, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the curved quad mesh from a file
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/12ce661d7c354c3d94c74b964b0f1c96/raw/8275b9a60c6e7ebbdea5fc4b4f091c47af3d5273/mesh_periodic_square_with_twist.mesh",
                           joinpath(@__DIR__, "mesh_periodic_square_with_twist.mesh"))

mesh = UnstructuredMesh2D(mesh_file, periodicity = true)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
