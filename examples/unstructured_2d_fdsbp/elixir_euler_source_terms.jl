
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

###############################################################################
# Get the FDSBP approximation operator

D_SBP = derivative_operator(SummationByPartsOperators.MattssonNordström2004(),
                            derivative_order = 1, accuracy_order = 4,
                            xmin = -1.0, xmax = 1.0, N = 10)
solver = FDSBP(D_SBP,
               surface_integral = SurfaceIntegralStrongForm(flux_lax_friedrichs),
               volume_integral = VolumeIntegralStrongForm())

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/12ce661d7c354c3d94c74b964b0f1c96/raw/8275b9a60c6e7ebbdea5fc4b4f091c47af3d5273/mesh_periodic_square_with_twist.mesh",
                           joinpath(@__DIR__, "mesh_periodic_square_with_twist.mesh"))

mesh = UnstructuredMesh2D(mesh_file, periodicity = true)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true)

callbacks = CallbackSet(summary_callback, analysis_callback,
                        alive_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(), abstol = 1.0e-9, reltol = 1.0e-9,
            save_everystep = false, callback = callbacks)
summary_callback() # print the timer summary
