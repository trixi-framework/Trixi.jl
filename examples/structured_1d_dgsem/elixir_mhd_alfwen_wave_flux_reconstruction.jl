using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
gamma = 5 / 3
equations = IdealGlmMhdEquations1D(gamma)

initial_condition = initial_condition_convergence_test

polydeg = 4
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_hllc
correction_function = Val(:g_DG) # Classic strong form DG
surface_integral = SurfaceIntegralFluxReconstruction(basis,
                                                     surface_flux = surface_flux,
                                                     correction_function = correction_function)
solver = DGSEM(polydeg = polydeg,
               surface_integral = surface_integral,
               volume_integral = VolumeIntegralStrongForm())

coordinates_min = 0.0
coordinates_max = 1.0
cells_per_dimension = (6,)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     # Need to overwrite default integrals since 
                                     # calculation of `l2_divb` and `linf_divb` is not implemented for 
                                     # StructuredMesh{1}
                                     analysis_integrals = (Trixi.entropy_timederivative,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.7)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, ode_default_options()..., callback = callbacks);
