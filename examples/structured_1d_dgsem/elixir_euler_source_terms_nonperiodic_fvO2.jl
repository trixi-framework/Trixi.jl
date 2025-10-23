
using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# you can either use a single function to impose the BCs weakly in all
# 2*ndims == 2 directions or you can pass a tuple containing BCs for
# each direction
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (x_neg = boundary_condition,
                       x_pos = boundary_condition)

polydeg = 8 # Governs in this case only the number of subcells
basis = LobattoLegendreBasis(polydeg)
volume_integral = VolumeIntegralPureLGLFiniteVolumeO2(basis,
                                                      volume_flux_fv = flux_hll,
                                                      reconstruction_mode = reconstruction_O2_inner,
                                                      slope_limiter = vanLeer)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = (0.0,)
coordinates_max = (2.0,)
cells_per_dimension = (8,)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                      periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, ParsaniKetchesonDeconinck3S82(),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
