
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test

source_terms = source_terms_convergence_test

# you can either use a single function to impose the BCs weakly in all
# 1*ndims == 2 directions or you can pass a tuple containing BCs for
# each direction
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (x_neg = boundary_condition,
                       x_pos = boundary_condition)

polydeg = 8 # Governs in this case only the number of subcells
basis = LobattoLegendreBasis(polydeg)
surf_flux = flux_hll
solver = DGSEM(polydeg = polydeg, surface_flux = surf_flux,
               volume_integral = VolumeIntegralPureLGLFiniteVolumeO2(basis,
                                                                     volume_flux_fv = surf_flux,
                                                                     reconstruction_mode = reconstruction_small_stencil_inner,
                                                                     slope_limiter = monotonized_central))

f1() = SVector(0.0)
f2() = SVector(2.0)
cells_per_dimension = (8,)
mesh = StructuredMesh(cells_per_dimension, (f1, f2), periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, ORK256(),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
