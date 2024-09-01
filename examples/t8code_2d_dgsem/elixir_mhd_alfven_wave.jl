using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the compressible ideal GLM-MHD equations.

gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

initial_condition = initial_condition_convergence_test

# Get the DG approximation space
volume_flux = (flux_central, flux_nonconservative_powell)

solver = DGSEM(polydeg = 4, surface_flux = (flux_hlle, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (sqrt(2.0), sqrt(2.0))

trees_per_dimension = (8, 8)

mesh = T8codeMesh(trees_per_dimension, polydeg = 3,
                  coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                  initial_refinement_level = 0, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 0.9
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary

# Finalize `T8codeMesh` to make sure MPI related objects in t8code are
# released before `MPI` finalizes.
!isinteractive() && finalize(mesh)
