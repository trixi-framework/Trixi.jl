using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the ideal multi-ion GLM-MHD equations

# Single ion species: the equations reduce to single-fluid GLM-MHD, and no
# source terms (source_terms_lorentz) are required.
equations = IdealGlmMhdMultiIonEquations2D(gammas = (5 / 3,),
                                           charge_to_mass = (1.0,))

initial_condition = initial_condition_convergence_test

# Entropy-stable numerical fluxes
volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive),
                flux_nonconservative_ruedaramirez_etal)

solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# The Alfvén wave travels diagonally; the domain [0, √2] × [0, √2] is
# exactly one wavelength in both the x- and y-directions for α = π/4.
coordinates_min = (0.0, 0.0)
coordinates_max = (sqrt(2), sqrt(2))
cells_per_dimension = (4, 4)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                      periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 1.0
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
