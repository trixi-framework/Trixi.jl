using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_weak_blast_wave

basis = LobattoLegendreBasis(5)
surface_flux = flux_lax_friedrichs
# Note: Plain weak form volume integral is still stable for this problem

volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(flux_ranocha)

# Standard Flux-Differencing volume integral, roughly twice as expensive as the adaptive one
#solver = DGSEM(basis, surface_flux, volume_integral_fluxdiff)

indicator = IndicatorEntropyViolation(basis; threshold = 1e-6)
volume_integral = VolumeIntegralAdaptive(indicator;
                                         volume_integral_default = VolumeIntegralWeakForm(),
                                         volume_integral_stabilized = volume_integral_fluxdiff)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0,)
coordinates_max = (2.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 8,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0,
            ode_default_options()..., callback = callbacks);
