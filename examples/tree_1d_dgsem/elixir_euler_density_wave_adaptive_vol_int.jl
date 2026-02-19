using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_density_wave

# Inviscid surface & volume flux, resulting in an entropy conservative scheme.
# Turned entropy-diffusive through `IndicatorEntropyChange` and `VolumeIntegralAdaptive` defined below.
surface_flux = flux_ranocha
volume_flux = flux_ranocha

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_integral_weakform = VolumeIntegralWeakForm()
volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(volume_flux)

# This indicator compares the entropy production of `volume_integral_default` to
# the true entropy evolution computed from the surface integral of the `entropy_potential`.
# If the difference is larger than `maximum_entropy_increase`,
# `volume_integral_stabilized` is used instead of the `volume_integral_default`.
indicator = IndicatorEntropyChange(maximum_entropy_increase = 0.0)

# Adaptive volume integral using the entropy production comparison indicator to perform the 
# stabilized/EC volume integral when needed and keeping the weak form if it is more diffusive.
volume_integral = VolumeIntegralAdaptive(volume_integral_default = volume_integral_weakform,
                                         volume_integral_stabilized = volume_integral_fluxdiff,
                                         indicator = indicator)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 30_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.True());
            dt = stepsize_callback(ode),
            ode_default_options()..., callback = callbacks);
