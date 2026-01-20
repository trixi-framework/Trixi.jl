using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations2D(1.4)

# We repeat test case for stability of EC fluxes from paper
#- Gregor J. Gassner, Magnus Svärd, Florian J. Hindenlang (2020)
#  Stability issues of entropy-stable and/or split-form high-order schemes
#  [DOI: 10.1007/s10915-021-01720-8](https://doi.org/10.1007/s10915-021-01720-8)
initial_condition = initial_condition_density_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_chandrashekar

polydeg = 5
basis = LobattoLegendreBasis(polydeg)

volume_integral_weakform = VolumeIntegralWeakForm()
volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(volume_flux)

# This indicator compares the entropy production of the weak form to the 
# entropy-conserving flux-differencing volume integral.
# If the entropy production of the weak form is lower than that of the
# flux-differencing form, we use the flux-differencing form to stabilize the solution.
indicator = IndicatorEntropyDiffusion(equations, basis)

# Adaptive volume integral using the entropy production comparison indicator to perform the 
# stabilized/EC volume integral when needed and keeping the weak form if it is more diffusive.
volume_integral = VolumeIntegralAdaptive(volume_integral_default = volume_integral_weakform,
                                         volume_integral_stabilized = volume_integral_fluxdiff,
                                         indicator = indicator)

#volume_integral = volume_integral_weakform # Stable, but unphysical entropy increase!
#volume_integral = volume_integral_fluxdiff # Crashes!

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2, # 4 x 4 elements
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.9) # In paper, CFL = 0.05 is used

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, ode_default_options()..., callback = callbacks);
