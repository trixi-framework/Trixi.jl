using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A medium blast wave (modified to lower density and higher pressure) taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> modified to lower density, higher pressure
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables
    rho = r > 0.5f0 ? RealT(0.1) : RealT(0.2691)
    v1 = r > 0.5f0 ? zero(RealT) : RealT(0.1882) * cos_phi
    v2 = r > 0.5f0 ? zero(RealT) : RealT(0.1882) * sin_phi
    p = r > 0.5f0 ? RealT(1.0E-1) : RealT(1.245)

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_blast_wave

surface_flux = flux_hllc
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_integral_weakform = VolumeIntegralWeakForm()
volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(volume_flux)

# Use Hennemann-Gassner shock indicator to identify cells which need the stabilized volume integral.
# The only relevant parameter in this case is `alpha_min`, which governs the sensitivity,
# i.e., the minimum shock indicator value at which the stabilized volume integral becomes active.
# `alpha_max` is not relevant for the volume integral, as there is no blending of fluxes performed.
# However, for the AMR callback below, `alpha_max` is relevant.
indicator = IndicatorHennemannGassner(equations, basis,
                                      alpha_max = 0.5,
                                      alpha_min = 0.001,
                                      alpha_smooth = true,
                                      variable = density_pressure)

# Adaptive volume integral using the Hennemann-Gassner shock indicator to perform the
# stabilized/EC volume integral when needed.
volume_integral = VolumeIntegralAdaptive(indicator = indicator,
                                         volume_integral_default = volume_integral_weakform,
                                         volume_integral_stabilized = volume_integral_fluxdiff)

#volume_integral = volume_integral_weakform # Crashes

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 10_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.6)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.0)

amr_controller = ControllerThreeLevel(semi, indicator,
                                      base_level = 3,
                                      max_level = 5, max_threshold = 0.01)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 10,
                           adapt_initial_condition = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
