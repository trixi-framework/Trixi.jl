using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t,
                                                        equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    RealT = eltype(x)
    slope = 15
    B = tanh(slope * x[2] + 7.5f0) - tanh(slope * x[2] - 7.5f0)
    rho = 0.5f0 + 0.75f0 * B
    v1 = 0.5f0 * (B - 1)
    v2 = convert(RealT, 0.1) * sinpi(2 * x[1])
    p = 1
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_integral_weakform = VolumeIntegralWeakForm()
volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(volume_flux)

# `threshold` governs the tolerated entropy increase due to the weak-form
# volume integral before switching to the stabilized version
indicator = IndicatorEntropyIncrease(threshold = 1e-1)
# Adaptive volume integral using the entropy increase indicator to perform the 
# stabilized/EC volume integral when needed
volume_integral = VolumeIntegralAdaptive(indicator;
                                         volume_integral_default = volume_integral_weakform,
                                         volume_integral_stabilized = volume_integral_fluxdiff)

#volume_integral = volume_integral_weakform # Crashes
#volume_integral = volume_integral_fluxdiff # Runs, but is more expensive

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 200)

stepsize_callback = StepsizeCallback(cfl = 1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
