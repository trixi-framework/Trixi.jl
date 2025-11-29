using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72
mu = 6.25e-4 # equivalent to Re = 1600

equations = CompressibleEulerEquations3D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

"""
    initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)

The classical viscous Taylor-Green vortex, as found for instance in

- Jonathan R. Bull and Antony Jameson
  Simulation of the Compressible Taylor Green Vortex using High-Order Flux Reconstruction Schemes
  [DOI: 10.2514/6.2014-3210](https://doi.org/10.2514/6.2014-3210)
"""
function initial_condition_taylor_green_vortex(x, t,
                                               equations::CompressibleEulerEquations3D)
    A = 1.0 # magnitude of speed
    Ms = 0.1 # maximum Mach number

    rho = 1.0
    v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
    v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
    v3 = 0.0
    p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
    p = p +
        1.0 / 16.0 * A^2 * rho *
        (cos(2 * x[1]) * cos(2 * x[3]) + 2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) +
         cos(2 * x[2]) * cos(2 * x[3]))

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_taylor_green_vortex

surface_flux = flux_hlle
volume_flux = flux_chandrashekar
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_integral_weakform = VolumeIntegralWeakForm()
volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(volume_flux)

# `threshold` governs the tolerated entropy increase due to the weak-form
# volume integral before switching to the stabilized version
indicator = IndicatorEntropyIncrease(threshold = 1e-3)
# Adaptive volume integral using the entropy increase indicator to perform the 
# stabilized/EC volume integral when needed
volume_integral = VolumeIntegralAdaptive(indicator;
                                         volume_integral_default = volume_integral_weakform,
                                         volume_integral_stabilized = volume_integral_fluxdiff)

#volume_integral = volume_integral_weakform # Crashes
#volume_integral = volume_integral_fluxdiff # Runs, but is more expensive

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = (1.0, 1.0, 1.0) .* pi
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (energy_kinetic,
                                                           enstrophy))

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); dt = 2e-2, adaptive = false,
            ode_default_options()..., callback = callbacks)
