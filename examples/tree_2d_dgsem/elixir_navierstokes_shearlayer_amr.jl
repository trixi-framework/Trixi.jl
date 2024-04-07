
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72
mu = 1.0 / 3.0 * 10^(-4) # equivalent to Re = 30,000

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

"""
A compressible version of the double shear layer initial condition. Adapted from
Brown and Minion (1995).

- David L. Brown and Michael L. Minion (1995)
  Performance of Under-resolved Two-Dimensional Incompressible Flow Simulations.
  [DOI: 10.1006/jcph.1995.1205](https://doi.org/10.1006/jcph.1995.1205)
"""
function initial_condition_shear_layer(x, t, equations::CompressibleEulerEquations2D)
    # Shear layer parameters
    k = 80
    delta = 0.05
    u0 = 1.0

    Ms = 0.1 # maximum Mach number

    rho = 1.0
    v1 = x[2] <= 0.5 ? u0 * tanh(k * (x[2] - 0.25)) : u0 * tanh(k * (0.75 - x[2]))
    v2 = u0 * delta * sin(2 * pi * (x[1] + 0.25))
    p = (u0 / Ms)^2 * rho / equations.gamma # scaling to get Ms

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_shear_layer

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_hllc,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# This uses velocity-based AMR
@inline function v1(u, equations::CompressibleEulerEquations2D)
    rho, rho_v1, _, _ = u
    return rho_v1 / rho
end
amr_indicator = IndicatorLöhner(semi, variable = v1)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 3,
                                      med_level = 5, med_threshold = 0.2,
                                      max_level = 7, max_threshold = 0.5)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 50,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
