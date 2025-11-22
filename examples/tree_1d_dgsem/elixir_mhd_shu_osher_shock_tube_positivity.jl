using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
gamma = 5 / 3
equations = IdealGlmMhdEquations1D(gamma)

"""
    initial_condition_shu_osher_shock_tube(x, t, equations::IdealGlmMhdEquations1D)

Extended version of the test of Shu and Osher for one dimensional ideal MHD equations.
Taken from Section 4.1 of
- Derigs et al. (2016)
  A Novel High-Order, Entropy Stable, 3D AMR MHD Solver withGuaranteed Positive Pressure
  [DOI: 10.1016/j.jcp.2016.04.048](https://doi.org/10.1016/j.jcp.2016.04.048)

with further decreased pressure in the right state to make the test more challenging/
requiring positivity preservation limiting.
"""
function initial_condition_shu_osher_shock_tube(x, t, equations::IdealGlmMhdEquations1D)
    # domain must be set to [-5, 5], γ = 5/3, final time = 0.7
    # initial shock location is taken to be at x = -4
    RealT = eltype(x)
    x_0 = -4
    rho = x[1] <= x_0 ? RealT(3.5) : 1 + RealT(0.2) * sin(5 * x[1])
    v1 = x[1] <= x_0 ? RealT(5.8846) : zero(RealT)
    v2 = x[1] <= x_0 ? RealT(1.1198) : zero(RealT)
    v3 = 0
    p = x[1] <= x_0 ? RealT(42.0267) : RealT(0.1) # Right state is 1 in reference
    B1 = 1
    B2 = x[1] <= x_0 ? RealT(3.6359) : one(RealT)
    B3 = 0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3), equations)
end
initial_condition = initial_condition_shu_osher_shock_tube

boundary_conditions = BoundaryConditionDirichlet(initial_condition)

surface_flux = flux_hlle
volume_flux = flux_hindenlang_gassner
basis = LobattoLegendreBasis(4)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = -5.0
coordinates_max = 5.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.7)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (energy_kinetic,
                                                                 energy_internal,
                                                                 energy_magnetic,
                                                                 cross_helicity))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_controller = ControllerThreeLevel(semi, indicator_sc,
                                      base_level = 4,
                                      max_level = 7, max_threshold = 0.01)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 10,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 0.4)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

# Positivity-preserving limiter setup
# - `alpha_max` is increased above the value used in the volume integral 
#               to allow room for positivity limiting.
# - `root_tol` can be set to this relatively high value while still ensuring positivity
# - `use_density_init` is set to false since in the modification of the initial condition
#                      only the pressure is decreased, i.e., density should be non-critical
#                      and is probably not corrected at all.
limiter! = LowerBoundPreservingLimiterRuedaRamirezGassner(semi;
                                                          alpha_max = 0.7,
                                                          root_tol = 1e-8,
                                                          use_density_init = false)
stage_callbacks = (limiter!,)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
