
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.001 # almost isothermal when gamma reaches 1
equations = CompressibleEulerEquations2D(gamma)

# This is a hand made colliding flow setup without reference. Features Mach=70 inflow from both
# sides, with relative low temperature, such that pressure keeps relatively small
# Computed with gamma close to 1, to simulate isothermal gas
function initial_condition_colliding_flow_astro(x, t,
                                                equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # resolution 128^2 elements (refined close to the interface) and polydeg=3 (total of 512^2 DOF)
    # domain size is [-64,+64]^2
    RealT = eltype(x)
    @unpack gamma = equations
    # the quantities are chosen such, that they are as close as possible to the astro examples
    # keep in mind, that in the astro example, the physical units are weird (parsec, mega years, ...)
    rho = convert(RealT, 0.0247)
    c = convert(RealT, 0.2)
    p = c^2 / gamma * rho
    vel =  convert(RealT, 13.907432274789372)
    slope = 1
    v1 = -vel * tanh(slope * x[1])
    # add small initial disturbance to the field, but only close to the interface
    if abs(x[1]) < 10
        v1 = v1 * (1 + RealT(0.01) * sinpi(x[2]))
    end
    v2 = 0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_colliding_flow_astro

boundary_conditions = (x_neg = BoundaryConditionDirichlet(initial_condition_colliding_flow_astro),
                       x_pos = BoundaryConditionDirichlet(initial_condition_colliding_flow_astro),
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

surface_flux = flux_lax_friedrichs # HLLC needs more shock capturing (alpha_max)
volume_flux = flux_ranocha # works with Chandrashekar flux as well
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

# shock capturing necessary for this tough example, however alpha_max = 0.5 is fine
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.0001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-64.0, -64.0)
coordinates_max = (64.0, 64.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = (false, true),
                n_cells_max = 100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 25.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

# Simulation also feasible without AMR: AMR reduces CPU time by a factor of about 2
amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 1.0,
                                          alpha_min = 0.0001,
                                          alpha_smooth = false,
                                          variable = Trixi.density)

amr_controller = ControllerThreeLevelCombined(semi, amr_indicator, indicator_sc,
                                              base_level = 2,
                                              med_level = 0, med_threshold = 0.0003, # med_level = current level
                                              max_level = 8, max_threshold = 0.003,
                                              max_threshold_secondary = indicator_sc.alpha_max)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback, save_solution)

# positivity limiter necessary for this tough example
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))

###############################################################################
# run the simulation
# use adaptive time stepping based on error estimates, time step roughly dt = 5e-3
sol = solve(ode, SSPRK43(stage_limiter!);
            ode_default_options()..., callback = callbacks);
summary_callback() # print the timer summary
