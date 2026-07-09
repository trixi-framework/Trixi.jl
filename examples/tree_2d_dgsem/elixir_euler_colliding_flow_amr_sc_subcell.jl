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
    vel = convert(RealT, 13.907432274789372)
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

boundary_conditions = (;
                       x_neg = BoundaryConditionDirichlet(initial_condition_colliding_flow_astro),
                       x_pos = BoundaryConditionDirichlet(initial_condition_colliding_flow_astro),
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

surface_flux = FluxLaxFriedrichs(max_abs_speed_naive) # flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                local_twosided_variables_cons = ["rho"],
                                max_iterations_newton = 50)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)

mortar = MortarIDP(equations, basis, limiter_idp)
solver = DGSEM(basis, surface_flux, volume_integral, mortar)

coordinates_min = (-64.0, -64.0)
coordinates_max = (64.0, 64.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = (false, true),
                n_cells_max = 100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 25.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.25,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 2,
                                      med_level = 0, med_threshold = 0.05, # med_level = current level
                                      max_level = 8, max_threshold = 0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 0.2)

limiting_analysis = LimitingAnalysisCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        limiting_analysis,
                        amr_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
