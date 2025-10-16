using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on Flash
- https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114000000000000000
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    # Setup based on https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114000000000000000
    r0 = 0.21875f0 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    # r0 = 0.5 # = more reasonable setup
    E = 1
    p0_inner = 3 * (equations.gamma - 1) * E / (3 * convert(RealT, pi) * r0^2)
    p0_outer = convert(RealT, 1.0e-5) # = true Sedov setup
    # p0_outer = convert(RealT, 1.0e-3) # = more reasonable setup

    # Calculate primitive variables
    rho = 1
    v1 = 0
    v2 = 0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_chandrashekar
basis = LobattoLegendreBasis(3)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_correction_factor = 0.5,
                                positivity_variables_nonlinear = [pressure],
                                local_twosided_variables_cons = ["rho"],
                                local_onesided_variables_nonlinear = [(Trixi.entropy_math,
                                                                       max)],
                                # Default parameters are not sufficient to fulfill bounds properly.
                                max_iterations_newton = 200,
                                newton_tolerances = (1.0e-13, 1.0e-14))
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
mortar = MortarIDP(equations, basis; basis_function = :piecewise_constant,
                   positivity_variables_cons = ["rho"],
                   positivity_variables_nonlinear = [pressure],
                   pure_low_order = false)
solver = DGSEM(basis, surface_flux, volume_integral, mortar)

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 100_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 0.5,
                                          alpha_min = 0.001,
                                          alpha_smooth = true,
                                          variable = density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 4,
                                      max_level = 6, max_threshold = 0.01)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 0.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        # amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(),
                   BoundsCheckCallback(save_errors = false))

sol = Trixi.solve(ode,
                  Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()...,
                  callback = callbacks);
