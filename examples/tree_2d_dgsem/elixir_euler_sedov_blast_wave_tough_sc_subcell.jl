using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_sedov_blast_wave_adapted(x, t,
                                                    equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    bell_sigma = convert(RealT, 0.5 * 5e-1)
    blast_sigma = convert(RealT, 0.5 * 3e-1)
    bell_mass = convert(RealT, 0.5)
    blast_energy = 1

    rho = 1
    pressure = convert(RealT, 1e-5)

    dim = 2
    bell_mass_normalized = bell_mass / sqrt((2 * pi)^dim) / bell_sigma^dim
    blast_energy_normalized = blast_energy / sqrt((2 * pi)^dim) / blast_sigma^dim

    # Calculate primitive variables
    rho = rho + bell_mass_normalized * exp(-0.5 * (r / bell_sigma)^2)
    v1 = 0
    v2 = 0
    p = pressure

    cons = prim2cons(SVector(rho, v1, v2, p), equations)

    return cons +
           SVector(0, 0, 0, blast_energy_normalized * exp(-0.5 * (r / blast_sigma)^2))
end
initial_condition = initial_condition_sedov_blast_wave_adapted

surface_flux = flux_lax_friedrichs
volume_flux = flux_chandrashekar
basis = LobattoLegendreBasis(6)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                positivity_correction_factor = 0.5,
                                # local_twosided_variables_cons = ["rho"],
                                # local_onesided_variables_nonlinear = [(Trixi.entropy_math,
                                #                                        max)],
                                # Default parameters are not sufficient to fulfill bounds properly.
                                max_iterations_newton = 100,
                                newton_tolerances = (1.0e-13, 1.0e-14),
                                bar_states = false)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.5, -1.5)
coordinates_max = (1.5, 1.5)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 100_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(#interval = 50,
                                     dt = 0.1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

# positivity_limiter = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-10, 1.0e-10),
#                                                          variables = (Trixi.density,
#                                                                       pressure))

# amr_indicator = IndicatorHennemannGassner(semi,
#                                           alpha_max = 0.5,
#                                           alpha_min = 0.001,
#                                           alpha_smooth = true,
#                                           variable = density_pressure)
# amr_controller = ControllerThreeLevel(semi, amr_indicator,
#                                       base_level = 4,
#                                       max_level = 6, max_threshold = 0.01)
# amr_callback = AMRCallback(semi, amr_controller,
#                            interval = 5,
#                            adapt_initial_condition = true,
#                            adapt_initial_condition_only_refine = true,
#                            limiter! = positivity_limiter)

stepsize_callback = StepsizeCallback(cfl = 0.4)

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
