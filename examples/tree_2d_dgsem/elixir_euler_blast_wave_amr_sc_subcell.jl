using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A medium blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables
    rho = r > 0.5f0 ? one(RealT) : RealT(1.1691)
    v1 = r > 0.5f0 ? zero(RealT) : RealT(0.1882) * cos_phi
    v2 = r > 0.5f0 ? zero(RealT) : RealT(0.1882) * sin_phi
    p = r > 0.5f0 ? RealT(1.0E-3) : RealT(1.245)
    # p = r > 0.5f0 ? RealT(1.0E-1) : RealT(1.245)

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_correction_factor = 0.5,
                                positivity_variables_nonlinear = [pressure],
                                local_twosided_variables_cons = ["rho"],
                                local_onesided_variables_nonlinear = [(Trixi.entropy_math,
                                                                       max)],
                                # Default parameters are not sufficient to fulfill bounds properly.
                                max_iterations_newton = 70,
                                newton_tolerances = (1.0e-13, 1.0e-14))
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
mortar = MortarIDP(equations, basis, alternative = false, local_factor = true,
                   basis_function = :piecewise_constant,
                   positivity_variables_cons = ["rho"],
                   positivity_variables_nonlinear = [pressure],
                   pure_low_order = true)
solver = DGSEM(basis, surface_flux, volume_integral, mortar)

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
refinement_patches = ((type = "box", coordinates_min = (0.0, -1.0),
                       coordinates_max = (1.0, 1.0)),)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000,
                refinement_patches = refinement_patches,
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

# amr_indicator = IndicatorMax(semi, variable = first)

# amr_controller = ControllerThreeLevel(semi, amr_indicator,
#                                       base_level = 4,
#                                       med_level = 5, med_threshold = 1.01,
#                                       max_level = 6, max_threshold = 1.1)

# amr_callback = AMRCallback(semi, amr_controller,
#                            interval = 10,
#                            adapt_initial_condition = true,
#                            adapt_initial_condition_only_refine = false)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        # amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback(save_errors = false))

sol = Trixi.solve(ode,
                  # Trixi.SimpleEuler(stage_callbacks = stage_callbacks);
                  Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()...,
                  callback = callbacks);
