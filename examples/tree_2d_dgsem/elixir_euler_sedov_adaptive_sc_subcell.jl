using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on example 35.1.4 from Flash
- https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    # Setup based on example 35.1.4 in https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
    r0 = 0.21875f0 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    E = 1
    p0_inner = 3 * (equations.gamma - 1) * E / (3 * convert(RealT, pi) * r0^2)
    p0_outer = convert(RealT, 1.0e-5) # = true Sedov setup

    # Calculate primitive variables
    rho = 1
    v1 = 0
    v2 = 0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave

surface_flux = flux_hll
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3)

# Use weak form as default volume integral
volume_integral_default = VolumeIntegralWeakForm()

# Setup stabilized volume integral
limiter_idp = SubcellLimiterIDP(equations, basis;
                                local_twosided_variables_cons = ["rho"],
                                local_onesided_variables_nonlinear = [(entropy_guermond_etal,
                                                                       min)],
                                positivity_variables_nonlinear = [pressure],
                                # Default parameters are not sufficient to fulfill bounds properly.
                                max_iterations_newton = 60,
                                newton_tolerances = (1.0e-13, 1.0e-15))
volume_integral_stabilized = VolumeIntegralSubcellLimiting(limiter_idp;
                                                           volume_flux_dg = volume_flux,
                                                           volume_flux_fv = surface_flux)

indicator = IndicatorHennemannGassner(equations, basis,
                                      alpha_max = 0.1, # irrelevant, only `alpha_min` is used for limiting activation
                                      alpha_min = 0.01, # governs when subcell limiting is considered
                                      alpha_smooth = true,
                                      variable = density_pressure)

# Adaptive volume integral selects based on the heuristic (!) a priori `indicator` 
# if the stabilized volume integral should be employed or if the default one is deemed sufficient.
volume_integral = VolumeIntegralAdaptive(indicator = indicator,
                                         volume_integral_default = volume_integral_default,
                                         volume_integral_stabilized = volume_integral_stabilized)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 100_000, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

stepsize_callback = StepsizeCallback(cfl = 0.4)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(),
                   BoundsCheckCallback(save_errors = false, interval = 100))

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
