using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_density_wave_high_density(x, t,
                                                     equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    rho = 2 + convert(RealT, 0.98) * sinpi(2 * (x[1] + x[2] - t * (v1 + v2)))
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 20
    rho_e_total = p / (equations.gamma - 1) + 0.5f0 * rho * (v1^2 + v2^2)
    return SVector(rho, rho_v1, rho_v2, rho_e_total)
end
initial_condition = initial_condition_density_wave_high_density
initial_condition = initial_condition_density_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

# `VolumeIntegralSubcellLimiting` is required in order to use `MortarIDP`.
# But, there is no actual limiting needed.
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                max_iterations_newton = 50)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
mortar = MortarIDP(equations, basis;
                   positivity_variables_cons = ["rho"],
                   positivity_variables_nonlinear = [pressure],
                   pure_low_order = false)
solver = DGSEM(basis, surface_flux, volume_integral, mortar)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
refinement_patches = ((type = "box", coordinates_min = (0.0, -0.5),
                       coordinates_max = (0.5, 0.5)),)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                refinement_patches = refinement_patches,
                n_cells_max = 10_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

stepsize_callback = StepsizeCallback(cfl = 0.7)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode,
                  Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
