
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
gamma = 2
equations = IdealGlmMhdEquations1D(gamma)

"""
    initial_condition_briowu_shock_tube(x, t, equations::IdealGlmMhdEquations1D)

Compound shock tube test case for one dimensional ideal MHD equations. It is basically an
MHD extension of the Sod shock tube. Taken from Section V of the article
- Brio and Wu (1988)
  An Upwind Differencing Scheme for the Equations of Ideal Magnetohydrodynamics
  [DOI: 10.1016/0021-9991(88)90120-9](https://doi.org/10.1016/0021-9991(88)90120-9)
"""
function initial_condition_briowu_shock_tube(x, t, equations::IdealGlmMhdEquations1D)
    # domain must be set to [0, 1], γ = 2, final time = 0.12
    rho = x[1] < 0.5 ? 1.0 : 0.125
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    p = x[1] < 0.5 ? 1.0 : 0.1
    B1 = 0.75
    B2 = x[1] < 0.5 ? 1.0 : -1.0
    B3 = 0.0
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3), equations)
end
initial_condition = initial_condition_briowu_shock_tube

boundary_conditions = BoundaryConditionDirichlet(initial_condition)

surface_flux = flux_hlle
volume_flux = flux_derigs_etal
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

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.12)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

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

stepsize_callback = StepsizeCallback(cfl = 0.65)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
