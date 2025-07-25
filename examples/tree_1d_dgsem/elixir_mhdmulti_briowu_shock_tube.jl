using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
equations = IdealGlmMhdMulticomponentEquations1D(gammas = (2.0, 2.0),
                                                 gas_constants = (2.0, 2.0))

"""
    initial_condition_briowu_shock_tube(x, t, equations::IdealGlmMhdMulticomponentEquations1D)

Compound shock tube test case for one dimensional ideal MHD equations. It is basically an
MHD extension of the Sod shock tube. Taken from Section V of the article
- Brio and Wu (1988)
  An Upwind Differencing Scheme for the Equations of Ideal Magnetohydrodynamics
  [DOI: 10.1016/0021-9991(88)90120-9](https://doi.org/10.1016/0021-9991(88)90120-9)
"""
function initial_condition_briowu_shock_tube(x, t,
                                             equations::IdealGlmMhdMulticomponentEquations1D)
    # domain must be set to [0, 1], γ = 2, final time = 0.12
    RealT = eltype(x)
    if x[1] < 0.5f0
        rho = one(RealT)
        prim_rho = SVector{ncomponents(equations), real(equations)}(2^(i - 1) * (1 - 2) *
                                                                    rho / (1 -
                                                                     2^ncomponents(equations))
                                                                    for i in eachcomponent(equations))
    else
        rho = RealT(0.125)
        prim_rho = SVector{ncomponents(equations), real(equations)}(2^(i - 1) * (1 - 2) *
                                                                    rho / (1 -
                                                                     2^ncomponents(equations))
                                                                    for i in eachcomponent(equations))
    end
    v1 = 0
    v2 = 0
    v3 = 0
    p = x[1] < 0.5f0 ? one(RealT) : RealT(0.1)
    B1 = 0.75f0
    B2 = x[1] < 0.5f0 ? 1 : -1
    B3 = 0

    prim_other = SVector(v1, v2, v3, p, B1, B2, B3)
    return prim2cons(vcat(prim_other, prim_rho), equations)
end
initial_condition = initial_condition_briowu_shock_tube

boundary_conditions = BoundaryConditionDirichlet(initial_condition)

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
volume_flux = flux_hindenlang_gassner
basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.8,
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

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
