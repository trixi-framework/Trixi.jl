using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations2D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::IdealGlmMhdEquations2D)

An MHD blast wave taken from
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
"""
function initial_condition_blast_wave(x, t, equations::IdealGlmMhdEquations2D)
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [-0.5, 0.5] x [-0.5, 0.5], γ = 1.4
    RealT = eltype(x)
    r = sqrt(x[1]^2 + x[2]^2)
    f = (convert(RealT, 0.1) - r) / convert(RealT, 0.01)
    if r <= RealT(0.09)
        p = convert(RealT, 1000)
    elseif r >= RealT(0.1)
        p = convert(RealT, 0.1)
    else
        p = convert(RealT, 0.1) + convert(RealT, 999.9) * f
    end
    rho = 1
    v1 = 0
    v2 = 0
    v3 = 0
    B1 = 100 / sqrt(4 * convert(RealT, pi))
    B2 = 0
    B3 = 0
    psi = 0
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_blast_wave

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_powell)
volume_flux = (flux_central, flux_nonconservative_powell)
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-0.5, -0.5)
coordinates_max = (0.5, 0.5)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.01)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 0.5,
                                          alpha_min = 0.001,
                                          alpha_smooth = false,
                                          variable = density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 4,
                                      max_level = 6, max_threshold = 0.01)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 7,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

cfl = 0.8
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
