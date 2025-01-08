
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
equations = IdealGlmMhdMulticomponentEquations2D(gammas = (1.4, 1.4),
                                                 gas_constants = (1.0, 1.0))

"""
    initial_condition_rotor(x, t, equations::IdealGlmMhdMulticomponentEquations2D)

The classical MHD rotor test case adapted to two density components.
"""
function initial_condition_rotor(x, t, equations::IdealGlmMhdMulticomponentEquations2D)
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [0, 1] x [0, 1], γ = 1.4
    RealT = eltype(x)
    dx = x[1] - 0.5f0
    dy = x[2] - 0.5f0
    r = sqrt(dx^2 + dy^2)
    f = (convert(RealT, 0.115) - r) / convert(RealT, 0.015)
    if r <= RealT(0.1)
        rho1 = convert(RealT, 10)
        rho2 = convert(RealT, 5)
        v1 = -20 * dy
        v2 = 20 * dx
    elseif r >= RealT(0.115)
        rho1 = one(RealT)
        rho2 = convert(RealT, 0.5)
        v1 = zero(RealT)
        v2 = zero(RealT)
    else
        rho1 = 1 + 9 * f
        rho2 = 0.5f0 + 4.5f0 * f
        v1 = -20 * f * dy
        v2 = 20 * f * dx
    end
    v3 = 0
    p = 1
    B1 = 5 / sqrt(4 * convert(RealT, pi))
    B2 = 0
    B3 = 0
    psi = 0
    return prim2cons(SVector(v1, v2, v3, p, B1, B2, B3, psi, rho1, rho2), equations)
end
initial_condition = initial_condition_rotor

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.8,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.15)
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
                           interval = 6,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

cfl = 1.0
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

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
