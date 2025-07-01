using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

"""
    initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)

The classical Orszag-Tang vortex test case. Here, the setup is taken from
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
"""
function initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [0, 1] x [0, 1], γ = 5/3
    rho = 1
    v1 = -sinpi(2 * x[2])
    v2 = sinpi(2 * x[1])
    v3 = 0
    p = 1 / equations.gamma
    B1 = -sinpi(2 * x[2]) / equations.gamma
    B2 = sinpi(4 * x[1]) / equations.gamma
    B3 = 0
    psi = 0
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_orszag_tang

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell_local_symmetric)
volume_flux = (flux_central, flux_nonconservative_powell_local_symmetric)

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                positivity_correction_factor = 0.9,
                                max_iterations_newton = 20)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# Get the curved quad mesh from a mapping function
# Mapping as described in https://arxiv.org/abs/2012.12040
function mapping(xi, eta)
    y = 0.5 + 0.5 * eta + 1.0 / 15.0 * (cos(1.5 * pi * xi) * cos(0.5 * pi * eta))

    x = 0.5 + 0.5 * xi + 1.0 / 15.0 * (cos(0.5 * pi * xi) * cos(2 * pi * y))

    return SVector(x, y)
end

cells_per_dimension = (32, 32)

mesh = StructuredMesh(cells_per_dimension, mapping, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

cfl = 0.4
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()..., callback = callbacks);
