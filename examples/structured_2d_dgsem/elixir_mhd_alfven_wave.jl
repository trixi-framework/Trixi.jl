using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

initial_condition = initial_condition_convergence_test

# Get the DG approximation space

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_powell)
volume_flux = (flux_central, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# Get the curved quad mesh from a mapping function
# Mapping as described in https://arxiv.org/abs/1809.01178
function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0, sqrt(2)]
    # Note, we use the domain [0, sqrt(2)]^2 for the Alfvén wave convergence test case
    xi = 0.5 * sqrt(2) * xi_ + 0.5 * sqrt(2)
    eta = 0.5 * sqrt(2) * eta_ + 0.5 * sqrt(2)

    y = eta +
        sqrt(2) / 12 * (cos(1.5 * pi * (2 * xi - sqrt(2)) / sqrt(2)) *
         cos(0.5 * pi * (2 * eta - sqrt(2)) / sqrt(2)))

    x = xi +
        sqrt(2) / 12 * (cos(0.5 * pi * (2 * xi - sqrt(2)) / sqrt(2)) *
         cos(2 * pi * (2 * y - sqrt(2)) / sqrt(2)))

    return SVector(x, y)
end

cells_per_dimension = (4, 4)
mesh = StructuredMesh(cells_per_dimension, mapping)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = false,
                                     extra_analysis_integrals = (entropy, energy_total,
                                                                 energy_kinetic,
                                                                 energy_internal,
                                                                 energy_magnetic,
                                                                 cross_helicity))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)
cfl = 2.0
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

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
