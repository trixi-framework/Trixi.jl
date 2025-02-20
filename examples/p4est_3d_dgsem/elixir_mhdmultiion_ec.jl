
using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Compute non-trivial electron pressure as 0.2 * total_pressure
function electron_pressure(u, equations::IdealGlmMhdMultiIonEquations3D)
    prim = cons2prim(u, equations)
    pe = zero(eltype(u))
    for k in eachcomponent(equations)
        rho, v1, v2, v3, p = get_component(k, prim, equations)
        pe += 0.2 * p
    end
    return zero(u[1])
end

# semidiscretization of the ideal GLM-MHD multi-ion equations
equations = IdealGlmMhdMultiIonEquations3D(gammas = (1.4, 1.667),
                                           charge_to_mass = (1.0, 2.0))

initial_condition = initial_condition_weak_blast_wave

volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
# For provably entropy-stable surface fluxes, use
# surface_flux = (FluxPlusDissipation(flux_ruedaramirez_etal, DissipationLaxFriedrichsEntropyVariables()),
#                 flux_nonconservative_ruedaramirez_etal)
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# Mapping as described in https://arxiv.org/abs/2012.12040
function mapping(xi_, eta_, zeta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5
    zeta = 1.5 * zeta_ + 1.5

    y = eta +
        3 / 8 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
         cos(0.5 * pi * (2 * eta - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    x = xi +
        3 / 8 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
         cos(2 * pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    z = zeta +
        3 / 8 * (cos(0.5 * pi * (2 * x - 3) / 3) *
         cos(pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    return SVector(x, y, z)
end

trees_per_dimension = (8, 8, 8)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3, initial_refinement_level = 0, mapping = mapping,
                 periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_lorentz)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 0.5

stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
