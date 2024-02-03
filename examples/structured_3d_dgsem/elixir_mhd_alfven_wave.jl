
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations3D(5 / 3)

initial_condition = initial_condition_convergence_test

volume_flux = (flux_central, flux_nonconservative_powell)
solver = DGSEM(polydeg = 5,
               surface_flux = (flux_hlle,
                               flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# Create the mesh
# Note, we use the domain [-1, 1]^3 for the Alfvén wave convergence test case so the
# warped mapping simplifies (quite a bit)

# Mapping as described in https://arxiv.org/abs/2012.12040
function mapping(xi, eta, zeta)
    y = eta + 0.125 * (cos(1.5 * pi * xi) * cos(0.5 * pi * eta) * cos(0.5 * pi * zeta))

    x = xi + 0.125 * (cos(0.5 * pi * xi) * cos(2 * pi * y) * cos(0.5 * pi * zeta))

    z = zeta + 0.125 * (cos(0.5 * pi * x) * cos(pi * y) * cos(0.5 * pi * zeta))

    return SVector(x, y, z)
end

cells_per_dimension = (4, 4, 4)
mesh = StructuredMesh(cells_per_dimension, mapping)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 1.2
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
