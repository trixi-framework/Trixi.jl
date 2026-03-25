using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

initial_condition = initial_condition_convergence_test

# Get the DG approximation space
volume_flux = (flux_central, flux_nonconservative_powell)
solver = DGSEM(polydeg = 4,
               surface_flux = (flux_hlle,
                               flux_nonconservative_powell),
volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# Curved periodic mapping on [0, sqrt(2)]^2
function mapping_twist(xi, eta)
    domain_length = sqrt(2.0)
    half_length = 0.5 * domain_length
    amplitude = 0.1 #* half_length

    y_affine = half_length * (eta + 1.0)
    x_affine = half_length * (xi + 1.0)

    y = y_affine + amplitude * sin(pi * xi) * cos(0.5 * pi * eta)
    x = x_affine + amplitude * sin(pi * eta) * cos(0.5 * pi * xi)
    return SVector(x, y)
end

trees_per_dimension = (2, 2)
mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 initial_refinement_level = 1,
                 periodicity = true,
                 mapping = mapping_twist)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 0.1
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33();
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks, adpative = false);
