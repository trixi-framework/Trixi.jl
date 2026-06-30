using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

initial_condition = initial_condition_convergence_test

# Get the DG approximation space
volume_flux = (flux_central, flux_nonconservative_powell_local_jump)
surface_flux = (flux_hlle,
                flux_nonconservative_powell_local_jump)

polydeg = 4
basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;)

volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# Create a periodic P4estMesh by a mapping from [-1,1]^2 to [0, sqrt(2)]^2
function mapping_twist(xi, eta)
    domain_length = sqrt(2.0)
    amplitude = 0.1

    y_affine = 0.5 * domain_length * (eta + 1.0)
    x_affine = 0.5 * domain_length * (xi + 1.0)

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

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation
stage_callback = (SubcellLimiterIDPCorrection(),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callback);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()..., callback = callbacks, adaptive = false);
