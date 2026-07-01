using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the visco-resistive compressible MHD equations

prandtl_number() = 0.72
mu() = 5e-3
eta() = 5e-3

equations = IdealGlmMhdEquations2D(2.0)
equations_parabolic = ViscoResistiveMhd2D(equations, mu = mu(),
                                          Prandtl = prandtl_number(),
                                          eta = eta(),
                                          gradient_variables = GradientVariablesPrimitive())

volume_flux = (flux_central, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3,
               surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive),
                               flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                periodicity = true,
                n_cells_max = 10_000) # set maximum capacity of tree data structure

# Test case proposed by Bohm (2018), Section 5.2 (https://arxiv.org/pdf/1802.07341).
function initial_condition_bohm_2d(x, t, equations)
    h = 0.5 * sin(2 * pi * (x[1] + x[2] - t)) + 2

    return SVector(h, h, h, 0, 2 * h^2 + h, h, -h, 0, 0)
end

@inline function source_terms_mhd_convergence_test_bohm_2d(u, x, t, equations)
    mu_ = mu()
    eta_ = eta()
    Pr_ = prandtl_number()

    h = 0.5 * sin(2 * pi * (x[1] + x[2] - t)) + 2
    h_x = pi * cos(2 * pi * (x[1] + x[2] - t))
    h_xx = -2 * pi^2 * sin(2 * pi * (x[1] + x[2] - t))

    r_1 = h_x
    r_2 = h_x + 4 * h * h_x
    r_3 = r_2
    r_4 = zero(h_x)
    r_5 = h_x + 12 * h * h_x - 4 * eta_ * (h_x^2 + h * h_xx) - 4 * mu_ * h_xx / Pr_
    r_6 = h_x - 2 * eta_ * h_xx
    r_7 = -h_x + 2 * eta_ * h_xx
    r_8 = 0
    r_9 = 0

    SVector(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9)
end

initial_condition = initial_condition_bohm_2d

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic),
                                             source_terms = source_terms_mhd_convergence_test_bohm_2d)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = analysis_interval)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 200,
                                     solution_variables = cons2prim)

cfl = 1.0
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

# RDPK3SpFSAL49 in non-adaptive mode gives 4th-order temporal accuracy,
# matching the polydeg=3 spatial accuracy for convergence testing.
sol = solve(ode, RDPK3SpFSAL49(); adaptive = false, dt = 1.0,
            ode_default_options()..., callback = callbacks)
