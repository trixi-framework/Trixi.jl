using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the visco-resistive compressible MHD equations

prandtl_number() = 0.72
# mu = 1e-2
# eta = 1e-2
mu_const = 1e-2
eta_const = 1e-2

equations = IdealGlmMhdEquations3D(5 / 3)
equations_parabolic = ViscoResistiveMhd3D(equations, mu = mu_const,
                                          Prandtl = prandtl_number(),
                                          eta = eta_const,
                                          gradient_variables = GradientVariablesPrimitive())

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0, 1.0) # maximum coordinates (max(x), max(y), max(z))

# Create a uniformly refined mesh
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 100_000) # set maximum capacity of tree data structure

function initial_condition_constant_alfven(x, t, equations)
    # Alfvén wave in three space dimensions
    # Altmann thesis http://dx.doi.org/10.18419/opus-3895
    # domain must be set to [-1, 1]^3, γ = 5/3
    p = 1
    omega = 2 * pi # may be multiplied by frequency
    # r: length-variable = length of computational domain
    r = 2
    # e: epsilon = 0.02
    e = 0.02
    nx = 1 / sqrt(r^2 + 1)
    ny = r / sqrt(r^2 + 1)
    sqr = 1
    Va = omega / (ny * sqr)
    phi_alv = omega / ny * (nx * (x[1] - 0.5 * r) + ny * (x[2] - 0.5 * r)) - Va * t

    rho = 1.0
    v1 = -e * ny * cos(phi_alv) / rho
    v2 = e * nx * cos(phi_alv) / rho
    v3 = e * sin(phi_alv) / rho
    B1 = nx - rho * v1 * sqr
    B2 = ny - rho * v2 * sqr
    B3 = -rho * v3 * sqr
    psi = 0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

@inline function source_terms_mhd_convergence_test(u, x, t, equations)
    r_1 = 0
    r_2 = pi * (0.04 * sqrt(5) * pi * mu_const * cos(pi * (-sqrt(5) * t + x[1] + 2 * x[2]))
           -
           1.35525271560688e-20 * sin(pi * (-2 * sqrt(5) * t + 2 * x[1] + 4 * x[2]))
           -
           2.60208521396521e-18 * sin(pi * (-sqrt(5) * t + x[1] + 2 * x[2])))
    r_3 = pi *
          (-0.02 * sqrt(5) * pi * mu_const * cos(pi * (-sqrt(5) * t + x[1] + 2 * x[2]))
           -
           2.71050543121376e-20 * sin(pi * (-2 * sqrt(5) * t + 2 * x[1] + 4 * x[2]))
           +
           3.46944695195361e-18 * sin(pi * (-sqrt(5) * t + x[1] + 2 * x[2])))
    r_4 = -0.1 * pi^2 * mu_const * sin(pi * (-sqrt(5) * t + x[1] + 2 * x[2]))
    r_5 = 2.71050543121376e-19 * pi^2 * mu_const *
          sin(pi * (-sqrt(5) * t + x[1] + 2 * x[2]))^2
    -2.71050543121376e-19 * pi^2 * mu_const *
    cos(pi * (sqrt(5) * t - x[1] - 2 * x[2] + 3.0))^2
    +2.71050543121376e-20 * sqrt(5) * pi *
    sin(pi * (-2 * sqrt(5) * t + 2 * x[1] + 4 * x[2]))
    r_6 = 0.04 * sqrt(5) * pi^2 * eta_const *
          cos(pi * (sqrt(5) * t - x[1] - 2 * x[2] + 3.0))
    r_7 = -0.02 * sqrt(5) * pi^2 * eta_const *
          cos(pi * (sqrt(5) * t - x[1] - 2 * x[2] + 3.0))
    r_8 = 0.1 * pi^2 * eta_const * sin(pi * (-sqrt(5) * t + x[1] + 2 * x[2]))
    r_9 = 0

    return SVector(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9)
end

initial_condition = initial_condition_constant_alfven

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             source_terms = source_terms_mhd_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.01)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
cfl = 0.25
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1e-5, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary.
summary_callback()
