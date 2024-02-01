using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the visco-resistive compressible MHD equations

prandtl_number() = 0.72
mu_const = 2e-2
eta_const = 2e-2
prandtl_const = prandtl_number()

equations = IdealGlmMhdEquations2D(5 / 3)
equations_parabolic = ViscoResistiveMhd2D(equations, mu = mu_const,
                                          Prandtl = prandtl_number(),
                                          eta = eta_const,
                                          gradient_variables = GradientVariablesPrimitive())

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 4,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 1,
                n_cells_max = 500_000) # set maximum capacity of tree data structure

function initial_condition_constant_alfven(x, t, equations)
    # Alfvén wave in three space dimensions modified by a periodic density variation.
    # For the system without the density variations see: Altmann thesis http://dx.doi.org/10.18419/opus-3895.
    # Domain must be set to [-1, 1]^3, γ = 5/3.
    omega = 2.0 * pi # may be multiplied by frequency
    # r = length-variable = length of computational domain
    r = 2.0
    # e = epsilon
    e = 0.02
    nx = 1 / sqrt(r^2 + 1.0)
    ny = r / sqrt(r^2 + 1.0)
    sqr = 1.0
    Va = omega / (ny * sqr)
    phi_alv = omega / ny * (nx * (x[1] - 0.5 * r) + ny * (x[2] - 0.5 * r)) - Va * t

    k = 2 * pi
    rho = 1.0
    v1 = 0
    v2 = -e * sin(k * x[1]) * sqrt(rho)
    v3 = 0
    B1 = 1
    B2 = e * sin(k * x[1])
    B3 = 0
    p = 1
    psi = 0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

@inline function source_terms_mhd_convergence_test(u, x, t, equations)
    r_1 = 0
    r_2 = -0.0004*pi*sin(4*pi*x[1])
    r_3 = pi*(0.08*pi*mu_const*sin(2*pi*x[1]) + 0.04*cos(2*pi*x[1]))
    r_4 = 0
    r_5 = pi*(-0.0016*pi*eta_const*sin(2*pi*x[1])^2 + 0.0016*pi*eta_const*cos(2*pi*x[1])^2 - 0.0127111111111111*pi*mu_const*sin(2*pi*x[1])^2 + 0.0127111111111111*pi*mu_const*cos(2*pi*x[1])^2 - 0.0008*sin(4*pi*x[1]))
    r_6 = 0
    r_7 = -pi*(0.08*pi*eta_const*sin(2*pi*x[1]) + 0.04*cos(2*pi*x[1]))
    r_8 = 0
    r_9 = 0

    return -SVector(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9)
end

initial_condition = initial_condition_constant_alfven

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             source_terms = source_terms_mhd_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.1)
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
cfl = 0.1
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.1, cfl = cfl)

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
