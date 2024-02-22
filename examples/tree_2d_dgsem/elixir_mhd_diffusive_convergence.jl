using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the visco-resistive compressible MHD equations

prandtl_number() = 0.72
mu_const = 1e-2
eta_const = 1e-2
prandtl_const = prandtl_number()

equations = IdealGlmMhdEquations2D(5 / 3)
equations_parabolic = ViscoResistiveMhd2D(equations, mu = mu_const,
                                          Prandtl = prandtl_number(),
                                          eta = eta_const,
                                          gradient_variables = GradientVariablesPrimitive())

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_hindenlang_gassner, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y), max(z))

# Create a uniformly refined mesh
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 1,
                n_cells_max = 150_000) # set maximum capacity of tree data structure

function initial_condition_constant_alfven_3d(x, t, equations)
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

    # 3d Alfven wave
    rho = 1.0 + e * cos(phi_alv + 1.0)
    v1 = -e * ny * cos(phi_alv) / rho
    v2 = e * nx * cos(phi_alv) / rho
    v3 = e * sin(phi_alv) / rho
    p = 1.0# + e*cos(phi_alv + 1.0)
    B1 = nx - rho * v1 * sqr
    B2 = ny - rho * v2 * sqr
    B3 = -rho * v3 * sqr
    psi = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

@inline function source_terms_mhd_convergence_test_3d(u, x, t, equations)
    # 3d Alfven wave, rho gets perturbed
    r_1 = 0.02*sqrt(5)*pi*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)

    r_2 = sqrt(5)*pi^2*mu_const*(-0.04*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + (0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*(0.0012*cos(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) - 0.0004*cos(1)) + 3.2e-5*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^3

    r_3 = -sqrt(5)*pi^2*mu_const*(-0.02*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + (0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*(0.0006*cos(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) - 0.0002*cos(1)) + 1.6e-5*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^3

    r_4 = -pi^2*mu_const*((0.003*sin(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) + 0.001*sin(1))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 0.1*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])) - 8.0e-5*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^3

    r_5 = 0.2*pi*(2.77777777777778e-7*pi*mu_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1) -2.58888888888889e-5*pi*mu_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2 + 1.38888888888889e-7*pi*mu_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^3 - 1.78888888888889e-5*pi*mu_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2 + 0.000547222222222222*pi*mu_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1) + 2.77777777777778e-7*pi*mu_const*sin(-sqrt(5)*pi*t +pi*x[1] + 2*pi*x[2] + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1) - 2.58888888888889e-5*pi*mu_const*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 1.38888888888889e-7*pi*mu_const*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^3 - 1.78888888888889e-5*pi*mu_const*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2 + 0.000547222222222223*pi*mu_const*cos(pi*(sqrt(5)*t - x[1] -2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1) + 1.0e-7*sqrt(5)*(-sin(-4*sqrt(5)*pi*t + 4*pi*x[1] + 8*pi*x[2] + 2) + 2*sin(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 2) - sin(2)) + 1.0e-7*sqrt(5)*(sin(-4*sqrt(5)*pi*t + 4*pi*x[1] + 8*pi*x[2] + 2) + 2*sin(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 2) + sin(2)) - 8.0e-9*sqrt(5)*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2 - 2.0e-5*sqrt(5)*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1) - 8.0e-9*sqrt(5)*sin(-sqrt(5)*pi*t+ pi*x[1] + 2*pi*x[2] + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2 - 2.0e-5*sqrt(5)*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2)/(-0.04*sin(pi*x[2])*sin(-sqrt(5)*pi*t + pi*x[1] + pi*x[2] + 1) + 0.02*cos(-sqrt(5)*pi*t + pi*x[1] + 1) - 1.0)^4

    r_6 = pi*(-0.04*(sqrt(5)*pi*eta_const*cos(pi*(-sqrt(5)*t + x[1] + 2*x[2])) + sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 + 0.04*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])) + 0.0004*sin(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) + 0.0004*sin(1))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2

    r_7 = pi*(0.02*(sqrt(5)*pi*eta_const*cos(pi*(-sqrt(5)*t + x[1] + 2*x[2])) + sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 - 0.02*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])) - 0.0002*sin(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) - 0.0002*sin(1))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1]+ 2*x[2] - 3.0) + 1) + 1)^2

    r_8 = pi*((0.1*pi*eta_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])) + 0.02*sqrt(5)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 - 0.02*sqrt(5)*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) +1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 0.0002*sqrt(5)*(cos(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) - cos(1)))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2

    r_9 = 0.0

    return SVector(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9)
end

initial_condition = initial_condition_constant_alfven_3d
source_terms = source_terms_mhd_convergence_test_3d

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             source_terms = source_terms)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 200,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
cfl = 0.5
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
