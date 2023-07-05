# A turbulent helically driven dynamo model in resistive compressible MHD.
# We start with a weak small-scale magnetic field provided as Gaussian noise.
# The energy is supplied via a forcing function f_force(x, t), which is delta-correlated in time.
# For a comparison look into the PC run pencil-code/simon/forced/alpha2_periodic/64_kf_5_a.

using OrdinaryDiffEq
using Trixi
using Random
using LinearAlgebra

###############################################################################
# initial condition and forcing function

"""
    initial_condition_gaussian_noise_mhd(x, t, equations)

Define the initial condition with a weak small-scale Gaussian noise magnetic field.
"""
function initial_condition_gaussian_noise_mhd(x, t, equations)
    amplitude = 1e-8

    rho = 1.0
    rho_v1 = 0.0
    rho_v2 = 0.0
    rho_v3 = 0.0
    rho_e = 10.0
    B1 = randn(1)[1] * amplitude
    B2 = randn(1)[1] * amplitude
    B3 = randn(1)[1] * amplitude
    psi = 0.0

    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
end

"""
    source_terms_helical_forcing(u, x, t, equations::CompressibleMhdDiffusion3D)

Forcing term that adds a helical small-scale driver to the system. that is
delta-correlated in time.
"""
function source_terms_helical_forcing(u, x, t, equations::IdealGlmMhdEquations3D)
    # forcing amplitude
    f_0 = 0.02
    # helicality of the forcing (-1, 1)
    sigma = 1.0

    # Extract some parameters for the computation.
    c_s = 0.1
    delta_t = 0.01

    # To make sure that the random numbers depend only on time we need to set the seeds.
    seed = reinterpret(UInt64, t)
    Random.seed!(seed)

    # Random phase -pi < phi <= pi
    phi = (rand(1)[1] * 2 - 1) * pi

    # Random vector k, also delta-correlated in time.
    k = [0.0, 0.0, 0.0]
    k[1] = (rand(1)[1] * 2 - 1) * pi
    k[2] = (rand(1)[1] * 2 - 1) * pi
    k[3] = (rand(1)[1] * 2 - 1) * pi
    k = k / norm(k)
    k = 6 * k
    k_norm = k / norm(k)

    # Random unit vector, not aligned with k.
    ee = [0.0, 0.0, 0.0]
    ee[1] = rand(1)[1] * 2 - 1
    ee[2] = rand(1)[1] * 2 - 1
    ee[3] = rand(1)[1] * 2 - 1
    # Normalize the vector e.
    ee = ee / norm(ee)

    # Compute the f-vectors.
    f_nohel = cross(k, ee) / sqrt(dot(k, k) - dot(k, ee)^2)
    M = [0 k_norm[3] -k_norm[2]; -k_norm[3] 0 k_norm[1]; k_norm[2] -k_norm[1] 0]
    R = (I - im * sigma * M) / sqrt(1 + sigma^2)
    f_k = R * f_nohel

    # Normalization factor to make sure that the time integration of f is 0.
    N = f_0 * c_s * sqrt(norm(k) * c_s / delta_t)

    forcing = real(N * f_k * exp(im * dot(k, x) + im * phi)) / u[1]

    return SVector(0.0, forcing[1], forcing[2], forcing[3], 0.0, 0.0, 0.0, 0.0, 0.0)
end


###############################################################################
# semidiscretization of the ideal compressible MHD equations

prandtl_number() = 0.72
# Corresponds to previous alpha^2 runs.
mu() = 2e-3
eta = 2e-3

gamma = 1.0 + 2.0 / 3.0

equations = IdealGlmMhdEquations3D(gamma)
equations_parabolic = CompressibleMhdDiffusion3D(equations, mu = mu(),
                                                 Prandtl = prandtl_number(), eta = eta,
                                                 gradient_variables = GradientVariablesPrimitive())

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-pi / 4, -pi / 4, 0.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (pi / 4, pi / 4, pi / 2) # maximum coordinates (max(x), max(y), max(z))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = (true, true, true),
                n_cells_max = 100_000) # set maximum capacity of tree data structure

initial_condition = initial_condition_gaussian_noise_mhd
source_terms = source_terms_helical_forcing

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             source_terms = source_terms)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.001)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

cfl = 1.5
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback,
                        save_restart)

###############################################################################
# run the simulation

time_int_tol = 1e-5
sol = solve(ode, RDPK3SpFSAL49(), abstol = time_int_tol, reltol = time_int_tol, dt = 1e-5,
            save_everystep = false, callback = callbacks)
summary_callback() # print the timer summary
