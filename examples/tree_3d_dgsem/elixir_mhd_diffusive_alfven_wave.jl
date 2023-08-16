
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the visco-resistive compressible MHD equations

prandtl_number() = 0.72
mu() = 4e-2
eta() = 4e-2

equations = IdealGlmMhdEquations3D(1.4)
equations_parabolic = ViscoResistiveMhd3D(equations, mu = mu(),
                                          Prandtl = prandtl_number(),
                                          eta = eta(),
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
                n_cells_max = 50_000) # set maximum capacity of tree data structure

function initial_condition_constant_alfven(x, t, equations)
    # Homogeneous background magnetic field in the x-direction that is perturbed
    # by a small change in the y-direction (e.g. Biskamp 2003, section 2.5.1).
    # This particular set-up is in line with Rembiasz et al. (2018)
    # DOI: doi.org/10.3847/1538-4365/aa6254, but with p = 1.
    # For a derivation of Alfven waves see e.g.:
    # Alfven H., 150, p. 450, Nature (1942), DOI: 10.1038/150405d0
    # Chandrasekhar, Hydrodynamic and hydromagnetic stability (1961)
    # Biskamp, Magnetohydrodynamic Turbulence (2003)

    epsilon = 0.02
    k = 2*pi*1
    p = 2e-3

    rho = 1.0
    rho_v1 = 0
    rho_v2 = -epsilon*sin(k*x[1])/sqrt(rho)
    rho_v3 = 0
    B1 = 1
    B2 = epsilon*sin(k*x[1])
    B3 = 0
    rho_e = 1
    psi = 0

    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
end

@inline function source_terms_mhd_convergence_test(u, x, t, equations)
    r_1 = 0
    r_2 = -0.0008*pi*sin(2*pi*x[1])*cos(2*pi*x[1])
    r_3 = -0.0032*pi^2*sin(2*pi*x[1]) + 0.04*pi*cos(2*pi*x[1])
    r_4 = 0
    r_5 = 0.001664*pi^2*sin(2*pi*x[1])^2 - 0.0016*pi*sin(2*pi*x[1])*cos(2*pi*x[1]) -
          0.001664*pi^2*cos(2*pi*x[1])^2
    r_6 = 0
    r_7 = 0.0032*pi^2*sin(2*pi*x[1]) - 0.04*pi*cos(2*pi*x[1])
    r_8 = 0
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
tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = 10,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 0.25
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-5
sol = solve(ode, RDPK3SpFSAL49(), dt = 1e-5,
            save_everystep = false, callback = callbacks, adaptive = false)

# Print the timer summary.
summary_callback()
