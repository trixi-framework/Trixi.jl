
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the visco-resistive compressible MHD equations

prandtl_number() = 0.72
mu() = 0e-2
eta() = 0e-2
mu_const = mu()
eta_const = eta()

equations = IdealGlmMhdEquations3D(5/3)
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
                n_cells_max = 100_000) # set maximum capacity of tree data structure

#function initial_condition_constant_alfven(x, t, equations)
#    # Homogeneous background magnetic field in the diagonl direction that is perturbed
#    # by a small change in the orthogonal direction (e.g. Biskamp 2003, section 2.5.1).
#    # This particular set-up is a modification of Rembiasz et al. (2018)
#    # DOI: doi.org/10.3847/1538-4365/aa6254, but with p = 1.
#    # For a derivation of Alfven waves see e.g.:
#    # Alfven H., 150, p. 450, Nature (1942), DOI: 10.1038/150405d0
#    # Chandrasekhar, Hydrodynamic and hydromagnetic stability (1961)
#    # Biskamp, Magnetohydrodynamic Turbulence (2003)
#
#    epsilon = 0.02
#    k = 2*pi*1
#    p = 2e-3
#
#    rho = 1.0
#    rho_v1 = 0
#    rho_v2 = -epsilon*sin(k*x[1])/sqrt(rho)
#    rho_v3 = 0
#    B1 = 1
#    B2 = epsilon*sin(k*x[1])
#    B3 = 0
#    rho_e = 1
#    psi = 0
#
#    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
#end
#
#@inline function source_terms_mhd_convergence_test(u, x, t, equations)
#    r_1 = 0
#    r_2 = -0.000266666666666667*pi*sin(2*pi*x[1])*cos(2*pi*x[1])
#    r_3 = -0.08*pi^2*mu_const*sin(2*pi*x[1]) - 0.04*pi*cos(2*pi*x[1])
#    r_4 = 0
#    r_5 = 0.0016*pi^2*eta_const*sin(2*pi*x[1])^2 +
#          -0.0016*pi^2*eta_const*cos(2*pi*x[1])^2 +
#	  -mu_const*(-0.0016*pi^2*sin(2*pi*x[1])^2 + 0.0016*pi^2*cos(2*pi*x[1])^2) +
#	  0.0016*pi*sin(2*pi*x[1])*cos(2*pi*x[1])
#    r_6 = 0
#    r_7 = 0.08*pi^2*eta_const*sin(2*pi*x[1]) + 0.04*pi*cos(2*pi*x[1])
#    r_8 = 0
#    r_9 = 0
#
#    return SVector(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9)
#end

function initial_condition_convergence_test(x, t, equations)
    # Initial condition for the convergence test.
    # This form has been conjured up by guess work just for this test.

    l = x[1]^2 + 2*x[2]^2 + 3*x[3]^2

    rho = exp(-l/10)
    rho_v1 = rho*x[2]
    rho_v2 = rho*x[1]
    rho_v3 = 1/10
    B1 = x[2]/10 + x[1]/100
    B2 = x[3]/10
    B3 = x[1]/10
    rho_e = 1
    psi = 0

    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
end

@inline function source_terms_mhd_convergence_test(u, x, t, equations)
    l = x[1]^2 + 2*x[2]^2 + 3*x[3]^2
#     z = x[3]
#     y = x[2]
#     x = x[1]

    r_1 =  -(3*x[1]*x[2] + 0.3*x[3])*exp(-x[1]^2/10 - x[2]^2/5 - 3*x[3]^2/10)/5
#     r_2 = (0.0666666666666667*x^3 - 0.533333333333333*x*y^2 + 0.00316666666666667*x*exp(x^2/10 + y^2/5 + 3*z^2/10) + 0.334*x - 0.06*y*z - 0.00166666666666667*y*exp(x^2/10 + y^2/5 + 3*z^2/10) - 0.01*z*exp(x^2/10 + y^2/5 + 3*z^2/10))*exp(-x^2/10 - y^2/5 - 3*z^2/10)
#     r_3 = (-0.466666666666667*x^2*y - 0.06*x*z - 0.00966666666666667*x*exp(x^2/10 + y^2/5 + 3*z^2/10) + 0.133333333333333*y^3 + 0.00333333333333333*y*exp(x^2/10 + y^2/5 + 3*z^2/10) + 0.334666666666667*y - 0.001*z*exp(x^2/10 + y^2/5 + 3*z^2/10))*exp(-x^2/10 - y^2/5 - 3*z^2/10)
#     r_4 = -0.06*x*y*exp(-x^2/10 - y^2/5 - 3*z^2/10) - x/500 - y/100 + 3*z*(0.333333333333333*x^2 + 0.333333333333333*y^2 + 0.00333333333333333)*exp(-x^2/10 - y^2/5 - 3*z^2/10)/5 - 0.006*z*exp(-x^2/10 - y^2/5 - 3*z^2/10) + 0.00333333333333333*z
#     r_5 = -0.03*eta_const - 4.0*mu_const + 0.2*x^3*y*exp(-x^2/10 - y^2/5 - 3*z^2/10) + 0.02*x^2*z*exp(-x^2/10 - y^2/5 - 3*z^2/10) - 0.00966666666666667*x^2 + 0.2*x*y^3*exp(-x^2/10 - y^2/5 - 3*z^2/10) - 1.33133333333333*x*y*exp(-x^2/10 - y^2/5 - 3*z^2/10) + 0.0065*x*y - 0.003*x*z - 0.0002*x + 0.02*y^2*z*exp(-x^2/10 - y^2/5 - 3*z^2/10) - 0.00166666666666667*y^2 - 0.03*y*z - 0.001*y + 0.0002*z*exp(-x^2/10 - y^2/5 - 3*z^2/10) + 0.000333333333333333*z
#     r_6 = x/10 - z/10
#     r_7 = -x/50 - y/10 + 0.01
#     r_8 = y/10 - 0.001
#     r_9 = equations.c_h/100

#     return SVector(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9)
    return SVector(r_1, 0, 0, 0, 0, 0, 0, 0, 0)
end

#initial_condition = initial_condition_constant_alfven
initial_condition = initial_condition_convergence_test

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
					     initial_condition, solver,
					     source_terms = source_terms_mhd_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.01)
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

#time_int_tol = 1e-5
#sol = solve(ode, RDPK3SpFSAL49(), dt = 1e-5,
#            save_everystep = false, callback = callbacks, adaptive = false)
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1e-5, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
# Print the timer summary.
summary_callback()
