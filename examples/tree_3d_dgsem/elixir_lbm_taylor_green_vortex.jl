
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the Lattice-Boltzmann equations for the D3Q27 scheme

L = 1.0 # reference length
equations = LatticeBoltzmannEquations3D(Ma = 0.1, Re = 1600.0; L = L)

"""
    initial_condition_taylor_green_vortex(x, t, equations::LatticeBoltzmannEquations3D)

Initialize the flow field to the Taylor-Green vortex setup
"""
function initial_condition_taylor_green_vortex(x, t, equations::LatticeBoltzmannEquations3D)
    @unpack u0, rho0, L = equations

    v1 = u0 * sin(x[1] / L) * cos(x[2] / L) * cos(x[3] / L)
    v2 = -u0 * cos(x[1] / L) * sin(x[2] / L) * cos(x[3] / L)
    v3 = 0
    p0 = (pressure(rho0, equations) +
          rho0 * u0^2 / 16 * (cos(2 * x[1] / L) + cos(2 * x[2] / L)) *
          (cos(2 * x[3] / L) + 2))
    rho = density(p0, equations)

    return equilibrium_distribution(rho, v1, v2, v3, equations)
end
initial_condition = initial_condition_taylor_green_vortex

solver = DGSEM(polydeg = 3, surface_flux = flux_godunov)

coordinates_min = (-pi * L, -pi * L, -pi * L)
coordinates_max = (pi * L, pi * L, pi * L)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 300_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20 * equations.L / equations.u0) # Final time is `20` in non-dimensional time
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 20
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_integrals = (Trixi.energy_kinetic_nondimensional,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2macroscopic)

stepsize_callback = StepsizeCallback(cfl = 0.3)

collision_callback = LBMCollisionCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        collision_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks,
            save_start = false, alias_u0 = true);
summary_callback() # print the timer summary
