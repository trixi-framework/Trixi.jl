using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
# with two additional passive tracer variables in a closed box with slip walls

gamma = 1.4
flow_equations = CompressibleEulerEquations2D(gamma)
equations = PassiveTracerEquations(flow_equations, n_tracers = 2)

# A smooth pressure pulse that is reflected at the walls, transporting a Gaussian blob
# (first tracer) and a linear profile (second tracer)
function initial_condition_pulse_tracers(x, t, equations::PassiveTracerEquations)
    r2 = (x[1] - 0.2f0)^2 + (x[2] + 0.1f0)^2
    rho = 1 + 0.1f0 * exp(-20 * r2)
    v1 = 0.1f0
    v2 = -0.05f0
    p = 1 + 0.2f0 * exp(-20 * r2)
    tracer1 = exp(-10 * ((x[1] + 0.3f0)^2 + (x[2] - 0.2f0)^2))
    tracer2 = 0.5f0 + 0.25f0 * x[1] - 0.25f0 * x[2]
    return prim2cons(SVector(rho, v1, v2, p, tracer1, tracer2), equations)
end
initial_condition = initial_condition_pulse_tracers

volume_flux = FluxTracerEquationsCentral(flux_ranocha)
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
trees_per_dimension = (8, 8)

mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = false)

# The mass flux and thus the tracer fluxes vanish at the slip walls,
# so that the total mass of each tracer is conserved
boundary_conditions = (; x_neg = boundary_condition_slip_wall,
                       x_pos = boundary_condition_slip_wall,
                       y_neg = boundary_condition_slip_wall,
                       y_pos = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
