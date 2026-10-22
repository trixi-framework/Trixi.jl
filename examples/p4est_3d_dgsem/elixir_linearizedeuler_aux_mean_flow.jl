using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linearized Euler equations with a spatially varying mean flow

equations = LinearizedEulerEquations3DAuxVars()

# Sheared mean flow in x-direction, prescribed at every node of the mesh. The auxiliary
# variables of `LinearizedEulerEquations3DAuxVars` are
# `(rho_mean, v1_mean, v2_mean, v3_mean, c_mean)`.
function aux_field(x, equations::LinearizedEulerEquations3DAuxVars)
    v1_mean = 0.5f0 * sinpi(x[2])
    return SVector(1.0f0, v1_mean, 0.0f0, 0.0f0, 1.0f0)
end

# A Gaussian pressure pulse at rest
function initial_condition_gauss_pulse(x, t, equations::LinearizedEulerEquations3DAuxVars)
    p_prime = exp(-16 * (x[1]^2 + x[2]^2 + x[3]^2))
    return SVector(0.0f0, 0.0f0, 0.0f0, 0.0f0, p_prime)
end

initial_condition = initial_condition_gauss_pulse

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)
trees_per_dimension = (4, 4, 4)

mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min,
                 coordinates_max = coordinates_max,
                 initial_refinement_level = 0,
                 periodicity = true)

# The `aux_field` is required since `equations` has auxiliary variables. They are evaluated
# once at all nodes and stored in the cache.
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic,
                                    aux_field = aux_field)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
