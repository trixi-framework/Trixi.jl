using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation with a spatially varying velocity

equations = ConservativeLinearScalarAdvectionEquation3D()

# Divergence-free shear flow in x-direction, prescribed at every node of the mesh. The
# auxiliary variables of `ConservativeLinearScalarAdvectionEquation3D` are the components
# `(a1, a2, a3)` of the advection velocity.
function velocity_shear(x)
    a1 = 1 + 0.5f0 * sinpi(x[2]) * cospi(x[3])
    return SVector(a1, zero(a1), zero(a1))
end

aux_field(x, equations::ConservativeLinearScalarAdvectionEquation3D) = velocity_shear(x)

# Since the velocity in x-direction does not depend on x, the exact solution is given by
# shifting the initial condition along the x-direction with the local velocity.
function initial_condition_sheared_wave(x, t,
                                        equations::ConservativeLinearScalarAdvectionEquation3D)
    a1 = velocity_shear(x)[1]
    return SVector(1 + 0.5f0 * sinpi(x[1] - a1 * t + x[2] + x[3]))
end

initial_condition = initial_condition_sheared_wave

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
# once at all nodes.
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

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
