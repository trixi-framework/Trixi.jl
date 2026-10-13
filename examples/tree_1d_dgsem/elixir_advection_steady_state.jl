using NonlinearSolveFirstOrder
using ADTypes
using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the hyperbolic diffusion equations

advection_velocity = (1e-3)
equations = LinearScalarAdvectionEquation1D(advection_velocity)

function initial_condition_zero(x, t, equations::LinearScalarAdvectionEquation1D)
  return SVector(0)
end
initial_condition = initial_condition_zero

bc_one(x, t, equations::LinearScalarAdvectionEquation1D) = SVector(1)

boundary_conditions = (x_neg = BoundaryConditionDirichlet(bc_one),
                       x_pos = boundary_condition_do_nothing)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (0.0,)
coordinates_max = (1.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE/Steady State problem

t_end = (coordinates_max[1] - coordinates_min[1]) / advection_velocity[1]
tspan = (t_end, t_end)

ode = semidiscretize(semi, tspan)

steady_state_prob = SteadyStateProblem(ode)

alg = NewtonRaphson(autodiff = AutoFiniteDiff())

sol_steady_state = NonlinearSolveFirstOrder.solve(steady_state_prob, alg)

# Supply steady state solution as initial condition for time-dependent run
ode.u0 .= sol_steady_state.u

###############################################################################
# ODE solvers, callbacks etc.

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
            ode_default_options()..., callback = callbacks);
