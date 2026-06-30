using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the pure diffusion equation

diffusivity() = 0.5
equations = LinearDiffusionEquation1D(diffusivity())

# Create DG solver with polynomial degree = 3
solver = DGSEM(polydeg = 3)
solver_parabolic = ParabolicFormulationLocalDG()

# Create a uniformly refined mesh with nonperiodic boundaries
mesh = TreeMesh(0.0, 1.0,
                initial_refinement_level = 4,
                n_cells_max = 30_000, # set maximum capacity of tree data structure
                periodicity = false)

function analytical_solution(x, t, equations)
    scalar = sinpi(x[1]) * exp(-diffusivity() * pi^2 * t)
    return SVector(scalar)
end
initial_condition = analytical_solution

boundary_conditions = (; x_neg = BoundaryConditionDirichlet(initial_condition),
                       x_pos = BoundaryConditionDirichlet(initial_condition))

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic = solver_parabolic,
                                   boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = 100)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
# For CI purposes, we use fixed time-stepping for this elixir.
sol = solve(ode, RDPK3SpFSAL35(); dt = 1.0e-4, adaptive = false,
            ode_default_options()..., callback = callbacks)
