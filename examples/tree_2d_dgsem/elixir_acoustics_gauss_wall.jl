using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the acoustic perturbation equations

equations = AcousticPerturbationEquations2D(v_mean_global = (0.5, 0.0), c_mean_global = 1.0,
                                            rho_mean_global = 1.0)

# Create DG solver with polynomial degree = 5 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 5, surface_flux = flux_lax_friedrichs)

coordinates_min = (-100.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (100.0, 200.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 100_000,
                periodicity = false)

"""
    initial_condition_gauss_wall(x, t, equations::AcousticPerturbationEquations2D)

A Gaussian pulse, used in the `gauss_wall` example elixir in combination with
[`boundary_condition_wall`](@ref). Uses the global mean values from `equations`.
"""
function initial_condition_gauss_wall(x, t, equations::AcousticPerturbationEquations2D)
    v1_prime = 0.0
    v2_prime = 0.0
    p_prime = exp(-log(2) * (x[1]^2 + (x[2] - 25)^2) / 25)

    prim = SVector(v1_prime, v2_prime, p_prime, global_mean_vars(equations)...)

    return prim2cons(prim, equations)
end
initial_condition = initial_condition_gauss_wall

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_wall)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 30.0
tspan = (0.0, 30.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100, solution_variables = cons2state)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.7)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks)

# Print the timer summary
summary_callback()
