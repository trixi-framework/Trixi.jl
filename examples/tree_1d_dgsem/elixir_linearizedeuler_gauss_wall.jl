
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linearized Euler equations

equations = LinearizedEulerEquations1D(v_mean_global = 0.5, c_mean_global = 1.0,
                                       rho_mean_global = 1.0)

solver = DGSEM(polydeg = 5, surface_flux = flux_hll)

coordinates_min = (0.0,)
coordinates_max = (90.0,)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 100_000,
                periodicity = false)

# Initialize density and pressure perturbation with a Gaussian bump
# that is advected to left with v - c and to the right with v + c.
# Correspondigly, the bump splits in half.
function initial_condition_gauss_wall(x, t, equations::LinearizedEulerEquations1D)
    v1_prime = 0.0
    rho_prime = p_prime = 2 * exp(-(x[1] - 45)^2 / 25)
    return SVector(rho_prime, v1_prime, p_prime)
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

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.7)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks)

# Print the timer summary
summary_callback()
