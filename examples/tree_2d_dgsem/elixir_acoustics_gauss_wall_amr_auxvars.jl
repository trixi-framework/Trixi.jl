using OrdinaryDiffEqLowStorageRK
using Trixi
using Plots

###############################################################################
# semidiscretization of the acoustic perturbation equations

equations = AcousticPerturbationEquations2DAuxVars(v_mean_global = (0.5, 0.0),
                                                   c_mean_global = 1.0,
                                                   rho_mean_global = 1.0)

# Create DG solver with polynomial degree = 5 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 5, surface_flux = flux_lax_friedrichs)

coordinates_min = (-100.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (100.0, 200.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 100_000,
                periodicity = false)

"""
    initial_condition_gauss_wall(x, t, equations::AcousticPerturbationEquations2DAuxVars)

A Gaussian pulse, used in the `gauss_wall` example elixir in combination with
[`boundary_condition_wall`](@ref). Uses the global mean values from `equations`.
"""
function initial_condition_gauss_wall(x, t,
                                      equations::AcousticPerturbationEquations2DAuxVars)
    RealT = eltype(x)
    v1_prime = 0
    v2_prime = 0
    p_prime = exp(-log(convert(RealT, 2)) * (x[1]^2 + (x[2] - 25)^2) / 25)
    p_prime_scaled = p_prime / equations.c_mean_global^2

    return SVector(v1_prime, v2_prime, p_prime_scaled)
end
initial_condition = initial_condition_gauss_wall

function auxiliary_variables_mean_values(x, equations)
    # constant auxiliary variables (mean state)
    return global_mean_vars(equations)
end

@inline function p_prime(u, equations::AcousticPerturbationEquations2DAuxVars)
    return u[3]
end

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_wall,
                                    auxiliary_field = auxiliary_variables_mean_values)

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

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = p_prime),
                                      base_level = 2,
                                      med_level = 4, med_threshold = 0.1,
                                      max_level = 5, max_threshold = 0.2)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

visualization = VisualizationCallback(interval = 5, show_mesh = true,
                                      variable_names = ["v1_prime", "v2_prime", "p_prime"])

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, #visualization,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks)
