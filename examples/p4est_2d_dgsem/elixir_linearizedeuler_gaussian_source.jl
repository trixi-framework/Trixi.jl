
using OrdinaryDiffEq
using Trixi

# Based on the TreeMesh example `elixir_acoustics_gaussian_source.jl`.
# The acoustic perturbation equations have been replaced with the linearized Euler
# equations and instead of the Cartesian `TreeMesh` a rotated `P4estMesh` is used

# Oscillating Gaussian-shaped source terms
function source_terms_gauss(u, x, t, equations::LinearizedEulerEquations2D)
    r = 0.1
    A = 1.0
    f = 2.0

    # Velocity sources
    s2 = 0.0
    s3 = 0.0
    # Density and pressure source
    s1 = s4 = exp(-(x[1]^2 + x[2]^2) / (2 * r^2)) * A * sin(2 * pi * f * t)

    return SVector(s1, s2, s3, s4)
end

function initial_condition_zero(x, t, equations::LinearizedEulerEquations2D)
    SVector(0.0, 0.0, 0.0, 0.0)
end

###############################################################################
# semidiscretization of the linearized Euler equations

# Create a domain that is a 30° rotated version of [-3, 3]^2
c = cospi(2 * 30.0 / 360.0)
s = sinpi(2 * 30.0 / 360.0)
rot_mat = Trixi.SMatrix{2, 2}([c -s; s c])
mapping(xi, eta) = rot_mat * SVector(3.0 * xi, 3.0 * eta)

# Mean density and speed of sound are slightly off from 1.0 to allow proper verification of
# curved LEE implementation using this elixir (some things in the LEE cancel if both are 1.0)
equations = LinearizedEulerEquations2D(v_mean_global = Tuple(rot_mat * SVector(-0.5, 0.25)),
                                       c_mean_global = 1.02, rho_mean_global = 1.01)

initial_condition = initial_condition_zero

# Create DG solver with polynomial degree = 3 and upwind flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_godunov)

# Create a uniformly refined mesh with periodic boundaries
trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 mapping = mapping,
                 periodicity = true, initial_refinement_level = 2)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_gauss)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 2.0
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.5)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
