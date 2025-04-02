using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear (advection) diffusion equation

advection_velocity = 0.0 # Note: This renders the equation mathematically purely parabolic
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 0.5
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -convert(Float64, pi) # minimum coordinate
coordinates_max = convert(Float64, pi) # maximum coordinate

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000, # set maximum capacity of tree data structure
                periodicity = true)

# Define initial condition if it is not defined already.
# For CI, the function is defined externally avoid "world age" issues that arise 
# when running `Trixi.convergence_test`. The `isdefined` check is to allow the 
# elixir to also be run outside of CI. 
function initial_condition_pure_diffusion_1d_convergence_test(x, t,
                                                              equation)
    nu = diffusivity()
    c = 0
    A = 1
    omega = 1
    scalar = c + A * sin(omega * sum(x)) * exp(-nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_pure_diffusion_1d_convergence_test

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

# A semidiscretization collects data structures and functions for the spatial discretization
solver_parabolic = ViscousFormulationLocalDG()
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition,
                                             solver;
                                             solver_parabolic,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = 100)

# The SaveRestartCallback allows to save a file from which a Trixi.jl simulation can be restarted
save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_restart)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
# For CI purposes, we use fixed time-stepping for this elixir. 
sol = solve(ode, RDPK3SpFSAL35(); dt = 1.0e-3, adaptive = false,
            ode_default_options()..., callback = callbacks)
