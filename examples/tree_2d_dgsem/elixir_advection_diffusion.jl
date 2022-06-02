using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation

advection_velocity = (1.5, 1.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
equations_parabolic = LaplaceDiffusion2D(5.0e-2, equations)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# Define initial conditions
initial_condition_zero(x, t, equations::LinearScalarAdvectionEquation2D) = SVector(0.0)
initial_condition = initial_condition_zero

# BC types
boundary_condition_left = BoundaryConditionDirichlet((x, t, equations) -> SVector(1 + 0.1 * x[2]))
boundary_condition_zero = BoundaryConditionDirichlet((x, t, equations) -> SVector(0.0))
boundary_condition_neumann_zero = BoundaryConditionNeumann((x, t, equations) -> SVector(0.0))

# define inviscid boundary conditions
boundary_conditions = (
                       x_neg=boundary_condition_left,
                       y_neg=boundary_condition_zero,
                       y_pos=boundary_condition_do_nothing,
                       x_pos=boundary_condition_do_nothing,
                      )

# define viscous boundary conditions
boundary_conditions_parabolic = (
                                 x_neg=boundary_condition_left,
                                 y_neg=boundary_condition_zero,
                                 y_pos=boundary_condition_zero,
                                 x_pos=boundary_condition_neumann_zero,
                                )

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, dg;
                                             boundary_conditions=(boundary_conditions,
                                                                  boundary_conditions_parabolic))


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.5));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
# analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval=analysis_interval)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
# callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)
callbacks = CallbackSet(summary_callback, alive_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
time_int_tol = 1e-6
sol = solve(ode, RDPK3SpFSAL49(), abstol=time_int_tol, reltol=time_int_tol,
            save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()
