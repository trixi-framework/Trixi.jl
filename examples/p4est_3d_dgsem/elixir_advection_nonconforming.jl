
using OrdinaryDiffEq
using Trixi


###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = ( 1.0,  1.0,  1.0) # maximum coordinates (max(x), max(y), max(z))
trees_per_dimension = (1, 1, 1)

# Note that it is not necessary to use mesh polydeg lower than the solver polydeg
# on a Cartesian mesh.
# See https://doi.org/10.1007/s10915-018-00897-9, Section 6.
mesh = P4estMesh(trees_per_dimension, polydeg=3,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=2)

# Refine bottom left quadrant of each tree to level 3
function refine_fn(p8est, which_tree, quadrant)
  quadrant_obj = unsafe_load(quadrant)
  if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.z == 0 && quadrant_obj.level < 3
    # return true (refine)
    return Cint(1)
  else
    # return false (don't refine)
    return Cint(0)
  end
end

# Refine recursively until each bottom left quadrant of a tree has level 3
# The mesh will be rebalanced before the simulation starts
refine_fn_c = @cfunction(refine_fn, Cint, (Ptr{Trixi.p8est_t}, Ptr{Trixi.p4est_topidx_t}, Ptr{Trixi.p8est_quadrant_t}))
Trixi.refine_p4est!(mesh.p4est, true, refine_fn_c, C_NULL)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
