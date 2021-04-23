
using OrdinaryDiffEq
using Trixi


function boundary_condition_test_1_left()
  other_mesh_id = 2

  function prolong2boundary_(boundary_condition, u, mesh)
    linear_indices = LinearIndices(size(mesh))

    for cell_y in axes(mesh, 2), j in 1:size(u, 3), v in 1:size(u, 1)
      boundary_condition.u_boundary[v, j, cell_y] = u[v, end, j, linear_indices[end, cell_y]]
    end
  end

  Trixi.BoundaryConditionCoupled(other_mesh_id, prolong2boundary_, 2, Float64)
end


function boundary_condition_test_1_right()
  other_mesh_id = 2

  function prolong2boundary_(boundary_condition, u, mesh)
    linear_indices = LinearIndices(size(mesh))
    
    for cell_y in axes(mesh, 2), j in 1:size(u, 3), v in 1:size(u, 1)
      boundary_condition.u_boundary[v, j, cell_y] = u[v, 1, j, linear_indices[1, cell_y]]
    end
  end

  Trixi.BoundaryConditionCoupled(other_mesh_id, prolong2boundary_, 2, Float64)
end


function boundary_condition_test_2_left()
  other_mesh_id = 1

  function prolong2boundary_(boundary_condition, u, mesh)
    linear_indices = LinearIndices(size(mesh))

    for cell_y in axes(mesh, 2), j in 1:size(u, 3), v in 1:size(u, 1)
      boundary_condition.u_boundary[v, j, cell_y] = u[v, end, j, linear_indices[end, cell_y]]
    end
  end

  Trixi.BoundaryConditionCoupled(other_mesh_id, prolong2boundary_, 2, Float64)
end


function boundary_condition_test_2_right()
  other_mesh_id = 1

  function prolong2boundary_(boundary_condition, u, mesh)
    linear_indices = LinearIndices(size(mesh))
    
    for cell_y in axes(mesh, 2), j in 1:size(u, 3), v in 1:size(u, 1)
      boundary_condition.u_boundary[v, j, cell_y] = u[v, 1, j, linear_indices[1, cell_y]]
    end
  end

  Trixi.BoundaryConditionCoupled(other_mesh_id, prolong2boundary_, 2, Float64)
end

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (-0.5,  1.0) # maximum coordinates (max(x), max(y))

cells_per_dimension = (4, 16)

# Create curved mesh with 16 x 16 elements
mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi1 = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver, 
                                     boundary_conditions=(boundary_condition_test_1_left(), boundary_condition_test_1_right(),
                                                          boundary_condition_periodic, boundary_condition_periodic))

coordinates_min = (1.5, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (3.0,  1.0) # maximum coordinates (max(x), max(y))

cells_per_dimension = (12, 16)

# Create curved mesh with 16 x 16 elements
mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi2 = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                     boundary_conditions=(boundary_condition_test_2_left(), boundary_condition_test_2_right(),
                                                          boundary_condition_periodic, boundary_condition_periodic))

# mesh = CurvedMesh((4, 4), coordinates_min, coordinates_max)
# semi3 = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

semi = SemidiscretizationHyperbolicCoupled((semi1, semi2))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# # The SaveSolutionCallback allows to save the solution to a file in regular intervals
# save_solution = SaveSolutionCallback(interval=100,
#                                      solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, stepsize_callback, analysis_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
