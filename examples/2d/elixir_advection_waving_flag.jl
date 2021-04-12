
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(3, flux_lax_friedrichs)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, s - 1.0)
f2(s) = SVector( 1.0, s + 1.0)
f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
f4(s) = SVector(s,  1.0 + sin(0.5 * pi * s))

cells_per_dimension = (16, 16)

# Create curved mesh with 16 x 16 elements
mesh = CurvedMesh(cells_per_dimension, (f1, f2, f3, f4))

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
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

using WriteVTK

# The following lines represent an MWE for a dummy structured VTK output
#
# Ni = 3
# Nj = 3
# Lx = 2
# Ly = 2
# xy = Array{Float64}(undef, 2, Ni, Nj)
# 
# for j in 1:Nj, i in 1:Ni
#   xy[1, i, j] = Lx/Ni * (i - 1)
#   xy[2, i, j] = Ly/Nj * (j - 1) + Ly/Nj/3 * (i - 1)
# end
# 
# vtk_grid("testfile", xy) do vtk
#   vtk["scalar"] = collect(Float64, reshape(1:4, 2, 2))
# end

function getxy(sol)
  semi = sol.prob.p
  mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
  node_coordinates = semi.cache.elements.node_coordinates
  nnodes_ = nnodes(solver)
  linear_indices = LinearIndices(size(mesh))
  Nx = size(mesh, 1)
  Ny = size(mesh, 2)

  xy = Array{Float64}(undef, 2, Nx * (nnodes_ - 1) + 1, Ny * (nnodes_ - 1) + 1)

  for cell_y in axes(mesh, 2), cell_x in axes(mesh, 1)
    for j in 1:nnodes_, i in 1:nnodes_
      index_x = (cell_x - 1) * (nnodes_ - 1) + i
      index_y = (cell_y - 1) * (nnodes_ - 1) + j
      xy[1, index_x, index_y] = node_coordinates[1, i, j, linear_indices[cell_x, cell_y]]
      xy[2, index_x, index_y] = node_coordinates[2, i, j, linear_indices[cell_x, cell_y]]
    end
  end

  for cell_y in axes(mesh, 2), cell_x in Nx
    for j in 1:nnodes_
      index_y = (cell_y - 1) * (nnodes_ - 1) + j
      xy[1, end, index_y] = node_coordinates[1, end, j, linear_indices[cell_x, cell_y]]
      xy[2, end, index_y] = node_coordinates[2, end, j, linear_indices[cell_x, cell_y]]
    end
  end

  for cell_y in Ny, cell_x in axes(mesh, 1)
    for i in 1:nnodes_
      index_x = (cell_x - 1) * (nnodes_ - 1) + i
      xy[1, index_x, end] = node_coordinates[1, i, end, linear_indices[cell_x, cell_y]]
      xy[2, index_x, end] = node_coordinates[2, i, end, linear_indices[cell_x, cell_y]]
    end
  end

  return xy
end

function getscalar(sol)
  u_ode = sol.u[end]
  semi = sol.prob.p
  u = Trixi.wrap_array(u_ode, semi)
  mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
  Nx = size(mesh, 1)
  Ny = size(mesh, 2)
  nvisnodes = nnodes(solver) - 1

  scalar = Array{Float64}(undef, Nx * nvisnodes, Ny * nvisnodes)
  linear_indices = LinearIndices(size(mesh))
  for cell_y in axes(mesh, 2), cell_x in axes(mesh, 1)
    for j in 1:nvisnodes, i in 1:nvisnodes
      index_x = (cell_x - 1) * nvisnodes + i
      index_y = (cell_y - 1) * nvisnodes + j
      scalar[index_x, index_y] = (u[1, i,   j,   linear_indices[cell_x, cell_y]] + 
                                  u[1, i+1, j,   linear_indices[cell_x, cell_y]] + 
                                  u[1, i,   j+1, linear_indices[cell_x, cell_y]] + 
                                  u[1, i+1, j+1, linear_indices[cell_x, cell_y]]) / 4
    end
  end

  return scalar
end

vtk_grid("curved", getxy(sol)) do vtk
  vtk["scalar"] = getscalar(sol)
end
