
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0, 1.0)
equations = LinearScalarAdvectionEquation3D(advectionvelocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

initial_condition = initial_condition_convergence_test
boundary_condition = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(
  :all => boundary_condition
)

# Mapping as described in https://arxiv.org/abs/2012.12040, but with less warping.
# The original mapping applied to this unstructured mesh creates extreme angles,
# which require a high resolution for proper results.
function mapping(xi, eta, zeta)
  # Don't transform input variables between -1 and 1 onto [0,3] to obtain curved boundaries
  # xi = 1.5 * xi_ + 1.5
  # eta = 1.5 * eta_ + 1.5
  # zeta = 1.5 * zeta_ + 1.5

  y = eta + 1/6 * (cos(1.5 * pi * (2 * xi - 3)/3) *
                   cos(0.5 * pi * (2 * eta - 3)/3) *
                   cos(0.5 * pi * (2 * zeta - 3)/3))

  x = xi + 1/6 * (cos(0.5 * pi * (2 * xi - 3)/3) *
                  cos(2 * pi * (2 * y - 3)/3) *
                  cos(0.5 * pi * (2 * zeta - 3)/3))

  z = zeta + 1/6 * (cos(0.5 * pi * (2 * x - 3)/3) *
                    cos(pi * (2 * y - 3)/3) *
                    cos(0.5 * pi * (2 * zeta - 3)/3))

  return SVector(x, y, z)
end

# Unstructured mesh with 68 cells of the cube domain [-1, 1]^3
mesh_file = joinpath(@__DIR__, "cube_unstructured_1.inp")
isfile(mesh_file) || download("https://gist.githubusercontent.com/efaulhaber/d45c8ac1e248618885fa7cc31a50ab40/raw/37fba24890ab37cfa49c39eae98b44faf4502882/cube_unstructured_1.inp",
                              mesh_file)

mesh = P4estMesh{3}(mesh_file, polydeg=3,
                    mapping=mapping,
                    initial_refinement_level=2)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 0.1
ode = semidiscretize(semi, (0.0, 0.1));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

# The SaveRestartCallback allows to save a file from which a Trixi simulation can be restarted
save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.2)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_restart, save_solution, stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
