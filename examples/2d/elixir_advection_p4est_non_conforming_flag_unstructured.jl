
using Downloads: download
using OrdinaryDiffEq
using Trixi


###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

initial_condition = initial_condition_convergence_test

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(
  :all => boundary_condition
)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, s - 1.0)
f2(s) = SVector( 1.0, s + 1.0)
f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
f4(s) = SVector(s,  1.0 + sin(0.5 * pi * s))
faces = (f1, f2, f3, f4)

Trixi.validate_faces(faces)
mapping_flag = Trixi.transfinite_mapping(faces)

# Unstructured mesh with 24 cells of the square domain [-1, 1]^n
mesh_file = joinpath(@__DIR__, "square_unstructured_2.inp")
isfile(mesh_file) || download("https://gist.githubusercontent.com/efaulhaber/63ff2ea224409e55ee8423b3a33e316a/raw/7db58af7446d1479753ae718930741c47a3b79b7/square_unstructured_2.inp",
                              mesh_file)

mesh = P4estMesh(mesh_file, polydeg=3,
                 mapping=mapping_flag,
                 initial_refinement_level=0)

# Refine bottom left quadrant of each tree to level 3
function refine_fn(p4est, which_tree, quadrant)
  if quadrant.x == 0 && quadrant.y == 0 && quadrant.level < 3
    # return true (refine)
    return Cint(1)
  else
    # return false (don't refine)
    return Cint(0)
  end
end

# Refine recursively until each bottom left quadrant of a tree has level 4
# The mesh will be rebalanced before the simulation starts
refine_fn_c = @cfunction(refine_fn, Cint, (Ptr{Trixi.p4est_t}, Ptr{Trixi.p4est_topidx_t}, Ptr{Trixi.p4est_quadrant_t}))
Trixi.p4est_refine(mesh.p4est, true, refine_fn_c, C_NULL)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 0.2));

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
