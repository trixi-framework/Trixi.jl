using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the linear advection equation.

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

initial_condition = initial_condition_convergence_test

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :inside => boundary_condition,
                         :outside => boundary_condition)

###############################################################################
# Build DG solver.

polydeg = 3

dg = DGMulti(polydeg = polydeg, element_type = Hex(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralFluxDifferencing(flux_central))

###############################################################################
# Build mesh.

lat_lon_elements = 5
layers = 3
inner_radius = 0.5
thickness = 0.5
outer_radius = inner_radius + thickness
initial_refinement_level = 0

is_on_boundary = Dict(
  :inside => (x,y,z; tol = 50*eps()) -> abs(sqrt(x^2 + y^2 + z^2) - inner_radius) < tol,
  :outside => (x,y,z; tol = 50*eps()) -> abs(sqrt(x^2 + y^2 + z^2) - outer_radius) < tol)

cmesh = Trixi.t8_cmesh_new_cubed_spherical_shell(
  inner_radius, thickness, lat_lon_elements, layers, Trixi.mpi_comm())

mesh = DGMultiMesh(dg, cmesh;
  initial_refinement_level = initial_refinement_level, is_on_boundary = is_on_boundary)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0.
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers.
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results.
analysis_callback = AnalysisCallback(semi, interval = 100, uEltype = real(dg))

# The SaveSolutionCallback allows to save the solution to a file in regular intervals.
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step.
stepsize_callback = StepsizeCallback(cfl = 1.2)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver.
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# Run the simulation.

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks.
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = estimate_dt(mesh, dg),
            save_everystep = false, callback = callbacks);

# Print the timer summary.
summary_callback()
