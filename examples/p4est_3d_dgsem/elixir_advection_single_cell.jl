using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

polydeg = 2
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)

initial_condition = initial_condition_constant

case_path = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS1/"
mesh_file = case_path * "Pointwise/TandemSpheresHexMesh1P1.inp"
#mesh_file = case_path * "Pointwise/TandemSpheresHexMesh1P2_fixed.inp"
mesh_file = case_path * "Pointwise/single_cell_Q27_test_straight.inp"

# Boundary symbols follow from nodesets in the mesh file
mesh = P4estMesh{3}(mesh_file)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, (0.0, 0.0))

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.2)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, ode_default_options()..., callback = callbacks);
