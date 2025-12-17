using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# One Euler system is coupled to one MHD system.

# Pressure wave, same for the Euler system.
function initial_condition_mhd(x, t, equations::IdealGlmMhdEquations2D)
    rho = ((1.0 + 0.01 * sin(x[1] * 2 * pi)))
    v1 = ((0.01 * sin((x[1] - 1 / 2) * 2 * pi)))
    v2 = 0.0
    v3 = 0.0
    p = rho^equations.gamma
    B1 = 0.0
    B2 = 0.0
    B3 = 0.0
    psi = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

# Pressure wave, same as for the MHD system.
function initial_condition_euler(x, t, equations::CompressibleEulerEquations2D)
    rho = ((1.0 + 0.01 * sin(x[1] * 2 * pi)))
    v1 = ((0.01 * sin((x[1] - 1 / 2) * 2 * pi)))
    v2 = 0.0
    p = rho .^ equations.gamma

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

# Define the parent mesh.
coordinates_min = (-2.0, -2.0) # minimum coordinates (min(x), min(y))
coordinates_max = (2.0, 2.0) # maximum coordinates (max(x), max(y))
trees_per_dimension = (8, 8)
# Here we set the priodicity to false for the coupling.
# Since we couple through the physical boundaries the system is effectively periodic.
parent_mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                        coordinates_min = coordinates_min,
                        coordinates_max = coordinates_max,
                        initial_refinement_level = 0, periodicity = (false, false))

equations1 = IdealGlmMhdEquations2D(5 / 3)
equations2 = CompressibleEulerEquations2D(5 / 3)

# Define the coupling function between every possible pair of systems.
coupling_functions = Array{Function}(undef, 2, 2)
coupling_functions[1, 1] = (x, u, equations_other, equations_own) -> u
coupling_functions[1, 2] = (x, u, equations_other, equations_own) -> SVector(u[1], u[2],
                                                                             u[3], 0.0,
                                                                             u[4], 0.0, 0.0,
                                                                             0.0, 0.0)
coupling_functions[2, 1] = (x, u, equations_other, equations_own) -> SVector(u[1], u[2],
                                                                             u[3], u[5])
coupling_functions[2, 2] = (x, u, equations_other, equations_own) -> u

# semi 1 MHD.
cell_ids1 = vcat(Vector(1:8), Vector(32:64))
mesh1 = P4estMeshView(parent_mesh, cell_ids1)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver1 = DGSEM(polydeg = 3,
                surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
boundary_conditions1 = Dict(:x_neg => BoundaryConditionCoupledP4est(coupling_functions),
                            :y_neg => BoundaryConditionCoupledP4est(coupling_functions),
                            :y_pos => BoundaryConditionCoupledP4est(coupling_functions),
                            :x_pos => BoundaryConditionCoupledP4est(coupling_functions))
semi1 = SemidiscretizationHyperbolic(mesh1, equations1, initial_condition_mhd, solver1,
                                     boundary_conditions = boundary_conditions1)

# semi 2 Euler
cell_ids2 = Vector(9:31)
mesh2 = P4estMeshView(parent_mesh, cell_ids2)
solver2 = DGSEM(polydeg = 3, surface_flux = flux_hll,
                volume_integral = VolumeIntegralWeakForm())
boundary_conditions2 = Dict(:x_neg => BoundaryConditionCoupledP4est(coupling_functions),
                            :y_neg => BoundaryConditionCoupledP4est(coupling_functions),
                            :y_pos => BoundaryConditionCoupledP4est(coupling_functions),
                            :x_pos => BoundaryConditionCoupledP4est(coupling_functions))
semi2 = SemidiscretizationHyperbolic(mesh2, equations2, initial_condition_euler, solver2,
                                     boundary_conditions = boundary_conditions2)

# Create a semidiscretization that bundles semi1 and semi2
semi = SemidiscretizationCoupledP4est(semi1, semi2)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, (0.0, 1.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback1 = AnalysisCallback(semi1, interval = 100)
analysis_callback2 = AnalysisCallback(semi2, interval = 100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback1, analysis_callback2)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.8)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = 0.8,
                                      semi_indices = Vector([1]))

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback,
                        #                         analysis_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 0.0001, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
