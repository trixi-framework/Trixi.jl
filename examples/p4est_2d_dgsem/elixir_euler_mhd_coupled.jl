using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Coupled setup: one IdealGlmMhdEquations2D system (MHD) coupled to one
# CompressibleEulerEquations2D system (Euler) on two non-overlapping P4estMeshViews
# of the same parent mesh.  The coupling converts the shared primitive variables
# (density, momentum, pressure) across the interface; the MHD-specific fields
# (magnetic field, GLM divergence-cleaning ψ) are set to zero when entering Euler.

###############################################################################
# Initial conditions — a small-amplitude pressure wave, identical for both systems.

function initial_condition_mhd(x, t, equations::IdealGlmMhdEquations2D)
    rho = 1.0 + 0.01 * sin(x[1] * 2 * pi)
    v1 = 0.01 * sin((x[1] - 1 / 2) * 2 * pi)
    v2 = 0.0
    v3 = 0.0
    p = rho^equations.gamma
    B1 = 0.0
    B2 = 0.0
    B3 = 0.0
    psi = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

function initial_condition_euler(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.0 + 0.01 * sin(x[1] * 2 * pi)
    v1 = 0.01 * sin((x[1] - 1 / 2) * 2 * pi)
    v2 = 0.0
    p = rho^equations.gamma

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

###############################################################################
# Parent mesh

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
trees_per_dimension = (8, 8)
# Periodicity is set to false so that the physical boundaries become the coupling
# interfaces; the resulting setup is effectively periodic through the coupling.
parent_mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                        coordinates_min = coordinates_min,
                        coordinates_max = coordinates_max,
                        initial_refinement_level = 0, periodicity = (false, false))

###############################################################################
# Equations and coupling

equations1 = IdealGlmMhdEquations2D(5 / 3)
equations2 = CompressibleEulerEquations2D(5 / 3)

# Coupling converters for every (source_system, target_system) pair.
# [i, j] converts a state from system j into the variable space of system i.
coupling_functions = Array{Function}(undef, 2, 2)
coupling_functions[1, 1] = (x, u, equations_other, equations_own) -> u
coupling_functions[1, 2] = (x, u, equations_other, equations_own) -> SVector(u[1], u[2],
                                                                              u[3], 0.0,
                                                                              u[4], 0.0,
                                                                              0.0, 0.0,
                                                                              0.0)
coupling_functions[2, 1] = (x, u, equations_other, equations_own) -> SVector(u[1], u[2],
                                                                              u[3], u[5])
coupling_functions[2, 2] = (x, u, equations_other, equations_own) -> u

###############################################################################
# System 1: MHD

cell_ids1 = vcat(Vector(1:8), Vector(32:64))
mesh1 = P4estMeshView(parent_mesh, cell_ids1)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver1 = DGSEM(polydeg = 3,
                surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
                volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
# The split is non-rectangular, so every face name can appear at both a view interface
# and a physical domain edge.  A Dirichlet fallback handles the physical-domain faces.
fallback_bc1 = BoundaryConditionDirichlet(initial_condition_mhd)
coupled_bc1 = BoundaryConditionCoupledP4est(coupling_functions; fallback_bc = fallback_bc1)
boundary_conditions1 = (; x_neg = coupled_bc1, y_neg = coupled_bc1,
                        y_pos = coupled_bc1, x_pos = coupled_bc1)
semi1 = SemidiscretizationHyperbolic(mesh1, equations1, initial_condition_mhd, solver1,
                                     boundary_conditions = boundary_conditions1)

###############################################################################
# System 2: Euler

cell_ids2 = Vector(9:31)
mesh2 = P4estMeshView(parent_mesh, cell_ids2)
solver2 = DGSEM(polydeg = 3, surface_flux = flux_hll,
                volume_integral = VolumeIntegralWeakForm())
fallback_bc2 = BoundaryConditionDirichlet(initial_condition_euler)
coupled_bc2 = BoundaryConditionCoupledP4est(coupling_functions; fallback_bc = fallback_bc2)
boundary_conditions2 = (; x_neg = coupled_bc2, y_neg = coupled_bc2,
                        y_pos = coupled_bc2, x_pos = coupled_bc2)
semi2 = SemidiscretizationHyperbolic(mesh2, equations2, initial_condition_euler, solver2,
                                     boundary_conditions = boundary_conditions2)

semi = SemidiscretizationCoupledP4est(semi1, semi2; coupling_functions = coupling_functions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback1 = AnalysisCallback(semi1, interval = 100)
analysis_callback2 = AnalysisCallback(semi2, interval = 100)
analysis_callback = AnalysisCallbackCoupledP4est(semi, analysis_callback1,
                                                 analysis_callback2)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.8)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = 0.8,
                                      semi_indices = [1])

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 0.0001, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks)

# Print the timer summary
summary_callback()
