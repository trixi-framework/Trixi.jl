# Non-periodic boundary conditions (Dirichlet) on a curved P4estMesh, verifying
# that BlockFV evaluates boundary conditions at the exact physical face
# midpoint and outward normal (not at the offset FV cell center).

using OrdinaryDiffEqLowOrderRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

solver = BlockFV(n_nodes = 4, surface_flux = flux_hllc)

# Mapping that introduces a curved warping to interior nodes, as used e.g. in
# p4est_2d_dgsem/elixir_advection_diffusion_periodic_curved.jl
function mapping(xi, eta)
    x = xi + 0.1 * sin(pi * xi) * sin(pi * eta)
    y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
    return SVector(x, y)
end

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 initial_refinement_level = 1,
                 mapping = mapping,
                 periodicity = false)

# Assign a single boundary condition to all boundaries
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms = source_terms_convergence_test,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation
sol = solve(ode, Euler();
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);

