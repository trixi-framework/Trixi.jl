# Manufactured solution of the compressible Euler equations on a curved mesh with the
# continuous Galerkin spectral element method (CGSEM), source terms, and weakly imposed
# Dirichlet boundary conditions, cf. examples/p4est_2d_dgsem/elixir_euler_source_terms.jl

using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

# The surface flux enters only through the weakly imposed boundary conditions
solver = CGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Curved mapping of [-1, 1]^2 onto [0, 2]^2 with a warped interior
function mapping(xi, eta)
    x = 1 + xi + 0.1 * sin(pi * xi) * sin(pi * eta)
    y = 1 + eta + 0.1 * sin(pi * xi) * sin(pi * eta)
    return SVector(x, y)
end

trees_per_dimension = (4, 4)

# Create a conforming, curved P4estMesh with 4 x 4 trees and 8 x 8 elements
mesh = P4estMesh(trees_per_dimension, polydeg = 3, mapping = mapping,
                 initial_refinement_level = 1, periodicity = false)

# Dirichlet boundary conditions with the exact solution on all boundaries
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms = source_terms_convergence_test,
                                    boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
