# Free-stream preservation of the compressible Euler equations on a curved mesh with the
# continuous Galerkin spectral element method (CGSEM), flux differencing, and weakly
# imposed Dirichlet boundary conditions,
# cf. examples/p4est_2d_dgsem/elixir_euler_free_stream.jl

using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant

# The surface flux enters only through the weakly imposed boundary conditions
solver = CGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

# Mapping as described in https://arxiv.org/abs/2012.12040
# (DOI: https://doi.org/10.1016/j.jcp.2021.110580) but reduced to 2D
function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5

    y = eta + 3 / 8 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
                       cos(0.5 * pi * (2 * eta - 3) / 3))

    x = xi + 3 / 8 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
                      cos(2 * pi * (2 * y - 3) / 3))

    return SVector(x, y)
end

trees_per_dimension = (4, 4)

# Create a conforming, curved P4estMesh with 4 x 4 trees and 8 x 8 elements
mesh = P4estMesh(trees_per_dimension, polydeg = 3, mapping = mapping,
                 initial_refinement_level = 1, periodicity = false)

# Dirichlet boundary conditions on all boundaries
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
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
