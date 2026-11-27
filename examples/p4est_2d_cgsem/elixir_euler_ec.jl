# Entropy-conservative discretization of the compressible Euler equations on a periodic
# curved mesh with the continuous Galerkin spectral element method (CGSEM) and flux
# differencing, cf. examples/p4est_2d_dgsem/elixir_euler_shockcapturing_ec.jl

using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_weak_blast_wave

# Entropy-conservative two-point flux in the volume integral. The surface flux enters
# only through weakly imposed boundary conditions, so it is unused on this periodic mesh.
volume_flux = flux_ranocha
solver = CGSEM(polydeg = 3, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# Curved mapping of [-1, 1]^2 onto itself that is the identity on the boundary and hence
# compatible with periodic boundaries
function mapping(xi, eta)
    x = xi + 0.1 * sin(pi * xi) * sin(pi * eta)
    y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
    return SVector(x, y)
end

trees_per_dimension = (4, 4)

# Create a conforming, periodic P4estMesh with 4 x 4 trees and 8 x 8 elements
mesh = P4estMesh(trees_per_dimension, polydeg = 3, mapping = mapping,
                 initial_refinement_level = 1, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,),
                                     extra_analysis_integrals = (entropy,))

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
