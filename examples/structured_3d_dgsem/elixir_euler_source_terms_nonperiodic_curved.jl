
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_convergence_test

# you can either use a single function to impose the BCs weakly in all
# 2*ndims == 4 directions or you can pass a tuple containing BCs for each direction
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (x_neg = boundary_condition,
                       x_pos = boundary_condition,
                       y_neg = boundary_condition,
                       y_pos = boundary_condition,
                       z_neg = boundary_condition,
                       z_pos = boundary_condition)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralWeakForm())

# Mapping as described in https://arxiv.org/abs/2012.12040 but with less warping.
function mapping(xi, eta, zeta)
    # Don't transform input variables between -1 and 1 onto [0,3] to obtain curved boundaries
    # xi = 1.5 * xi_ + 1.5
    # eta = 1.5 * eta_ + 1.5
    # zeta = 1.5 * zeta_ + 1.5

    y = eta +
        1 / 6 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
         cos(0.5 * pi * (2 * eta - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    x = xi +
        1 / 6 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
         cos(2 * pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    z = zeta +
        1 / 6 * (cos(0.5 * pi * (2 * x - 3) / 3) *
         cos(pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    # Transform the weird deformed cube to be approximately the cube [0,2]^3
    return SVector(x + 1, y + 1, z + 1)
end

cells_per_dimension = (4, 4, 4)
mesh = StructuredMesh(cells_per_dimension, mapping, periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
