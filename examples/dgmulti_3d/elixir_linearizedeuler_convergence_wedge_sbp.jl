using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linearized Euler equations

equations = LinearizedEulerEquations3D(v_mean_global = (0.0, 0.0, 0.0),
                                       c_mean_global = 1.0,
                                       rho_mean_global = 1.0)

initial_condition = initial_condition_convergence_test

# Tensor-product wedge (triangular prism) elements with SBP operators on both the
# triangular base and the line in the extrusion direction.
# The polynomial degrees are given as `(polydeg_tri, polydeg_line)`.
polydeg = (3, 3)
dg = DGMulti(element_type = Wedge(),
             approximation_type = SBP(),
             surface_flux = flux_hll,
             polydeg = polydeg)

cells_per_dimension = (4, 4, 4)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (-1.0, -1.0, -1.0),
                   coordinates_max = (1.0, 1.0, 1.0),
                   periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
