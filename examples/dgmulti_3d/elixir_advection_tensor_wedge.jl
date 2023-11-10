using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
equations = LinearScalarAdvectionEquation3D(1.0, 1.0, 1.0)

initial_condition = initial_condition_convergence_test

# Define the polynomial degrees for the polynoms of the triangular base and the line
# of the tensor-prism
tensor_polydeg = (3, 4)

dg = DGMulti(element_type = Wedge(),
             approximation_type = Polynomial(),
             surface_flux = flux_lax_friedrichs,
             polydeg = tensor_polydeg)

cells_per_dimension = (8, 8, 8)
mesh = DGMultiMesh(dg,
                   cells_per_dimension,
                   coordinates_min = (-1.0, -1.0, -1.0),
                   coordinates_max = (1.0, 1.0, 1.0),
                   periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false), dt = 1.0,
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
