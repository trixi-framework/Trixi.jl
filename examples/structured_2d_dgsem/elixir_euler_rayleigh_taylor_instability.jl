
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

initial_condition = initial_condition_rti
source_terms = source_terms_rti

surface_flux = flux_hll
volume_flux  = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# The domain is [0, .25] x [0, 1]
mapping(xi, eta) = SVector(.25 * .5 * (1 + xi), .5 * (1 + eta))

num_elements_per_dimension = 32
cells_per_dimension = (num_elements_per_dimension, num_elements_per_dimension * 4)
mesh = StructuredMesh(cells_per_dimension, mapping)

boundary_condition_wall = BoundaryConditionWall(boundary_state_slip_wall)
boundary_conditions = (
                       x_neg=boundary_condition_wall,
                       x_pos=boundary_condition_wall,
                       y_neg=boundary_condition_wall,
                       y_pos=boundary_condition_wall,
                      )

## Alternative setup: left/right periodic BCs and Dirichlet BCs on the top/bottom.
# boundary_conditions = (
#                        x_neg=boundary_condition_periodic,
#                        x_pos=boundary_condition_periodic,
#                        y_neg=BoundaryConditionDirichlet(initial_condition),
#                        y_pos=BoundaryConditionDirichlet(initial_condition),
#                       )

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
