# TODO: FD
# !!! warning "Experimental feature"
#     This is an experimental feature and may change in any future releases.

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the two-dimensional incompressible Euler equations

equations = IncompressibleEulerEquations2D()

#initial_condition = initial_condition_constant
initial_condition = initial_condition_pulse

D_SBP = derivative_operator(SummationByPartsOperators.MattssonNordström2004(),
                            derivative_order=1, accuracy_order=4,
                            xmin=0.0, xmax=1.0, N=100)
solver = DG(D_SBP, nothing #= mortar =#,
            SurfaceIntegralStrongForm(flux_lax_friedrichs),
            VolumeIntegralStrongForm())

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=0,
                n_cells_max=30_000,
                periodicity=false)
# FIXME: this is a hack. incompressible Euler is using Wall BCs but not
#        via the BoundaryConditionWall formalism in other parts of Trixi

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

#visualization = VisualizationCallback(interval=3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)
#                        visualization)


###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, callback=callbacks, maxiters=1e5)
summary_callback()
