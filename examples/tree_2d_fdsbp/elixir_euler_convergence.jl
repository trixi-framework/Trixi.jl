# TODO: FD
# !!! warning "Experimental feature"
#     This is an experimental feature and may change in any future releases.

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation
equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

D_plus  = derivative_operator(SummationByPartsOperators.Mattsson2017(:plus),
                              derivative_order=1,
                              accuracy_order=4,
                              xmin=-1.0, xmax=1.0,
                              N=16)
D_minus = derivative_operator(SummationByPartsOperators.Mattsson2017(:minus),
                              derivative_order=1,
                              accuracy_order=4,
                              xmin=-1.0, xmax=1.0,
                              N=16)

# TODO: Super hacky.
# Abuse the mortars to save the second derivative operator and get it into the run
flux_splitting = steger_warming_splitting
solver = DG(D_plus, D_minus #= mortar =#,
            SurfaceIntegralUpwind(flux_splitting),
            VolumeIntegralUpwind(flux_splitting))

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=30_000,
                periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(energy_total,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        save_solution,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(), abstol=1.0e-9, reltol=1.0e-9,
            save_everystep=false, callback=callbacks)
summary_callback()
