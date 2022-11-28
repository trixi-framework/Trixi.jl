
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers' equation

equations = InviscidBurgersEquation1D()

function initial_condition_linear_stability(x, t, equation::InviscidBurgersEquation1D)
  k = 1
  2 + sinpi(k * (x[1] - 0.7)) |> SVector
end

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
flux_splitting = lax_friedrichs_splitting
solver = DG(D_plus, D_minus #= mortar =#,
            SurfaceIntegralUpwind(flux_splitting),
            VolumeIntegralUpwind(flux_splitting))

coordinates_min = -1.0
coordinates_max =  1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_linear_stability, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l2_error_primitive,
                                                            :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(), abstol=1.0e-6, reltol=1.0e-6,
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
