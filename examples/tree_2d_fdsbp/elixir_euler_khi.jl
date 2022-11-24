# TODO: FD
# !!! warning "Experimental feature"
#     This is an experimental feature and may change in any future releases.
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation
equations = CompressibleEulerEquations2D(1.4)

function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
  amplitude = 0.02
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end

initial_condition = initial_condition_kelvin_helmholtz_instability

D_plus  = derivative_operator(SummationByPartsOperators.Mattsson2017(:plus),
#D_plus  = derivative_operator(SummationByPartsOperators.WIP(:plus),
                              derivative_order=1,
                              #accuracy_order=1,
                              accuracy_order=7,
                              xmin=-1.0, xmax=1.0,
                              N=64)
D_minus = derivative_operator(SummationByPartsOperators.Mattsson2017(:minus),
#D_minus = derivative_operator(SummationByPartsOperators.WIP(:minus),
                              derivative_order=1,
                              #accuracy_order=1,
                              accuracy_order=7,
                              xmin=-1.0, xmax=1.0,
                              N=64)
# TODO: Super hacky.
# Abuse the mortars to save the second derivative operator and get it into the run
flux_splitting = steger_warming_splitting
#flux_splitting = vanleer_splitting
#flux_splitting = lax_friedrichs_splitting
surface_flux = flux_hllc
solver = DG(D_plus, D_minus #= mortar =#,
            SurfaceIntegralUpwind(flux_splitting),
            #SurfaceIntegralStrongForm(surface_flux),
            VolumeIntegralUpwind(flux_splitting))

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=30_000,
                periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.7)
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

sol = solve(ode, SSPRK43(), abstol=1.0e-6, reltol=1.0e-6, dt=1e-3,
            save_everystep=false, callback=callbacks)
summary_callback()
