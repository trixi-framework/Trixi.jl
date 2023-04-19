using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg=4, element_type = Quad(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(FluxLaxFriedrichs()),
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations2D(1.4)

"""
A compressible version of the double shear layer initial condition. Adapted from
Brown and Minion (1995). See Section 3, equations (27)-(28) for the original
incompressible version.

- David L. Brown and Michael L. Minion (1995)
  Performance of Under-resolved Two-Dimensional Incompressible Flow Simulations.
  [DOI: 10.1006/jcph.1995.1205](https://doi.org/10.1006/jcph.1995.1205)
"""
function initial_condition_BM_vortex(x, t, equations::CompressibleEulerEquations2D)
  pbar = 9.0 / equations.gamma
  delta = 0.05
  epsilon = 30
  H = (x[2] < 0) ? tanh(epsilon * (x[2] + 0.25)) :  tanh(epsilon * (0.25 - x[2]))
  rho = 1.0
  v1 = H
  v2 = delta * cos(2.0 * pi * x[1])
  p = pbar
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_BM_vortex

cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min=(-0.5, -0.5), coordinates_max=(0.5, 0.5),
                   periodicity=true)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

tol = 1.0e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol=tol, reltol=tol,
            ode_default_options()..., callback=callbacks);

summary_callback() # print the timer summary
