
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

function initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)
  A  = 1.0 # magnitude of speed
  Ms = 0.1 # maximum Mach number

  rho = 1.0
  v1  =  A * sin(x[1]) * cos(x[2]) * cos(x[3])
  v2  = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
  v3  = 0.0
  p   = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
  p   = p + 1.0/16.0 * A^2 * rho * (cos(2*x[1])*cos(2*x[3]) + 2*cos(2*x[2]) + 2*cos(2*x[1]) + cos(2*x[2])*cos(2*x[3]))

  return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_taylor_green_vortex

volume_flux  = flux_ranocha
solver = DGMulti(element_type = Hex(),
                 approximation_type = periodic_derivative_operator(
                   derivative_order=1, accuracy_order=4, xmin=0.0, xmax=1.0, N=128),
                 surface_flux = flux_lax_friedrichs,
                 volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh = DGMultiMesh(solver, coordinates_min=(-pi, -pi, -pi),
                           coordinates_max=( pi,  pi,  pi))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 25.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(solver))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(), abstol = 1.0e-7, reltol = 1.0e-7,
            save_everystep = false, callback = callbacks)

summary_callback() # print the timer summary
