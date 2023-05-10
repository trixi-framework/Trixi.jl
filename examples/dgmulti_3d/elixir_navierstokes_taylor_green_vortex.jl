
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

# TODO: parabolic; unify names of these accessor functions
prandtl_number() = 0.72
mu() = 6.25e-4 # equivalent to Re = 1600

equations = CompressibleEulerEquations3D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu=mu(),
                                                          Prandtl=prandtl_number())

"""
    initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)

The classical inviscid Taylor-Green vortex.
"""
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

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
dg = DGMulti(polydeg = 3, element_type = Hex(), approximation_type = GaussSBP(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = ( 1.0,  1.0,  1.0) .* pi
cells_per_dimension = (8, 8, 8)
mesh = DGMultiMesh(dg, cells_per_dimension;
                   coordinates_min, coordinates_max,
                   periodicity=(true, true, true))

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, dg)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg),
                                     extra_analysis_integrals=(energy_kinetic,
                                                               energy_internal,
                                                               enstrophy))
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks)
summary_callback() # print the timer summary
