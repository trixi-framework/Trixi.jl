
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations1D)

The Sedov blast wave setup based on Flash
- http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
"""
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations1D)
  # Set up polar coordinates
  inicenter = SVector(0.0)
  x_norm = x[1] - inicenter[1]
  r = abs(x_norm)

  # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
  r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
  # r0 = 0.5 # = more reasonable setup
  E = 1.0
  p0_inner = 6 * (equations.gamma - 1) * E / (3 * pi * r0)
  p0_outer = 1.0e-5 # = true Sedov setup

  # Calculate primitive variables
  rho = 1.0
  v1  = 0.0
  p   = r > r0 ? p0_outer : p0_inner

  return prim2cons(SVector(rho, v1, p), equations)
end

initial_condition = initial_condition_sedov_blast_wave
surface_flux = flux_hll
volume_flux  = flux_ranocha
basis = LobattoLegendreBasis(3)
shock_indicator_variable = density_pressure
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=shock_indicator_variable)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

cells_per_dimension = (64,)
coordinates_min = (-2.0,)
coordinates_max = ( 2.0,)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max, periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 12.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback, 
                        analysis_callback, 
                        alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary