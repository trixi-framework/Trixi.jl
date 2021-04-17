
using Random: seed!
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5/3
equations = CompressibleEulerEquations2D(gamma)

seed!(0)
function initial_condition_khi3(x, t, equations::CompressibleEulerEquations2D)
  # from https://arxiv.org/pdf/2001.01927.pdf
  # typical resolution 128^2, 256^2
  # domain size is [0,1]^2
  L = 0.025
  U1 = 0.5
  U2 =-0.5
  Um = 0.5 * (U1-U2)
  rho1 = 1.0
  rho2 = 2.0
  rhom = 0.5 * (rho1 - rho2)

  v2 = 0.01 * sin(4 * pi * x[1])
  p = 2.5

  if (0 <= x[2] <= 0.25)
    rho = rho1 - rhom * exp((x[2] - 0.25) / L)
    v1  = U1 - Um * exp((x[2] - 0.25) / L)
  elseif (0.25 < x[2] <= 0.5)
    rho = rho2 + rhom * exp((-x[2] + 0.25) / L)
    v1  = U2 + Um * exp((-x[2] + 0.25) / L)
  elseif (0.5 < x[2] <= 0.75)
    rho = rho2 + rhom * exp((x[2] - 0.75) / L)
    v1  = U2 + Um * exp((x[2] - 0.75) / L)
  elseif (0.75 < x[2] <= 1.0)
    rho = rho1 - rhom * exp((-x[2] + 0.75) / L)
    v1  = U1 - Um * exp((-x[2] + 0.75) / L)
  else
    println("shyte", x[2])
  end
  if (rho < 0.0)
   println("density", x[2], rho)
  end
  if (p < 0.0)
   println("pressure")
  end
  return prim2cons(SVector(rho, v1, v2, p), equations)
end



initial_condition = initial_condition_khi3

#surface_flux = flux_lax_friedrichs
surface_flux = flux_hllc
#surface_flux = flux_hll
#surface_flux = flux_shima_etal
volume_flux  = flux_ranocha
#volume_flux  = flux_shima_etal
#volume_flux  = flux_central
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
#indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                        alpha_max=0.002,
#                                        alpha_min=0.0001,
#                                        alpha_smooth=true,
#                                        variable=density_pressure)
#volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                volume_flux_dg=volume_flux,
#                                                volume_flux_fv=surface_flux)

volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
#volume_integral = VolumeIntegralWeakForm()
solver = DGSEM(basis, surface_flux, volume_integral)

#surface_flux  = flux_hllc
#volume_flux  = flux_ranocha
#solver = DGSEM(7, surface_flux, VolumeIntegralLocalComparison(VolumeIntegralFluxDifferencing(volume_flux)))

#surface_flux  = flux_hllc
#volume_flux = flux_ranocha
#volume_integral = VolumeIntegralFluxComparison(flux_central, volume_flux)
#solver = DGSEM(15, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = ( 1.0,  1.0)

#restart_filename = joinpath("out", "solution_010700.h5")
#mesh = load_mesh(restart_filename, n_cells_max=800_000)

println("start mesh")
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=7,
                n_cells_max=4_000_000)
println("end mesh")
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.7)
#tspan = (load_time(restart_filename), 3.7)
ode = semidiscretize(semi, tspan)
#ode = semidiscretize(semi, tspan, restart_filename)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_errors=(:conservation_error,),
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

save_restart = SaveRestartCallback(interval=1000)

stepsize_callback = StepsizeCallback(cfl=0.8)

#amr_indicator = IndicatorLöhner(semi,
#                                variable=Trixi.density)
#amr_controller = ControllerThreeLevel(semi, amr_indicator,
#                                      base_level=1,
#                                      med_level =0, med_threshold=0.1, # med_level = current level
#                                      max_level =6, max_threshold=0.5)
#amr_callback = AMRCallback(semi, amr_controller,
#                           interval=3,
#                           adapt_initial_condition=false,
#                           adapt_initial_condition_only_refine=true)

limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-5, 5.0e-5),
                                               variables=(Trixi.density, pressure))
stage_limiter! = limiter!
step_limiter!  = limiter!

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        save_restart,
#                        amr_callback, 
#			stepsize_callback
			)


###############################################################################
# run the simulation

#sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#sol = solve(ode, SSPRK33(stage_limiter!, step_limiter!),
sol = solve(ode, SSPRK43(stage_limiter!, step_limiter!),
#sol = solve(ode, SSPRK33(),
            #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
