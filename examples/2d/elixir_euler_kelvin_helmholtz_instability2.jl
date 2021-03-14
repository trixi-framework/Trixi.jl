
using Random: seed!
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

seed!(0)
function initial_condition_khi2(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_khi2

#surface_flux  = flux_hllc
#basis = LobattoLegendreBasis(3)
#volume_integral = VolumeIntegralPureLGLFiniteVolume(basis; 
#                                                    volume_flux_fv = flux_hllc,
#                                                    reconstruction_mode = reconstruction_small_stencil,
#                                                    slope_limiter = minmod)
#solver = DGSEM(basis, surface_flux, volume_integral)

surface_flux  = flux_lax_friedrichs
#volume_flux  = flux_ranocha
volume_flux  = flux_chandrashekar
solver = DGSEM(3, surface_flux, VolumeIntegralLocalComparison(VolumeIntegralFluxDifferencing(volume_flux)))

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_errors=(:conservation_error,),
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=20,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback, 
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

#sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
sol = solve(ode, SSPRK33(),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
