
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

epsilon_relaxation = 5.0e-6
a1 = a2 = a3 = a4 = 30.0
b1 = b2 = b3 = b4 = 30.0

equations_relaxation = CompressibleEulerEquations2D(1.4)
equations = JinXinCompressibleEulerEquations2D(epsilon_relaxation, a1, a2, a3, a4, b1, b2, b3, b4,equations_relaxation)

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

#initial_condition = initial_condition_constant
initial_condition = Trixi.InitialConditionJinXin(initial_condition_kelvin_helmholtz_instability)
#initial_condition = Trixi.InitialConditionJinXin(initial_condition_density_wave)
polydeg = 3
basis = LobattoLegendreBasis(polydeg; polydeg_projection = 2 * polydeg)
solver = DGSEM(basis, Trixi.flux_upwind)
#solver = DGSEM(polydeg = 3, surface_flux = Trixi.flux_upwind)

#surface_flux = Trixi.flux_upwind
#volume_flux  = flux_central
#basis = LobattoLegendreBasis(7)
#limiter_idp = SubcellLimiterIDP(equations, basis;
#                                positivity_variables_cons=[1],
#                                positivity_correction_factor=0.5)
#volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
#                                                volume_flux_dg=volume_flux,
#                                                volume_flux_fv=surface_flux)
#solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=1_000_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)#,source_terms=source_terms_JinXin_Relaxation)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.7)
#tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

collision_callback = LBMCollisionCallback()  

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,stepsize_callback)#,collision_callback)

###############################################################################
# run the simulation
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                                     variables=(Trixi.density, pressure))

#sol = solve(ode, CarpenterKennedy2N54(stage_limiter!,williamson_condition=false),
sol = solve(ode, SSPRK43(stage_limiter!),
#sol = solve(ode, SSPRK33(stage_limiter!),
#sol = solve(ode, RDPK3SpFSAL49(),
#sol = solve(ode, AutoTsit5(Rosenbrock23()),
            dt=1.0e-3, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks,maxiters=1e7);
summary_callback() # print the timer summary
