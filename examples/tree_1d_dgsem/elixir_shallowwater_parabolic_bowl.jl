
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant=9.81)
cfl = 0.3


function parabolic_bowl_analytic_1D_H(gravity, x,t)
  a = 1
  h_0 = 0.1
  σ = 0.5
  ω = sqrt(2*gravity*h_0)/a

  H = σ * h_0/a^2 * (2*x[1]*cos(ω*t) - σ) + h_0
  return H
end

# Implemented based on Wintermeyer (here seen in 2D)
function initial_condition_parabolic_bowl(x, t, equations:: ShallowWaterEquations1D)
  a = 1
  h_0 = 0.1
  σ = 0.5
  ω = sqrt(2*equations.gravity*h_0)/a

  v = -σ*ω*sin(ω*t)

  b = h_0 * x[1]^2/a^2

  H = max(b, parabolic_bowl_analytic_1D_H(equations.gravity, x, t))

  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_parabolic_bowl

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_hll_cn, hydrostatic_reconstruction_chen_noelle),
                flux_nonconservative_chen_noelle)

basis = LobattoLegendreBasis(5)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)
            
###############################################################################
# Create the TreeMesh for the domain [-2, 2]

coordinates_min = -2.0
coordinates_max = 2.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=10,
                n_cells_max=10_000
                )

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                    extra_analysis_integrals=(energy_kinetic,
                                                              energy_internal,
                                                              lake_at_rest_error))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=cfl)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(equations.threshold_limiter,),
                                                     variables=(Trixi.waterheight,))

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(stage_limiter!), dt=1.0,
            save_everystep=false, callback=callbacks, adaptive=false, maxiters=1e+9);

summary_callback() # print the timer summary