
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant=9.812)
cfl = 0.2

function initial_condition_beach(x, t, equations:: ShallowWaterEquations1D)
  D = 1
  δ = 0.02
  γ = sqrt((3*δ)/(4*D))
  xₐ = sqrt((4*D)/(3*δ)) * acosh(sqrt(20))

  f = D + 40*δ * sech(γ*(8*x[1]-xₐ))^2

  # steep wall
  b = 0.01 + 99/409600 * 4^x[1]

  if x[1] >= 6
    H = b
    v = 0
  else
    H = f
    v = sqrt(equations.gravity/D) * H
  end

  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_beach
boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_hll_cn, hydrostatic_reconstruction_chen_noelle),
                flux_nonconservative_chen_noelle)

basis = LobattoLegendreBasis(7)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=.5,
                                         alpha_min=.001,
                                         alpha_smooth=true,
                                         variable=waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)
              
###############################################################################
# Create the TreeMesh for the domain [0, 8]

coordinates_min = 0.0
coordinates_max = 8.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=8,
                n_cells_max=10_000,
                periodicity=false
                )

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition)

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