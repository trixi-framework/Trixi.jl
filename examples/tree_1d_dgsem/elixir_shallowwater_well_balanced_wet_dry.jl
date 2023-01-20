
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant=9.812)
cfl = .3

function initial_condition_well_balanced_CN(x, t, equations:: ShallowWaterEquations1D)
  v = 0
  b = sin(4*pi*x[1]) + 3

  if x[1] >= 0.5
    b = sin(4*pi*x[1]) + 1
  end

  H = max(b, 2.5)

  if x[1] >= 0.5
    H = max(b, 1.5)
  end

  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_well_balanced_CN

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
# Create the TreeMesh for the domain [0, 1]

coordinates_min = 0.
coordinates_max = 1.

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000,
                )

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 25.0)
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
            save_everystep=false, callback=callbacks, adaptive=false);

summary_callback() # print the timer summary