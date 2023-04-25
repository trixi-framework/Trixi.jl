
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations


equations = ShallowWaterEquations2D(gravity_constant=9.812)

"""
    initial_condition_well_balanced_chen_noelle(x, t, equations:: ShallowWaterEquations2D)

Initial condition with a complex (discontinuous) bottom topography to test the well-balanced
property for the [`hydrostatic_reconstruction_chen_noelle`](@ref) including dry areas within the 
domain. The errors from the analysis callback are not important but the error for this 
lake-at-rest test case `∑|H0-(h+b)|` should be around machine roundoff.
The initial condition was found in the section 5.2 of the paper:
- Guoxian Chen and Sebastian Noelle (2017) 
  A new hydrostatic reconstruction scheme based on subcell reconstructions
  [DOI:10.1137/15M1053074](https://dx.doi.org/10.1137/15M1053074)
"""
function initial_condition_complex_bottom_well_balanced(x, t, equations:: ShallowWaterEquations2D)
  v1 = 0
  v2 = 0
  b = sin(4 * pi * x[1]) + 3

  if x[1] >= 0.5
    b = sin(4 * pi * x[1]) + 1
  end

  H = max(b, 2.5)

  if x[1] >= 0.5
    H = max(b, 1.5)
  end

  # It is mandatory to shift the water level at dry areas to make sure the water height h
  # stays positive. The system would not be stable for h set to a hard 0 due to division by h in 
  # the computation of velocity, e.g., (h v1) / h. Therefore, a small dry state threshold
  # (1e-13 per default, set in the constructor for the ShallowWaterEquations) is added if h = 0. 
  # This default value can be changed within the constructor call depending on the simulation setup.
  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_complex_bottom_well_balanced

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)

surface_flux = (FluxHydrostaticReconstruction(flux_hll_chen_noelle, hydrostatic_reconstruction_chen_noelle),
                flux_nonconservative_chen_noelle)

basis = LobattoLegendreBasis(3)

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
# Create the StructuredMesh for the domain [0, 1]^2

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

cells_per_dimension = (16, 16)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)


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

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution, stepsize_callback)

stage_limiter! = PositivityPreservingLimiterShallowWater(thresholds=(equations.threshold_limiter,),
                                                         variables=(Trixi.waterheight,))

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(stage_limiter!), dt=1.0,
            save_everystep=false, callback=callbacks, adaptive=false);

summary_callback() # print the timer summary