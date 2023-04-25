
using OrdinaryDiffEq
using Trixi

 ###############################################################################
 # Semidiscretization of the shallow water equations

equations = ShallowWaterEquations2D(gravity_constant=9.81, H0=1.4)                        

"""
    initial_condition_conical_island(x, t, equations::ShallowWaterEquations2D)

Initial condition [`ShallowWaterEquations2D`](@ref) to test the [`hydrostatic_reconstruction_chen_noelle`](@ref)
and its handling of discontinuous water heights at the start in combination with wetting and
drying. The bottom topography is given by a conical island in the middle of the domain. Around that
island, there is a cylindrical water column at t=0 and the rest of the domain is dry. This
discontinuous water height is smoothed by a logistic function. This simulation uses periodic
boundary conditions.
"""
function initial_condition_conical_island(x, t, equations::ShallowWaterEquations2D)
  # Set the background values
  
  v1 = 0.0
  v2 = 0.0

  x1, x2 = x
  b = max(0.1, 1.0 - 4.0 * sqrt(x1^2 + x2^2))
  
  # use a logistic function to tranfer water height value smoothly
  L  = equations.H0    # maximum of function
  x0 = 0.3   # center point of function
  k  = -25.0 # sharpness of transfer
  
  H = max(b, L/(1.0 + exp(-k*(sqrt(x1^2+x2^2) - x0))))

  # It is mandatory to shift the water level at dry areas to make sure the water height h
  # stays positive. The system would not be stable for h set to a hard 0 due to division by h in 
  # the computation of velocity, e.g., (h v1) / h. Therefore, a small dry state threshold
  # (1e-13 per default, set in the constructor for the ShallowWaterEquations) is added if h = 0. 
  # This default value can be changed within the constructor call depending on the simulation setup.
  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_conical_island

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_hll_chen_noelle, hydrostatic_reconstruction_chen_noelle),
                flux_nonconservative_chen_noelle)

basis = LobattoLegendreBasis(4)

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
# Get the StructuredMesh and setup a periodic mesh

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0,  1.0)

cells_per_dimension = (16, 16)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solver

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

stage_limiter! = PositivityPreservingLimiterShallowWater(thresholds=(equations.threshold_limiter,),
                                                         variables=(Trixi.waterheight,))

sol = solve(ode, SSPRK43(stage_limiter!), dt=1.0,
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary