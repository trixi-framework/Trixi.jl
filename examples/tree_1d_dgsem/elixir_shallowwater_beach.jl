
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant=9.812)

"""
    initial_condition_beach(x, t, equations:: ShallowWaterEquations1D)
Initial condition to simulate a wave running towards a beach and crashing. Difficult test
including both wetting and drying in the domain using slip wall boundary conditions.
The water height and speed function used here, are an adaption of the initial condition 
found in section 5.2 of the paper:
  - Andreas Bollermann, Sebastian Noelle, Maria Lukáčová-Medvid’ová (2011)
    Finite volume evolution Galerkin methods for the shallow water equations with dry beds\n
    [DOI: 10.4208/cicp.220210.020710a](https://dx.doi.org/10.4208/cicp.220210.020710a)
The used bottom topography differs from the one out of the paper to be differentiable
"""
function initial_condition_beach(x, t, equations:: ShallowWaterEquations1D)
  D = 1
  delta = 0.02
  gamma = sqrt((3 * delta) / (4 * D))
  x_a = sqrt((4 * D) / (3 * delta)) * acosh(sqrt(20))

  f = D + 40 * delta * sech(gamma * (8 * x[1] - x_a))^2

  # steep wall
  b = 0.01 + 99/409600 * 4^x[1]

  if x[1] >= 6
    H = b
    v = 0.0
  else
    H = f
    v = sqrt(equations.gravity/D) * H
  end

  # It is mandatory to shift the water level at dry areas to make sure the water height h
  # stays positive. The system would not be stable for h set to a hard 0 due to division by h in 
  # the computation of velocity, e.g., (h v) / h. Therefore, a small dry state threshold
  # (1e-13 per default, set in the constructor for the ShallowWaterEquations) is added if h = 0. 
  # This default value can be changed within the constructor call depending on the simulation setup.
  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_beach
boundary_condition = boundary_condition_slip_wall

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
# Create the TreeMesh for the domain [0, 8]

coordinates_min = 0.0
coordinates_max = 8.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=7,
                n_cells_max=10_000,
                periodicity=false)

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

save_solution = SaveSolutionCallback(interval=250,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

stage_limiter! = PositivityPreservingLimiterShallowWater(thresholds=(equations.threshold_limiter,),
                                                     variables=(Trixi.waterheight,))

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(stage_limiter!), dt=1.0,
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary