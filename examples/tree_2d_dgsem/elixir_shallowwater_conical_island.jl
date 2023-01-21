
using OrdinaryDiffEq
using Trixi
using Plots
###############################################################################
# semidiscretization of the shallow water equations

equations = ShallowWaterEquations2D(gravity_constant=9.81, H0=1.4)

cfl = 0.1                                    

function initial_condition_conical_island(x, t, equations::ShallowWaterEquations2D)
  # Set the background values
  
  v1 = 0.0
  v2 = 0.0

  x1, x2 = x
  b = max(0.1, - sqrt(16*x1^2+16*x2^2) + 1)
  
  # use a logistic function to tranfer water height value smoothly
  L  = equations.H0    # maximum of function
  x0 = 0.3   # center point of function
  k  = -25.0 # sharpness of transfer
  
  H = max(b, L/(1.0 + exp(-k*(sqrt(x1^2+x2^2) - x0))))

  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_conical_island
# Works with Dirichlet boundary conditions as well
#boundary_conditions = BoundaryConditionDirichlet(initial_condition)
###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_hll_cn, hydrostatic_reconstruction_chen_noelle),
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
# Get the TreeMesh and setup a periodic mesh

coordinates_min = (-1, -1)
coordinates_max = (1,  1)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000
                #,periodicity=false
               )
            
# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
                                    #, boundary_conditions=boundary_conditions)

###############################################################################
# ODE solver

tspan = (0.0, 10.)
ode = semidiscretize(semi, tspan)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=cfl)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(equations.threshold_limiter,),
                                                     variables=(Trixi.waterheight,))

sol = solve(ode, SSPRK43(stage_limiter!),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks, adaptive=false);
summary_callback() # print the timer summary
