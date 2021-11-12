
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerMultichemistryEquations2D(gammas             = (1.4, 1.4),
                                                       gas_constants      = (0.287, 0.287),
                                                       heat_of_formations = (0.0, 5.196e9))

function initial_condition_naive_chemistry(x, t, equations::CompressibleEulerMultichemistryEquations2D)
  
  # Example 4.5 from Zhang et. al (2014) http://dx.doi.org/10.1016/j.jcp.2013.12.043

  switch = 0.0
  if abs(x[2]-0.0025) >= 0.001
    switch = 0.004
  else 
    switch = 0.005 - abs(x[2]-0.0025)
  end 

  b     = (-8.321e5) - (1.201e-3*5.196e9*(1.4-1.0))
  c     = (8.321e5)^2 + 2*(1.4-1.0)*8.321e5*1.201e-3*5.196e9/(1.4+1.0)
  p_b   = (-b) + sqrt(b^2-c) 
  rho_b =  1.201e-3*(p_b*(1.4+1.0)-8.321e5)/(1.4 * p_b)

  if x[1] <= switch
    rho_burnt   = rho_b 
    rho_unburnt = 0.0
    v1          = 8.162e4
    v2          = 0.0
    p           = p_b
  else 
    rho_burnt   = 0.0
    rho_unburnt = 1.201e-3
    v1          = 0.0
    v2          = 0.0
    p           = 8.321e5
  end 

  return prim2cons(SVector(v1, v2, p, rho_burnt, rho_unburnt), equations)
end

function boundary_condition_naive_chemistry(u_inner, direction, x, t,
                                            surface_flux_function,
                                            equations::CompressibleEulerMultichemistryEquations2D)

  u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5])
  flux = surface_flux_function(u_inner, u_boundary, direction, equations)

  return flux
end

initial_condition = initial_condition_naive_chemistry

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(
                        :x_neg => boundary_condition,
                        :x_pos => boundary_condition,
                        :y_neg => boundary_condition_naive_chemistry,
                        :y_pos => boundary_condition_naive_chemistry)    

chemistry_term = chemistry_knallgas_2_naive                    

# Get the DG approximation space
surface_flux = flux_lax_friedrichs#FluxRotated(flux_hllc)#
volume_flux = flux_ranocha#flux_central #FluxRotated(flux_ranocha)#flux_central
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.0,
                                         alpha_smooth=true,
                                         variable=pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)

###############################################################################

coordinates_min = (0.0, 0.0)
coordinates_max = (0.015, 0.005)

trees_per_dimension = (3, 1)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=3, initial_refinement_level=5,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max, 
                 periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions,
                                    source_terms=nothing, chemistry_terms=chemistry_term)
 
###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0e-8)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10000,
                                     save_initial_solution=false,
                                     save_final_solution=true)

amr_indicator = IndicatorHennemannGassner(semi,
                                     alpha_max=1.0,
                                     alpha_min=0.0,
                                     alpha_smooth=true,
                                     variable=pressure)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                 base_level=5,
                                 max_level =7, max_threshold=0.01)
amr_callback = AMRCallback(semi, amr_controller,
                      interval=1,
                      adapt_initial_condition=true,
                      adapt_initial_condition_only_refine=true)                                     

stepsize_callback = StepsizeCallback(cfl=0.9)

chemistry_callback = KROMEChemistryCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        stepsize_callback,
                        chemistry_callback)

###############################################################################
# run the simulation

# Using Framework of ZhangShu Limiter.. not nice right now.
limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                               variables=(Trixi.density, pressure))
stage_limiter! = limiter!
step_limiter! = limiter!

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
