
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerMultichemistryEquations2D(gammas             = (1.4, 1.4),
                                                       gas_constants      = (0.287, 0.287),
                                                       heat_of_formations = (0.0, 0.0))#22.429))

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on Flash
- http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
"""
function initial_condition_double_mach_reflection_detonation(x, t, equations::CompressibleEulerMultichemistryEquations2D)
  
  x0  = 1/6

  # |v1| + |v2|     = 11.269
  # sqrt(1/3) * 20  = 11.547
  # Q = 31.4 * p/rho = 31.4 * 1.0/1.4 = 22.429
  # T_ign = 20/t0 => t0 = p/rho => 20*rho/p = 28

  if x[1] == 0
    rho_burnt = 8.0
    rho_unburnt = 0.0
    v1  = 8.25  * cosd(30)
    v2  = (-8.25) * sind(30) 
    p   = 116.5
  elseif x[1] == 4
    rho_burnt = 0.0
    rho_unburnt = 1.4
    v1  = 0.0
    v2  = 0.0
    p   = 1.0
  else 
    if x[1] < (x0 + sqrt(1/3)*(x[2] + 20*t))#11.667 * t))
      rho_burnt = 8.0
      rho_unburnt = 0.0
      v1  = 8.25  * cosd(30)
      v2  = (-8.25) * sind(30) 
      p   = 116.5
    else
      rho_burnt = 0.0
      rho_unburnt = 1.4
      v1  = 0.0
      v2  = 0.0
      p   = 1.0
    end
  end 

  return prim2cons(SVector(v1, v2, p, rho_burnt, rho_unburnt), equations)
end

initial_condition = initial_condition_double_mach_reflection_detonation

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(
                        :x_neg => boundary_condition,#_slip_wall, (mach 10 inflow)
                        :x_pos => boundary_condition,#_supersonic_outflow,#_slip_wall, (supersonic outflow)
                        :y_neg => boundary_condition_dmr,#_slip_wall,
                        :y_pos => boundary_condition)      

chemistry_term = chemistry_knallgas_2_naive                           

# Get the DG approximation space
surface_flux = FluxRotated(flux_hllc)#
volume_flux = flux_central
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.0,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)

###############################################################################

coordinates_min = (0.0, 0.0)
coordinates_max = (60.0, 15.0)

#mesh = StructuredMesh((16, 4), coordinates_min, coordinates_max, periodicity=true)

trees_per_dimension = (4, 1)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=3, initial_refinement_level=4,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max, 
                 periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions,
                                    source_terms=nothing, chemistry_terms=chemistry_term)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=0.3)

chemistry_callback = KROMEChemistryCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        chemistry_callback)

###############################################################################
# run the simulation
#limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
#                                               variables=(Trixi.density, pressure))
#stage_limiter! = limiter!
#step_limiter!  = limiter!

#sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, step_limiter!, williamson_condition=false),
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
