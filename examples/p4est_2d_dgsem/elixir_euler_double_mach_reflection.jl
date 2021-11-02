
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

The Sedov blast wave setup based on Flash
- http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
"""
function initial_condition_double_mach_reflection(x, t, equations::CompressibleEulerEquations2D)
  
  x0  = 1/6

  # |v1| + |v2|     = 11.269
  # sqrt(1/3) * 20  = 11.547
  # 

  if x[1] == 0
    rho = 8.0
    v1  = 8.25  * cosd(30)
    v2  = -8.25 * sind(30) 
    p   = 116.5
  elseif x[1] == 4
    rho = 1.4
    v1  = 0.0
    v2  = 0.0
    p   = 1.0
  else 
    if x[1] < (x0 + sqrt(1/3)*(x[2]+20*t))
      rho = 8.0
      v1  = 8.25  * cosd(30)
      v2  = -8.25 * sind(30) 
      p   = 116.5
    else
      rho = 1.4
      v1  = 0.0
      v2  = 0.0
      p   = 1.0
    end
  end 

  return prim2cons(SVector(rho, v1, v2, p), equations)
end

initial_condition = initial_condition_double_mach_reflection

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(
                        :x_neg => boundary_condition,#_slip_wall, (mach 10 inflow)
                        :x_pos => boundary_condition,#_supersonic_outflow,#_slip_wall, (supersonic outflow)
                        :y_neg => boundary_condition_dmr,#_slip_wall,
                        :y_pos => boundary_condition)      

# Get the DG approximation space
surface_flux = flux_lax_friedrichs#FluxRotated(flux_hllc)
volume_flux = flux_central#ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)

###############################################################################

coordinates_min = (0.0, 0.0)
coordinates_max = (4.0, 1.0)

#mesh = StructuredMesh((16, 4), coordinates_min, coordinates_max, periodicity=true)

trees_per_dimension = (4, 1)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=3, initial_refinement_level=4,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max, 
                 periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)

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

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
