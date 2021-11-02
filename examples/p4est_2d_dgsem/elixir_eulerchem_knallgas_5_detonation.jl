
using OrdinaryDiffEq
using Trixi
using KROME

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerMultichemistryEquations2D(gammas             = (1.4, 1.4, 1.4, 1.4, 1.4),
                                                       gas_constants      = (4.1242, 0.2598, 0.4, 0.4615, 0.2968),
                                                       heat_of_formations = (0.0, 0.0, -50.0, -100.0, 0.0))

function initial_condition_knallgas_5_detonation(x, t, equations::CompressibleEulerMultichemistryEquations2D)
  
  if x[1] <= 0.5  
    u     = 10.0
    v     = 0.0
    p     = 40.0  # T = 10
    H2    = 0.0
    O2    = 0.0
    OH    = 0.17 * 2.0
    H2O   = 0.63 * 2.0 
    N2    = 0.2  * 2.0
  elseif x[1] > 0.5 && x[2] >= 1.2  
    u     = 0.0
    v     = 0.0
    p     = 1.0   # T = 1
    H2    = 0.0
    O2    = 0.0
    OH    = 0.17
    H2O   = 0.63
    N2    = 0.2
  elseif x[1] > 0.5 && x[2] < 1.2 
    u     = 0.0
    v     = 0.0
    p     = 1.0   # T = 1
    H2    = 0.08
    O2    = 0.72
    OH    = 0.0
    H2O   = 0.0
    N2    = 0.2
  end
  
  rho1  = H2
  rho2  = O2
  rho3  = OH 
  rho4  = H2O
  rho5  = N2
  
  prim_rho    = SVector{5, real(equations)}(rho1, rho2, rho3, rho4, rho5)
  
  prim_other  = SVector{3, real(equations)}(u, v, p)
  
  return prim2cons(vcat(prim_other, prim_rho), equations)
end

initial_condition = initial_condition_knallgas_5_detonation
# Example 5 from Wang et. al (2019) https://doi.org/10.1016/j.combustflame.2019.03.034

function boundary_condition_knallgas_5(u_inner, direction, x, t,
                                         surface_flux_function,
                                         equations::CompressibleEulerMultichemistryEquations2D)

  u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5], u_inner[6], u_inner[7], u_inner[8])
  flux = surface_flux_function(u_inner, u_boundary, direction, equations)

  return flux
end

boundary_condition = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(
                        :x_neg => boundary_condition,
                        :x_pos => boundary_condition,
                        :y_neg => boundary_condition_knallgas_5,
                        :y_pos => boundary_condition_knallgas_5)           

chemistry_term = chemistry_knallgas_5_detonation

surface_flux = flux_lax_friedrichs#
volume_flux  = flux_central
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.0,
                                         alpha_smooth = true,
                                         variable=pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = ( 0.0, 0.0)
coordinates_max = ( 6.0, 2.0)   # eigentlich (6, 2)


trees_per_dimension = (6, 2)

mesh = P4estMesh(trees_per_dimension,
                 polydeg=3, initial_refinement_level=7, #8
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, 
                                    boundary_conditions=boundary_conditions, source_terms=nothing,
                                    chemistry_terms=chemistry_term)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10000,
                                     save_initial_solution=false,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

chemistry_callback = KROMEChemistryCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback, 
                        alive_callback, 
                        save_solution,
                        stepsize_callback,
                        chemistry_callback)

###############################################################################
# run the simulation

limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                               variables=(Trixi.density, pressure))
stage_limiter! = limiter!

sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, williamson_condition=false), # ohne step_limiter!
#sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),   #stage_limiter!, step_limiter!, 
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks, maxiters=1e5);
summary_callback() # print the timer summary