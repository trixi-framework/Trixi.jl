
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleMoistEulerEquations2D()


function moist_state(y, dz, y0, r_t0, theta_e0, equations::CompressibleMoistEulerEquations2D)
  @unpack p_0, g, c_pd, c_pv, c_vd, c_vv, R_d, R_v, c_pl, L_00 = equations
  (p, rho, T, r_t, r_v, rho_qv, theta_e) = y
  p0 = y0[1]

  F = zeros(7,1)
  rho_d = rho / (1 + r_t)
  p_d = R_d * rho_d * T
  T_C = T - 273.15
  p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
  L = L_00 - (c_pl - c_pv) * T

  F[1] = (p - p0) / dz + g * rho 
  F[2] = p - (R_d * rho_d + R_v * rho_qv) * T
  # H = 1 is assumed
  F[3] = (theta_e - T * (p_d / p_0)^(-R_d / (c_pd + c_pl * r_t)) *
         exp(L * r_v / ((c_pd + c_pl * r_t) * T)))
  F[4] = r_t - r_t0
  F[5] = rho_qv - rho_d * r_v
  F[6] = theta_e - theta_e0
  a = p_vs / (R_v * T) - rho_qv
  b = rho - rho_qv - rho_d
  F[7] = a+b-sqrt(a*a+b*b)

  return F
end


function generate_function_of_y(dz, y0, r_t0, theta_e0, equations::CompressibleMoistEulerEquations2D)
  function function_of_y(y)
    return moist_state(y, dz, y0, r_t0, theta_e0, equations)
  end
end

struct AtmossphereLayers{RealT<:Real}
  equations::CompressibleMoistEulerEquations2D
  # structure:  1--> i-layer (z = total_hight/precision *(i-1)),  2--> rho, rho_theta, rho_qv, rho_ql
  LayerData::Matrix{RealT}
  total_hight::RealT
  preciseness::Int
  layers::Int
  ground_state::NTuple{2, RealT}
  equivalentpotential_temperature::RealT
  mixing_ratios::NTuple{2, RealT}
end


function AtmossphereLayers(equations ; total_hight=10000.0, preciseness=10, ground_state=(1.4, 100000.0), equivalentpotential_temperature=320, mixing_ratios=(0.02, 0.02), RealT=Float64)
  @unpack kappa, p_0, c_pd, c_vd, c_pv, c_vv, R_d, R_v, c_pl = equations
  rho0, p0 = ground_state
  r_t0, r_v0 = mixing_ratios
  theta_e0 = equivalentpotential_temperature

  rho_qv0 = rho0 * r_v0
  T0 = theta_e0
  y0 = [p0, rho0, T0, r_t0, r_v0, rho_qv0, theta_e0]

  n = convert(Int, total_hight / preciseness)
  dz = 0.01
  LayerData = zeros(RealT, n+1, 4)

  F = generate_function_of_y(dz, y0, r_t0, theta_e0, equations)
  sol = nlsolve(F, y0)
  p, rho, T, r_t, r_v, rho_qv, theta_e = sol.zero
  
  rho_d = rho / (1 + r_t)
  rho_ql = rho - rho_d - rho_qv
  kappa_M=(R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
  rho_theta = rho * (p0 / p)^kappa_M * T * (1 + (R_v / R_d) *r_v) / (1 + r_t) 

  LayerData[1, :] = [rho, rho_theta, rho_qv, rho_ql]
   for i in (1:n)
    y0 = deepcopy(sol.zero)
    dz = preciseness
    F = generate_function_of_y(dz, y0, r_t0, theta_e0, equations)
    sol = nlsolve(F, y0)
    p, rho, T, r_t, r_v, rho_qv, theta_e = sol.zero
    
    rho_d = rho / (1 + r_t)
    rho_ql = rho - rho_d - rho_qv
    kappa_M=(R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
    rho_theta = rho * (p0 / p)^kappa_M * T * (1 + (R_v / R_d) *r_v) / (1 + r_t) 

    LayerData[i+1, :] = [rho, rho_theta, rho_qv, rho_ql]
   end
  
  return AtmossphereLayers{RealT}(equations, LayerData, total_hight, dz, n, ground_state, theta_e0, mixing_ratios)
end


AtmossphereData = AtmossphereLayers(equations)

function initial_condition_moist(x, t, equations)
  return initial_condition_moist_bubble(x, t, equations, AtmossphereData)
end

initial_condition = initial_condition_moist

boundary_conditions = Dict(
  :y_neg => boundary_condition_slip_wall,
  :y_pos => boundary_condition_slip_wall
)

source_term = source_terms_moist_bubble

###############################################################################
# Get the DG approximation space

polydeg = 4
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_LMARS
volume_flux = flux_ranocha

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

#volume_integral=VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-5000.0, 0.0)
coordinates_max = (5000.0, 10000.0)

trees_per_dimension = (32, 32)

mesh = P4estMesh(trees_per_dimension, polydeg=polydeg,
                 periodicity=(true, false),
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions,
                                    source_terms=source_term)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1000.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
solution_variables = cons2aeqpot

analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=solution_variables)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=velocity),
                                      base_level=0, max_level=2,
                                      med_threshold=0.2, max_threshold=1.5)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=false,
                           adapt_initial_condition_only_refine=false)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        stepsize_callback)

#limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
#                                               variables=(Trixi.density, pressure))
#stage_limiter! = limiter!
#step_limiter!  = limiter!

###############################################################################
# run the simulation
#CarpenterKennedy2N54(stage_limiter!, step_limiter!, williamson_condition=false)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
