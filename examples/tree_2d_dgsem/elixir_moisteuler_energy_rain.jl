
using OrdinaryDiffEq
using Trixi

function moist_state(y, dz, y0, H_0, theta_0, rho_ql0, equations::CompressibleMoistEulerEquations2D)
  @unpack p_0, g, c_pd, c_pv, c_vd, c_vv, R_d, R_v, c_pl = equations
  (p, rho, T, theta, H, rho_qv, rho_ql) = y
  p0 = y0[1]

  F = zeros(7,1)
  rho_d = rho - rho_ql - rho_qv
  r_v = rho_qv / rho_d
  r_l = rho_ql / rho_d
  p_d = R_d * rho_d * T
  p_v = r_v * (R_v / R_d) * p_d
  T_C = T - 273.15
  # Magnus-Formel für den Sättigungsdampfdruck
  p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
  H = p_v * inv(p_vs)
  R_m = R_d + r_v * R_v
  c_pml = c_pd + r_v * c_pv + r_l * c_pl
  kappa_m =  R_m * inv(c_pml)

  F[1] = (p - p0) / dz + g * rho 
  F[2] = p - (R_d * rho_d + R_v * rho_d * r_v) * T
  F[3] = theta - T * (p_0 / p)^(kappa_m)
  F[4] = H - H_0
  F[5] = theta - theta_0
  F[6] = rho_ql - rho_ql0
  a = p_vs / (R_v * T) - (rho_d * r_v)
  b = rho - (1 + r_v) * rho_d
  F[7] = a+b-sqrt(a*a+b*b)
  #F[7] = p - (p_d + p_v)

  return F
end

struct AtmossphereLayers{RealT<:Real}
  equations::CompressibleMoistEulerEquations2D
  # structure:  1--> i-layer (z = total_hight/precision *(i-1)),  2--> rho, rho_theta, rho_qv, rho_ql
  LayerData::Matrix{RealT}
  total_hight::RealT
  preciseness::Int
  layers::Int
  ground_state::NTuple{2, RealT}
  moistpotential_temperature::RealT
  H::RealT
  mixing_ratios::NTuple{2, RealT}
end

function AtmossphereLayers(equations ; total_hight=5120.0, preciseness=10, ground_state=(1.2, 100000.0), moistpotential_temperature=300, H = 0.5, mixing_ratios = (0.01, 0.0), RealT=Float64)
  @unpack kappa, p_0, c_pd, c_vd, c_pv, c_vv, R_d, R_v, c_pl = equations
  rho0, p0 = ground_state
  H_0 = H
  r_v0, r_l0 = mixing_ratios
  theta_0 = moistpotential_temperature

  T0 = theta_0
  rho_qv0, rho_ql0 = rho0 .* (r_v0, r_l0)
  y0 = [p0, rho0, T0, theta_0, H_0, rho_qv0, rho_ql0]

  n = convert(Int, total_hight / preciseness)
  dz = 0.01
  LayerData = zeros(RealT, n+1, 4)

  p, rho, T, theta, H, rho_qv, rho_ql = zeros(Float64, 7)
  sol = zeros(Float64, 7)
  F = generate_function_of_y(dz, y0, H_0, theta_0, rho_ql0, equations)
  sol = nlsolve(F, y0)
  p, rho, T, theta, H, rho_qv, rho_ql = sol.zero
  LayerData[1, :] = [rho, T, rho_qv, rho_ql]

   for i in (1:n)
    y0 = deepcopy(sol.zero)
    dz = preciseness
    F = generate_function_of_y(dz, y0, H_0, theta_0, r_l0, equations)
    sol = nlsolve(F, y0)
    p, rho, T, theta, H, rho_qv, rho_ql  = sol.zero
    LayerData[i+1, :] = [rho, T, rho_qv, rho_ql]
   end
  
  return AtmossphereLayers{RealT}(equations, LayerData, total_hight, dz, n, ground_state, theta_0, H_0, mixing_ratios)
end

function generate_function_of_y(dz, y0, H_0, theta_0, r_l0, equations::CompressibleMoistEulerEquations2D)
  function function_of_y(y)
    return moist_state(y, dz, y0, H_0, theta_0, r_l0, equations)
  end
end

function initial_condition_rain(x, t, equations::CompressibleMoistEulerEquations2D, AtmosphereLayers)
  @unpack LayerData, preciseness, total_hight = AtmosphereLayers
  @unpack p_0, c_pd, c_vd, c_pv, c_vv, R_d, R_v, c_pl, L_00 = equations
  dz = preciseness
  z = x[2] 
  if (z > total_hight && !(isapprox(z, total_hight)))
    error("The atmossphere does not match the simulation domain")
  end
  n = convert(Int, floor(z/dz)) + 1
  z_l = (n-1) * dz
  (rho_l, T_l, rho_qv_l, rho_ql_l) = LayerData[n, :]
  z_r = n * dz
  if (z_l == total_hight)
    z_r = z_l + dz 
    n = n-1
  end
  (rho_r, T_r, rho_qv_r, rho_ql_r) = LayerData[n+1, :]
  rho = (rho_r * (z - z_l) + rho_l * (z_r - z)) / dz
  T = (T_r * (z - z_l) + T_l * (z_r - z)) / dz 
  rho_qv = rho * (rho_qv_r / rho_r * (z - z_l) + rho_qv_l/ rho_l * (z_r - z)) / dz
  rho_ql = rho * (rho_ql_r / rho_r * (z - z_l) + rho_ql_l/ rho_l * (z_r - z)) / dz
  rho_d = rho - rho_qv - rho_ql
  rho_ql = 0

  rho_e = (c_vd * rho_d + c_vv * rho_qv + c_pl * rho_ql) * T + L_00 * rho_qv

  v1 = 0
  v2 = 0
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_E = rho_e + 1/2 * (rho_v1*v1 + rho_v2*v2)

  return SVector(rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql)
end

equations = CompressibleMoistEulerEquations2D(;Rain=true)

AtmossphereData = AtmossphereLayers(equations)

function initial_condition_moist_rain(x, t, equations)
  return initial_condition_rain(x, t, equations, AtmossphereData)
end

initial_condition = initial_condition_moist_rain

boundary_condition = (x_neg=boundary_condition_periodic,
                      x_pos=boundary_condition_periodic,
                      y_neg=boundary_condition_slip_wall,
                      y_pos=boundary_condition_slip_wall)

source_term = source_terms_rain

###############################################################################
# Get the DG approximation space


polydeg = 4
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_LMARS_rain
volume_flux = flux_ranocha

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_liquid)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

#volume_integral=VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux)

coordinates_min = (-860.0, 2000.0)
coordinates_max = (860.0, 3520.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                periodicity=(true, false),
                n_cells_max=40_000)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition,
                                    source_terms=source_term)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 40.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
solution_variables = cons2cons

analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=solution_variables)


#amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=velocity),
#                                      base_level=3, max_level=6,
#                                      med_threshold=1, max_threshold=5)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=velocity),
                                      base_level=3, max_level=5,
                                      med_threshold=0.2, max_threshold=1.0)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=false,
                           adapt_initial_condition_only_refine=false)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
#                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

#limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
#                                               variables=(Trixi.density, pressure))
#stage_limiter! = limiter!
#step_limiter!  = limiter!

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
