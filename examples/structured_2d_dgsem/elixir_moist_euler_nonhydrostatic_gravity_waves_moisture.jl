using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation


equations = CompressibleMoistEulerEquations2D()


function moist_state(y, dz, z, y0, H0, equations::CompressibleMoistEulerEquations2D)
  @unpack p_0, g, c_pd, c_pv, c_vd, c_vv, R_d, R_v, c_pl, L_00 = equations
  (p, rho, T, H, rho_qv, rho_ql, theta) = y

    if(10000 > z > 8000)
      a=8000
      b=10000
      y =  ((b-z)/(b-a))
      H0 = H0 * (exp(-1/y)/ (exp(-1/y) + exp(-1/(1-y))))
    elseif (z > 10000)
      H0 = 0
    end

  p0 = first(y0)
  theta0 = last(y0)
  N = 0.01

  F = zeros(7,1)
  rho_qd = rho - rho_qv - rho_ql
  T_C = T - 273.15
  p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
  qd = rho_qd / rho
  qv = rho_qv / rho
  R_m = rho*(qd * R_d + qv * R_v)
  # Assume no liquid
  c_pml = rho*(qd * c_pd + qv * c_pv)
  kappa_m =  R_m * inv(c_pml)
  p_v = R_v * rho_qv * T
  H = p_v / p_vs


  F[1] = (p - p0) / dz + g * rho 
  F[2] = p - (R_d * rho_qd + R_v * rho_qv) * T
  F[3] = theta - T * (p_0 / p)^(kappa_m)
  F[4] = H - H0
  # Assume no liquid
  F[5] = rho_ql
  F[6] = (theta - theta0) / dz + N^2 * theta  / g
  a = p_vs / (R_v * T) - rho_qv
  b = rho - rho_qv - rho_qd
  # ql=0 => phi=0
  F[7] = a+b-sqrt(a*a+b*b)

  return F
end

function generate_function_of_y(dz, z, y0, H0, equations::CompressibleMoistEulerEquations2D)
  function function_of_y(y)
    return moist_state(y, dz, z, y0, H0, equations)
  end
end

struct AtmossphereLayers{RealT<:Real}
  equations::CompressibleMoistEulerEquations2D
  # structure:  1--> i-layer (z = total_hight/precision *(i-1)),  2--> rho, rho_theta, rho_qv, rho_ql
  LayerData::Matrix{RealT}
  total_hight::RealT
  preciseness::Int
  layers::Int
  theta0::RealT
  ground_state::NTuple{2, RealT}
  mixing_ratios::NTuple{2, RealT}
end


function AtmossphereLayers(equations ; total_hight=15010.0, preciseness=10, theta0=280, ground_state=(1.4, 100000.0), mixing_ratios=(0.02, 0.02), RealT=Float64)
  @unpack kappa, p_0, c_pd, c_vd, c_pv, c_vv, R_d, R_v, c_pl = equations
  rho0, p0 = ground_state
  r_t0, r_v0 = mixing_ratios
  theta_0 = theta0
  z = 0.0

  T0 = theta_0
  H0 = 0.8
  T_C = T0 - 273.15
  p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))

  rho_qv0 = H0 * p_vs/ (R_v * T0)
  rho_ql0 = 0.0
  

  y0 = [p0, rho0, T0, H0, rho_qv0, rho_ql0, theta_0]

  n = convert(Int, total_hight / preciseness) 
  LayerData = zeros(RealT, n+1, 4)
  dz = preciseness
  LayerData[1, :] = [rho0, T0, rho_qv0, rho_ql0]
  

   for i in (1:n)

    z = z + dz
    F = generate_function_of_y(dz, z, y0, H0, equations)
    sol = nlsolve(F, y0)
    p, rho, T, H, rho_qv, rho_ql, theta = sol.zero
    y0 = deepcopy(sol.zero)
    LayerData[i+1, :] = [rho, T, rho_qv, rho_ql]
   end
  
  return AtmossphereLayers{RealT}(equations, LayerData, total_hight, dz, n, theta_0, ground_state, mixing_ratios)
end


function initial_condition_nonhydrostatic_gravity_wave_moist(x, t, equations::CompressibleMoistEulerEquations2D, AtmosphereLayers::AtmossphereLayers)
  @unpack LayerData, preciseness, total_hight = AtmosphereLayers
  @unpack p_0, c_pd, c_vd, c_pv, c_vv, R_d, R_v, c_pl, L_00 = equations

  dz = preciseness
  z = x[2] 
  if (z > total_hight && !(isapprox(z, total_hight)))
    error("The atmossphere does not match the simulation domain")
  end
  n = convert(Int, floor((z+eps())/dz)) + 1
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
  rho_qv =  (rho_qv_r  * (z - z_l) + rho_qv_l  * (z_r - z)) / dz
  rho_ql =  (rho_ql_r  * (z - z_l) + rho_ql_l  * (z_r - z)) / dz

  rho_qd = rho - rho_qv - rho_ql

  rho_e = (c_vd * rho_qd + c_vv * rho_qv + c_pl * rho_ql) * T + L_00 * rho_qv

  if isapprox(rho_qv / rho, 0.0, atol=5.0e-14)
    rho_qv = 0.0
  end
  if isapprox(rho_ql/rho, 0.0, atol=5.0e-14)
    rho_ql = 0.0
  end

  v1 = 10.0
  v2 = 0.0
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_E = rho_e + 1/2 * rho *(v1^2 + v2^2)

  return SVector(rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql)
end

AtmossphereData = AtmossphereLayers(equations)

function initial_condition_(x, t, equations)
  return initial_condition_nonhydrostatic_gravity_wave_moist(x, t, equations, AtmossphereData)
end

initial_condition = initial_condition_


function source_term(u, x, t, equations::CompressibleMoistEulerEquations2D)
  return (Trixi.source_terms_geopotential(u, equations) +
          Trixi.source_terms_phase_change(u, equations::CompressibleMoistEulerEquations2D) +
          Trixi.source_terms_nonhydrostatic_raylight_sponge(u, x, t, equations::CompressibleMoistEulerEquations2D))
end


source_term=source_term


boundary_conditions = (
                       x_neg=boundary_condition_periodic,
                       x_pos=boundary_condition_periodic,
                       y_neg=boundary_condition_slip_wall,
                       y_pos=boundary_condition_slip_wall,
                      )

polydeg = 4
basis = LobattoLegendreBasis(polydeg)
surface_flux = flux_LMARS
volume_flux = flux_chandrashekar
                                                              
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


# Deformed rectangle that has "witch of agnesi" as bottom 

function bottom(x)
  h = 400.0
  a = 1000.0
  x_length = 40000.0
  # linear function cx for x in [-1,1]
  c = x_length / 2
  # return (cx , f(cx)-f(c))
  return SVector(c * x , (h * a^2 * inv((c * x)^2+a^2)) - (h * a^2 * inv((c)^2+a^2)))
end

f1(s) = SVector(-20000.0, 8000.0 * s + 8000.0)
f2(s) = SVector( 20000.0, 8000.0 * s + 8000.0)
f3(s) = bottom(s)
f4(s) = SVector( 20000.0 * s, 16000.0)


f1(s) = SVector(-20000.0, 7500.0 * s + 7500.0)
f2(s) = SVector( 20000.0, 7500.0 * s + 7500.0)
f3(s) = SVector( 20000.0 * s, (400.0 * 1000.0^2 * inv((20000.0 * s)^2+1000.0^2))-(400.0 * 1000.0^2 * inv((20000.0)^2+1000.0^2)))
f4(s) = SVector( 20000.0 * s, 15000.0)

faces = (f1, f2, f3, f4)


# dx = 0.2*a  dz = 10-200 m  für (40,16) km 
cells_per_dimension = (120, 100)#(50, 40)
#(64, 32)
#(288, 125)

mesh = StructuredMesh(cells_per_dimension, faces)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_term, boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# t = 21.6*a/v_1
tspan = (0.0, 2160.0)


ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2drypot)


stepsize_callback = StepsizeCallback(cfl=0.2)

callbacks = CallbackSet(summary_callback,
                        #analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback);


###############################################################################
# run the simulation

limiter! = PositivityPreservingLimiterZhangShu(thresholds=(0.0, 0.0 ),
                                               variables=(Trixi.density_liquid, Trixi.density_vapor))
stage_limiter! = limiter!
step_limiter!  = limiter!

sol = solve(ode, SSPRK33(stage_limiter!, step_limiter!),
            maxiters=1.0e7,
            dt=1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

#sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#            dt=1, # solve needs some value here but it will be overwritten by the stepsize_callback
#            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
