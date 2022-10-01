using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

function initial_condition_nonhydrostatic_gravity_wave(x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack p_0, kappa, gamma, g, c_pd, c_vd, R_d, R_v = equations
  z = x[2]
  T_0 = 280.0
  theta_0 = T_0
  N = 0.01

  theta = theta_0 * exp(N^2 *z / g)
  p = p_0*(1 + g^2 * inv(c_pd * theta_0 * N^2) * (exp(-z * N^2 / g) - 1))^(1/kappa)
  # ??????????????
  #rho = p_0 * inv(theta * R_d * (p / p_0)^(c_vd / c_pd))
  rho = p / ((p / p_0)^kappa*R_d*theta)
  T = p / (R_d * rho)

  v1 = 10
  v2 = 0
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_E = rho * c_vd * T + 0.5 * rho * (v1^2 + v2^2)
  rho_qv = 0
  rho_ql = 0
  return SVector(rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql)
end

equations = CompressibleMoistEulerEquations2D()

initial_condition = initial_condition_nonhydrostatic_gravity_wave

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
volume_flux = flux_chandrasekhar
                                                              
#indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                         alpha_max=0.5,
#                                         alpha_min=0.001,
#                                         alpha_smooth=true,
#                                         variable=density_pressure)
#volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                 volume_flux_dg=volume_flux,
#                                                 volume_flux_fv=surface_flux) 

volume_integral=VolumeIntegralFluxDifferencing(volume_flux)
                      
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


f1(s) = SVector(-20000.0, 8000.0 * s + 8000.0)
f2(s) = SVector( 20000.0, 8000.0 * s + 8000.0)
f3(s) = SVector( 20000.0 * s, (400.0 * 1000.0^2 * inv((20000.0 * s)^2+1000.0^2))-(400.0 * 1000.0^2 * inv((20000.0)^2+1000.0^2)))
f4(s) = SVector( 20000.0 * s, 16000.0)

faces = (f1, f2, f3, f4)


# dx = 0.2*a  dz = 10-200 m  für (40,16) km 
cells_per_dimension = (50, 40)#(200, 160)
#(64, 32)
#(288, 125)

mesh = StructuredMesh(cells_per_dimension, faces)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_term, boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# t = 21.6*a/v_1
tspan = (0.0, 2160.0)

#tspan = (0.0, 600.0)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=5000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2drypot)


stepsize_callback = StepsizeCallback(cfl=0.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback);


###############################################################################
# run the simulation


sol = solve(ode, SSPRK33(),
            maxiters=1.0e7,
            dt=1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

#sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#            dt=1, # solve needs some value here but it will be overwritten by the stepsize_callback
#            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
