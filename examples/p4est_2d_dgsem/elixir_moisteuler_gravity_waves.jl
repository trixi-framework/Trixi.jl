
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation


equations = CompressibleMoistEulerEquations2D()

initial_condition = Trixi.initial_condition_gravity_wave


function source_term(u, x, t, equations::CompressibleMoistEulerEquations2D)
  return (Trixi.source_terms_geopotential(u, equations) +
          Trixi.source_terms_phase_change(u, equations::CompressibleMoistEulerEquations2D) +
          Trixi.source_terms_raylight_sponge(u, x, t, equations::CompressibleMoistEulerEquations2D))
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
surface_flux = Trixi.flux_LMARS
volume_flux = Trixi.flux_chandrasekhar
                                                              
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


# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.

function bottom(x)
  h = 1.0
  a = 10000.0
  x_length = 240000.0
  # linear function cx for x in [-1,1]
  c = x_length / 2
  # return (cx , f(cx)-f(c))
  return SVector(c * x , (h * a^2 * inv((c * x)^2+a^2)) - (h * a^2 * inv((c)^2+a^2)))
end

f1(s) = SVector(-120000.0, 15000.0 * s + 15000.0)
f2(s) = SVector( 120000.0, 15000.0 * s + 15000.0 )
f3(s) = bottom(s)
f4(s) = SVector(120000.0 * s, 30000.0)


f1(s) = SVector(-120000.0, 15000.0 * s + 15000.0)
f2(s) = SVector( 120000.0, 15000.0 * s + 15000.0)
f3(s) = SVector(120000.0 * s, (1.0 * 10000.0^2 * inv((120000.0 * s)^2+10000.0^2))-(1.0 * 10000.0^2 * inv((120000.0)^2+10000.0^2)))
f4(s) = SVector(120000.0 * s, 30000.0 )

faces = (f1, f2, f3, f4)



cells_per_dimension = (80, 80)
#(64, 32)
#(288, 125)

mesh = StructuredMesh(cells_per_dimension, faces)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_term, boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 15000.0)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10000
#analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
#                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=5000,
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


sol = solve(ode, SSPRK33(),
            maxiters=1.0e7,
            dt=1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

#sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#            dt=1, # solve needs some value here but it will be overwritten by the stepsize_callback
#            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
