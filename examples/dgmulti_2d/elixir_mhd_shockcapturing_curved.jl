
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations2D(1.4)

initial_condition = initial_condition_weak_blast_wave
# initial_condition = initial_condition_constant

polydeg = 3
basis = DGMultiBasis(Quad(), polydeg, approximation_type=GaussSBP())

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = volume_integral)

# curved mapping from [-1, 1]^2 to [-2, 2]^2
function mapping(xi, eta)
  x = xi  + 0.1 * sin(pi * xi) * sin(pi * eta)
  y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
  return 2 * SVector(x, y)
end
cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension, mapping;
                   coordinates_min=(-2.0, -2.0), coordinates_max=(2.0, 2.0),
                   periodicity=true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
alive_callback = AliveCallback(analysis_interval=analysis_interval)

cfl = .75
stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        alive_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
