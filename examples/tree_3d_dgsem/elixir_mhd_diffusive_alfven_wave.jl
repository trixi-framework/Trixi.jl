
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the resistive compressible MHD equations

prandtl_number() = 0.72
mu() = 4e-2
eta = 4e-2

equations = IdealGlmMhdEquations3D(1.4)
equations_parabolic = CompressibleMhdDiffusion3D(equations, mu=mu(), Prandtl=prandtl_number(),
						 eta=eta,
                                                 gradient_variables=GradientVariablesPrimitive())

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = ( 1.0,  1.0,  1.0) # maximum coordinates (max(x), max(y), max(z))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=50_000) # set maximum capacity of tree data structure


function initial_condition_constant_alfven(x, t, equations)
  p = 1
  omega = 2*pi / 2 # may be multiplied by frequency
  # r: length-variable = length of computational domain
  r = 2
  e = 0.02
  sqr = 1
  Va  = omega
  phi_alv = omega * (x[1] - t)

  rho = 1.
  rho_v1  = 0
  rho_v2  = e*cos(phi_alv)
  rho_v3  =  0
  rho_e = 10.
  B1  = 1
  B2  = -rho_v2*sqr
  B3  = 0
  psi = 0

  return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
end

initial_condition = initial_condition_constant_alfven

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
					     initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

cfl = 1.5
stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)


###############################################################################
# run the simulation

time_int_tol = 1e-5
sol = solve(ode, RDPK3SpFSAL49(), abstol=time_int_tol, reltol=time_int_tol, dt = 1e-5,
            save_everystep=false, callback=callbacks)

# Print the timer summary.
summary_callback()

