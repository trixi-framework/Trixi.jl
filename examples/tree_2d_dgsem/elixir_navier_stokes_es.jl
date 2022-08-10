using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

reynolds_number() = 1600
prandtl_number() = 0.72

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, Reynolds=reynolds_number(), Prandtl=prandtl_number(),
                                                          Mach_freestream=0.5)

# TODO: For entropy stability testing
volume_flux = flux_ranocha
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-2.0, -2.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 2.0,  2.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000) # set maximum capacity of tree data structure

function initial_condition_blast_wave(x, t, equations)
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)
  sin_phi, cos_phi = sincos(phi)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 2.0
  v1  = r > 0.5 ? 0.0 : 0.2 * cos_phi
  v2  = r > 0.5 ? 0.0 : 0.2 * sin_phi
  p   = r > 0.5 ? 1.0 : 3.0

  return prim2cons(SVector(rho, v1, v2, p), equations)
end

initial_condition = initial_condition_blast_wave

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback, save_solution)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(), abstol=time_int_tol, reltol=time_int_tol, dt = 1e-5,
            save_everystep=false, callback=callbacks)
summary_callback() # print the timer summary

