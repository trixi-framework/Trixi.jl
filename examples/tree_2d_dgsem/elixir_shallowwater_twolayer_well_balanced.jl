
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the two-layer shallow water equations with a discontinuous
# bottom topography function

equations = ShallowWaterTwoLayerEquations2D(gravity_constant=9.81, H0=0.6, rho1=0.9, rho2=1.0)

# An initial condition with constant total water height an optional perturbation and zero velocities
# to test well-balancedness and entropy conservation.
function initial_condition_well_balanced(x, t, equations::ShallowWaterTwoLayerEquations2D)
  # Set the background values
  inicenter = 0.5
  x_norm = sqrt((x[1] - inicenter)^2+(x[2] - inicenter)^2)
  r = abs(x_norm)


  # Add perturbation to h1
  add_perturbation = false
  if add_perturbation == true
    h1 = (0.38<=x[1]<=0.42 && 0.38<=x[2]<=0.42) ? 0.15 : 0.1
  else 
    h1 = 0.1
  end

  H2 = 0.5
  H1 = H2 + h1
  v1 = 0.0
  w1 = 0.0
  v2 = 0.0
  w2 = 0.0

  # Bottom Topography
  b = ((x[1]-0.5)^2+(x[2]-0.5)^2) < 0.04 ? 0.2*(cos(4*π*sqrt((x[1]-0.5)^2+(x[2]-0.5)^2))+1) : 0.0
  return prim2cons(SVector(H1, v1, w1, H2, v2, w2, b), equations)
end

initial_condition = initial_condition_well_balanced

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_es, flux_nonconservative_fjordholm_etal)
solver = DGSEM(polydeg=3, surface_flux=surface_flux,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000,
                periodicity=true)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solver

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(lake_at_rest_error,))

stepsize_callback = StepsizeCallback(cfl=1.0)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
