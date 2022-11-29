
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the shallow water equations with a discontinuous
# bottom topography function

equations = TwoLayerShallowWaterEquations2D(gravity_constant=1.0, H0=0.6, rho1=0.9, rho2=1.0)

# An initial condition with constant total water height an optional perturbation and zero velocities
# to test well-balancedness and entropy conservation.
function initial_condition_well_balancedness(x, t, equations::TwoLayerShallowWaterEquations2D)
  # Set the background values
  inicenter = 0.5
  x_norm = sqrt((x[1] - inicenter)^2+(x[2] - inicenter)^2)
  r = abs(x_norm)


  # Add perturbation to h2
  add_perturbation = false
  if add_perturbation == true
    h1 = (0.38<=x[1]<=0.42 && 0.38<=x[2]<=0.42) ? 0.15 : 0.1 # circle
    #h1 = (0.38<=x[2]<=0.42) ? 0.15 : 0.1
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
  #b = 0.4 < x[1] < 0.6 ? 0.2 * (cos(10*π*(x[1] - 0.5)) + 1) : 0.0    # x-direction
  #b  = 0.4 < x[2] < 0.6 ? 0.2 * (cos(10*π*(x[2] - 0.5)) + 1) : 0.0   # y-direction
  b = ((x[1]-0.5)^2+(x[2]-0.5)^2) < 0.04 ? 0.2*(cos(4*π*sqrt((x[1]-0.5)^2+(x[2]-0.5)^2))+1) : 0.0
  #b = 0.0
  return prim2cons(SVector(H1, v1, w1, H2, v2, w2, b), equations)
end

initial_condition = initial_condition_well_balancedness

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal)
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

###############################################################################
# Workaround to set a discontinuous bottom topography and initial condition for debugging and testing.

# alternative version of the initial conditinon used to setup a truly discontinuous
# bottom topography function for this academic testcase of well-balancedness.
# The errors from the analysis callback are not important but the error for this lake at rest test case
# `∑|H0-(h+b)|` should be around machine roundoff
# In contrast to the usual signature of initial conditions, this one get passed the
# `element_id` explicitly. In particular, this initial conditions works as intended
# only for the TreeMesh2D with initial_refinement_level=2.
###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 10
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_integrals=(energy_total,lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-10, reltol=1.0e-10,
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
