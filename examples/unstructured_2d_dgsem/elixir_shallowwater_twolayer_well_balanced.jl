
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the shallow water equations with a discontinuous
# bottom topography function (set in the initial conditions)

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
    #h1 = (0.38<=x[1]<=0.42 && 0.38<=x[2]<=0.42) ? 0.15 : 0.1 # circle
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
  b = ((x[1]-0.5)^2+(x[2]-0.5)^2) < 0.05 ? 0.2*(cos(4*π*sqrt((x[1]-0.5)^2+(x[2]-0.5)^2))+1) : 0.0
  #b = 0.0
  return prim2cons(SVector(H1, v1, w1, H2, v2, w2, b), equations)
end

initial_condition = initial_condition_well_balancedness

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux=(flux_fjordholm_etal, flux_nonconservative_fjordholm_etal)
solver = DGSEM(polydeg=6, surface_flux=surface_flux,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# This setup is for the curved, split form well-balancedness testing

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "mesh_alfven_wave_with_twist_and_flip.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/8f8cd23df27fcd494553f2a89f3c1ba4/raw/85e3c8d976bbe57ca3d559d653087b0889535295/mesh_alfven_wave_with_twist_and_flip.mesh",
                                      default_mesh_file)
mesh_file = default_mesh_file

mesh = UnstructuredMesh2D(mesh_file, periodicity=true)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solver

tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Workaround to set a discontinuous bottom topography for debugging and testing.

# alternative version of the initial conditinon used to setup a truly discontinuous
# bottom topography function for this academic testcase.
# The errors from the analysis callback are not important but the error for this lake at rest test case
# `∑|H0-(h+b)|` should be around machine roundoff
# In contrast to the usual signature of initial conditions, this one get passed the
# `element_id` explicitly. In particular, this initial conditions works as intended
# only for the specific mesh loaded above!

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
