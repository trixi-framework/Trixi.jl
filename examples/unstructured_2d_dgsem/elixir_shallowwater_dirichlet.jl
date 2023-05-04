
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the shallow water equations with a continuous
# bottom topography function (set in the initial conditions)

equations = ShallowWaterEquations2D(gravity_constant=1.0, H0=3.0)

# An initial condition with constant total water height and zero velocities to test well-balancedness.
function initial_condition_well_balancedness(x, t, equations::ShallowWaterEquations2D)
  # Set the background values
  H = equations.H0
  v1 = 0.0
  v2 = 0.0
  # bottom topography taken from Pond.control in [HOHQMesh](https://github.com/trixi-framework/HOHQMesh)
  x1, x2 = x
  b = (  1.5 / exp( 0.5 * ((x1 - 1.0)^2 + (x2 - 1.0)^2) )
       + 0.75 / exp( 0.5 * ((x1 + 1.0)^2 + (x2 + 1.0)^2) ) )
  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_well_balancedness

boundary_condition_constant = BoundaryConditionDirichlet(initial_condition)
boundary_condition = Dict( :OuterCircle => boundary_condition_constant )

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=4, surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# This setup is for the curved, split form well-balancedness testing

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "mesh_outer_circle.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/9beddd9cd00e2a0a15865129eeb24928/raw/be71e67fa48bc4e1e97f5f6cd77c3ed34c6ba9be/mesh_outer_circle.mesh",
                                      default_mesh_file)
mesh_file = default_mesh_file

mesh = UnstructuredMesh2D(mesh_file)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition)

###############################################################################
# ODE solvers, callbacks, etc.

tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     save_analysis=true,
                                     extra_analysis_integrals=(lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(); abstol=1.0e-11, reltol=1.0e-11,
            ode_default_options()..., callback=callbacks);
summary_callback() # print the timer summary
