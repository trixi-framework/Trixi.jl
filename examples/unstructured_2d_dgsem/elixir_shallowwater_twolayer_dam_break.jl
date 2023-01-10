
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the two-layer shallow water equations with a discontinuous
# bottom topography function (set in the initial conditions)

equations = ShallowWaterTwoLayerEquations2D(gravity_constant=1.0, rho1=0.9, rho2=1.0)

function initial_condition_dam_break(x, t,equations::ShallowWaterTwoLayerEquations2D)
  if x[1] < 1.44
    H1 = 1.0
    H2 = 0.6
  else
    H1 = 0.9
    H2 = 0.5
  end

  v1 = 0.0
  w1 = 0.0
  v2 = 0.0
  w2 = 0.0
  b  = 0.0

  return prim2cons(SVector(H1, v1, w1, H2, v2, w2, b), equations)
end

initial_condition = initial_condition_dam_break

boundary_condition_constant = BoundaryConditionDirichlet(initial_condition_dam_break)


###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux= (flux_lax_friedrichs, flux_nonconservative_fjordholm_etal)
solver = DGSEM(polydeg=6, surface_flux=surface_flux,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# This setup is for the curved, split form well-balancedness testing

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "mesh_alfven_wave_with_twist_and_flip.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/8f8cd23df27fcd494553f2a89f3c1ba4/raw/85e3c8d976bbe57ca3d559d653087b0889535295/mesh_alfven_wave_with_twist_and_flip.mesh",
                                      default_mesh_file)
mesh_file = default_mesh_file

mesh = UnstructuredMesh2D(mesh_file, periodicity=false)

# Boundary conditions
boundary_condition = Dict(:Top    => boundary_condition_slip_wall,
                          :Left   => boundary_condition_slip_wall,
                          :Right  => boundary_condition_slip_wall,
                          :Bottom => boundary_condition_slip_wall)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, 
                                    solver, boundary_conditions=boundary_condition)

###############################################################################
# ODE solver

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

###############################################################################
# Workaround to set a discontinuous bottom topography and initial condition for debugging and testing.

# alternative version of the initial conditinon used to setup a truly discontinuous
# bottom topography function and initial condtion for this academic testcase of entropy conservation.
# The errors from the analysis callback are not important but `∑∂S/∂U ⋅ Uₜ` should be around machine roundoff
# In contrast to the usual signature of initial conditions, this one get passed the
# `element_id` explicitly. In particular, this initial conditions works as intended
# only for the specific mesh loaded above!

function initial_condition_discontinuous_bottom(x, t, element_id,equations::ShallowWaterTwoLayerEquations2D)
  v1 = 0.0
  w1 = 0.0
  v2 = 0.0
  w2 = 0.0
  b  = 0.0

  # Left side of disconinuity
  if element_id == 1  || element_id == 2 || element_id == 5 || element_id == 6 || element_id == 9 || element_id == 10 || element_id == 13 || element_id == 14
    H1 = 1.0
    H2 = 0.6
  # Right side of disconinuity
  else
    H1 = 0.9
    H2 = 0.5
  end

  return prim2cons(SVector(H1, v1, w1, H2, v2, w2, b), equations)
end

# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
  for j in eachnode(semi.solver), i in eachnode(semi.solver)
    x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations, semi.solver, i, j, element)
    u_node = initial_condition_discontinuous_bottom(x_node, first(tspan), element, equations)
    Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, j, element)
  end
end


###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,save_analysis=false,
                                     extra_analysis_integrals=(lake_at_rest_error,energy_total,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=500,
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
