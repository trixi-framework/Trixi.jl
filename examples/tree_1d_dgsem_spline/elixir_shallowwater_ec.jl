
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations with a discontinuous
# bottom topography function

equations = ShallowWaterEquations1D(gravity_constant=9.81)

# bottom topography function
bottom_topography(x) = sin(x) # arbitrary continuous function

# Setting
range_x         = [-1.0, 1.0]
num_interp_val  = 10
x_val           = Vector(LinRange(range_x[1], range_x[2], num_interp_val))
y_val           = bottom_topography.(x_val)

# Spline interpolation
spline          = cubic_spline(x_val, y_val)
spline_func(x)  = spline_interpolation(spline, x)

# Note, this initial condition is used to compute errors in the analysis callback but the initialization is
# overwritten by `initial_condition_ec_discontinuous_bottom` below.

function initial_condition_weak_blast_wave_spline(x, t, equations::ShallowWaterEquations1D)
  
  inicenter = 0.7
  x_norm = x[1] - inicenter
  r = abs(x_norm)

  # Calculate primitive variables
  H = r > 0.5 ? 3.25 : 4.0
  v = r > 0.5 ? 0.0 : 0.1882
  b = spline_func(x[1])

  return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_weak_blast_wave_spline

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=4, surface_flux=(flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solver

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Workaround to set a discontinuous bottom topography and initial condition for debugging and testing.

# alternative version of the initial conditinon used to setup a truly discontinuous
# bottom topography function and initial condition for this academic testcase of entropy conservation.
# The errors from the analysis callback are not important but `∑∂S/∂U ⋅ Uₜ` should be around machine roundoff
# In contrast to the usual signature of initial conditions, this one get passed the
# `element_id` explicitly. In particular, this initial conditions works as intended
# only for the TreeMesh1D with `initial_refinement_level=4`.
function initial_condition_ec_discontinuous_bottom(x, t, element_id, equations::ShallowWaterEquations1D)
  # Set the background values
  H = 4.25
  v = 0.0
  b = sin(x[1]) # arbitrary continuous function

  # setup the discontinuous water height and velocity
  if element_id == 10
    H = 5.0
    v = 0.1882
  end

  # Setup a discontinuous bottom topography using the element id number
  if element_id == 7
    b = 2.0 + 0.5 * sin(2.0 * pi * x[1])
  end

  return prim2cons(SVector(H, v, b), equations)
end

# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
  for i in eachnode(semi.solver)
    x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations, semi.solver, i, element)
    u_node = initial_condition_ec_discontinuous_bottom(x_node, first(tspan), element, equations)
    Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, element)
  end
end

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=3.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
