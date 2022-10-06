###############################################################################
# This example is equivalent to tree_1d_dgsem/elixir_shallowwater_ec.jl,      #         
# but instead of a function for the bottom topography, this version uses a    #
# cubic B-spline interpolation with not-a-knot boundary condition to          #
# approximate the bottom topography. The interpolation points are provided    #
# via a gist.                                                                 #        
###############################################################################

using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations with a discontinuous
# bottom topography function

equations = ShallowWaterEquations1D(gravity_constant=9.81)

###############################################################################
# The data for the bottom topography is saved as a .txt-file in a gist.
# To create the data, the following arbitrary continuous function
# bottom_topography(x) = sin(x)
# has been evaluated at 10 equally spaced points between [-1,1] and the
# resulting values have been saved.
spline_data_1 = download("https://gist.githubusercontent.com/maxbertrand1996/da02a5fbfe6cda853709574590328d90/raw/475d5dc420b5029c0189ad3261fe48b1f53c183c/data_swe_ec_1D_1.txt")

# Spline interpolation
spline          = cubic_b_spline(spline_data; boundary = "not-a-knot")
spline_func(x)  = spline_interpolation(spline, x)

# Note, this initial condition is used to compute errors in the analysis 
# callback but the initialization is
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
# Workaround to set a discontinuous bottom topography and initial condition for 
# debugging and testing.
# Alternative version of the initial conditinon used to setup a truly 
# discontinuous bottom topography function and initial condition for this 
# academic testcase of entropy conservation.
# The errors from the analysis callback are not important but `∑∂S/∂U ⋅ Uₜ` 
# should be around machine roundoff. In contrast to the usual signature of 
# initial conditions, this one get passed the
# `element_id` explicitly. In particular, this initial conditions works as 
# intended only for the TreeMesh1D with `initial_refinement_level=4`.

###############################################################################
# The data for the discountinuous part of the 
# bottom topography is saved as a .txt-file in a gist.
# To create the data, the following function
# disc_bottom_topography(x) = 2.0 + 0.5 * sin(2.0 * pi * x)
# has been evaluated at 10 equally spaced points between [-1,1] and the
# resulting values have been saved.
spline_data_2 = download("https://gist.githubusercontent.com/maxbertrand1996/fe07089bbacdc11afbf4b8677db81eaf/raw/9b509c9abef2705168af28cbd5413cf7fa3e6d1a/data_swe_ec_1D_2.txt")

# Spline interpolation
disc_spline         = cubic_b_spline(spline_data_2; boundary = "not-a-knot")
disc_spline_func(x) = spline_interpolation(disc_spline, x)


function initial_condition_ec_discontinuous_bottom(x, t, element_id, equations::ShallowWaterEquations1D)
  # Set the background values
  H = 4.25
  v = 0.0
  b = spline_func(x[1]) # arbitrary continuous function

  # setup the discontinuous water height and velocity
  if element_id == 10
    H = 5.0
    v = 0.1882
  end

  # Setup a discontinuous bottom topography using the element id number
  if element_id == 7
    b = disc_spline_func(x[1])
  end

  return prim2cons(SVector(H, v, b), equations)
end

# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
  for i in eachnode(semi.solver)
    x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations, semi.solver, i, 
                                   element)
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
