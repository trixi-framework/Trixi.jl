
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the two-layer shallow water equations for a dam break test with a 
# discontinuous bottom topography function to test entropy conservation

equations = ShallowWaterTwoLayerEquations1D(gravity_constant=9.81, H0=2.0, rho_upper=0.9, rho_lower=1.0)

###############################################################################
# Workaround to set a discontinuous bottom topography and initial condition.

# This test case uses a special work around to setup a truly discontinuous bottom topography 
# function and initial condition for this academic testcase of entropy conservation. First, a 
# dummy initial condition is introduced to create the semidiscretization. Then the initial condition 
# is reset with the true discontinuous values from initial_condition_dam_break. Note, that this
# initial condition only works for TreeMesh1D with `initial_refinement_level=5`.
initial_condition = initial_condition_convergence_test

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))


###############################################################################
# Get the TreeMesh and setup a non-periodic mesh

coordinates_min = 0.0
coordinates_max = 20.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=10000,
                periodicity=false)

boundary_condition = boundary_condition_slip_wall

# create the semidiscretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, 
                                    boundary_conditions=boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0,0.4)
ode = semidiscretize(semi, tspan)

# Initial conditions dam break test case
function initial_condition_dam_break(x, t, element_id, equations::ShallowWaterTwoLayerEquations1D)
  v1_upper = 0.0
  v1_lower = 0.0

  # Set the discontinuity
  if element_id <= 16
    H_lower = 2.0
    H_upper = 4.0
    b  = 0.0
  else
    H_lower = 1.5
    H_upper = 3.0
    b  = 0.5
  end

  return prim2cons(SVector(H_upper, v1_upper, H_lower, v1_lower, b), equations)
end


# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
  for i in eachnode(semi.solver)
    x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates,
      equations, semi.solver, i, element)
    u_node = initial_condition_dam_break(x_node, first(tspan), element, equations)
    Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, element)
  end
end




summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false, 
    extra_analysis_integrals=(energy_total, energy_kinetic, energy_internal,))

stepsize_callback = StepsizeCallback(cfl=1.0)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=500,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary