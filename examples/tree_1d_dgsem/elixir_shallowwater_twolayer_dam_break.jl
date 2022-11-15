
using OrdinaryDiffEq
using Trixi

# own module
#using Autoplot

###############################################################################
# Semidiscretization of the shallow water equations

equations = TwoLayerShallowWaterEquations1D(gravity_constant=9.81,H0=2.0,rho1=0.9,rho2=1.0)

initial_condition = initial_condition_convergence_test

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
              volume_integral=VolumeIntegralFluxDifferencing(volume_flux))


###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = 0.0
coordinates_max = 20.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10000,
                periodicity=false)


boundary_condition = boundary_condition_slip_wall

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0,0.4)
ode = semidiscretize(semi, tspan)

# Initial Conditions well balanced test case
function initial_condition_dam_break(x, t, element_id, equations::TwoLayerShallowWaterEquations1D)
  inicenter = 10.0
  x_norm = x[1] - inicenter
  r = abs(x_norm)

  v1 = 0.0
  v2 = 0.0
  # TODO: Discontinuity cannot be represented this way with LGL 
  if element_id <= 32
    H2 = 2.0
    H1 = 4.0
  else
    H2 = 1.5
    H1 = 3.0
  end

  b  = 0.0
  return prim2cons(SVector(H1, v1, H2, v2, b), equations)
end


# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
for i in eachnode(semi.solver)
  x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations, semi.solver, i, element)
  u_node = initial_condition_dam_break(x_node, first(tspan), element, equations)
  Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, element)
end
end




summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true, 
                      extra_analysis_integrals=(energy_total,))

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