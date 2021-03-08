using OrdinaryDiffEq
using Trixi
using Plots


function initial_condition(x, t, equations::AcousticPerturbationEquations2D)
  m = 0.3 # Mach number

  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = 0.0

  v1_mean = x[2] > 1 ? m : m * (2*x[2] - 2*x[2]^2 + x[2]^4)
  v2_mean = 0.0
  c_mean = 1.0
  rho_mean = 1.0

  return SVector(v1_prime, v2_prime, p_prime, v1_mean, v2_mean, c_mean, rho_mean)
end

function boundary_conditions(u_inner, orientation, direction, x, t, surface_flux_function,
                             equations::AcousticPerturbationEquations2D)
  # Calculate boundary flux
  if direction == 1 # Boundary at -x
    u_boundary = SVector(0.0, 0.0, 0.0, u_inner[4], u_inner[5], u_inner[6], u_inner[7])
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  elseif direction == 2 # Boundary at +x
    u_boundary = SVector(0.0, 0.0, 0.0, u_inner[4], u_inner[5], u_inner[6], u_inner[7])
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  elseif direction == 3 # Boundary at -y
    if -0.05 <= x[1] <= 0.05 # Monopole
      v1_prime = 0.0
      v2_prime = p_prime = sin(2 * pi * t)

      u_boundary = SVector(v1_prime, v2_prime, p_prime, u_inner[4], u_inner[5], u_inner[6],
                           u_inner[7])
    else # Wall
      u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5], u_inner[6],
                           u_inner[7])
    end

    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  else # Boundary at +y
    u_boundary = SVector(0.0, 0.0, 0.0, u_inner[4], u_inner[5], u_inner[6], u_inner[7])
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  end

  return flux
end

###############################################################################
# semidiscretization of the acoustic perturbation equations

equations = AcousticPerturbationEquations2D()

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(3, flux_lax_friedrichs)

coordinates_min = (-20.6, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (30.6, 51.2) # maximum coordinates (max(x), max(y))

# Create a uniformely refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=8,
                n_cells_max=100_000,
                periodicity=false)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 24.0
ode = semidiscretize(semi, (0.0, 24.0))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

visualization = VisualizationCallback(interval=10; clims=(-0.1, 0.1), variable_names=["p_prime"],
                                      c=:grayC)

alive = AliveCallback(analysis_interval=200)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.8)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
#callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)
callbacks = CallbackSet(summary_callback, alive, stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()

pd = PlotData2D(sol)
plot(pd["p_prime"], clims=(-0.1, 0.1), c=:grayC)