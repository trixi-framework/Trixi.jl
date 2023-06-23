using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the acoustic perturbation equations

equations = AcousticPerturbationEquations2D(v_mean_global=(0.0, 0.0), c_mean_global=0.0,
                                            rho_mean_global=0.0)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-20.6, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (30.6, 51.2) # maximum coordinates (max(x), max(y))

"""
  initial_condition_monopole(x, t, equations::AcousticPerturbationEquations2D)

Initial condition for the monopole in a boundary layer setup, used in combination with
[`boundary_condition_monopole`](@ref).
"""
function initial_condition_monopole(x, t, equations::AcousticPerturbationEquations2D)
  m = 0.3 # Mach number

  v1_prime = 0.0
  v2_prime = 0.0
  p_prime = 0.0

  v1_mean = x[2] > 1 ? m : m * (2*x[2] - 2*x[2]^2 + x[2]^4)
  v2_mean = 0.0
  c_mean = 1.0
  rho_mean = 1.0

  prim = SVector(v1_prime, v2_prime, p_prime, v1_mean, v2_mean, c_mean, rho_mean)

  return prim2cons(prim, equations)
end
initial_condition = initial_condition_monopole # does not use the global mean values given above

"""
  boundary_condition_monopole(u_inner, orientation, direction, x, t, surface_flux_function,
                              equations::AcousticPerturbationEquations2D)

Boundary condition for a monopole in a boundary layer at the -y boundary, i.e. `direction = 3`.
This will return an error for any other direction. This boundary condition is used in combination
with [`initial_condition_monopole`](@ref).
"""
function boundary_condition_monopole(u_inner, orientation, direction, x, t, surface_flux_function,
                                     equations::AcousticPerturbationEquations2D)
  if direction != 3
    error("expected direction = 3, got $direction instead")
  end

  # Wall at the boundary in -y direction with a monopole at -0.05 <= x <= 0.05. In the monopole area
  # we use a sinusoidal boundary state for the perturbed variables. For the rest of the -y boundary
  # we set the boundary state to the inner state and multiply the perturbed velocity in the
  # y-direction by -1.
  if -0.05 <= x[1] <= 0.05 # Monopole
    v1_prime = 0.0
    v2_prime = p_prime = sin(2 * pi * t)

    prim_boundary = SVector(v1_prime, v2_prime, p_prime, u_inner[4], u_inner[5], u_inner[6], u_inner[7])

    u_boundary = prim2cons(prim_boundary, equations)
  else # Wall
    u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5], u_inner[6],
                         u_inner[7])
  end

  # Calculate boundary flux
  flux = surface_flux_function(u_boundary, u_inner, orientation, equations)

  return flux
end

"""
    boundary_condition_zero(u_inner, orientation, direction, x, t, surface_flux_function,
                            equations::AcousticPerturbationEquations2D)

Boundary condition that uses a boundary state where the state variables are zero and the mean
variables are the same as in `u_inner`.
"""
function boundary_condition_zero(u_inner, orientation, direction, x, t, surface_flux_function,
                                 equations::AcousticPerturbationEquations2D)
  value = zero(eltype(u_inner))
  u_boundary = SVector(value, value, value, cons2mean(u_inner, equations)...)

  # Calculate boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end

boundary_conditions = (x_neg=boundary_condition_zero,
                       x_pos=boundary_condition_zero,
                       y_neg=boundary_condition_monopole,
                       y_pos=boundary_condition_zero)

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=100_000,
                periodicity=false)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 24.0
tspan =  (0.0, 24.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100, solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.8)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()