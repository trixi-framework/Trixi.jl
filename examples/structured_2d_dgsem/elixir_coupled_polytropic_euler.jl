# using Revise
# using Infiltrator
# using AbbreviatedStackTraces
# using OrdinaryDiffEq
# # using Trixi2Vtk
using Trixi


"""
Coupled two polytropic Euler systems.
"""


###############################################################################
# define the initial conditions

function initial_condition_wave_isotropic(x, t, equations::PolytropicEulerEquations2D)
  gamma = 2.0
  kappa = 1.0

  rho = 1.0
  v1 = 0.0
  if x[1] > 10.0
    rho = ((1.0 + 0.01*sin(x[1]*2*pi)) / kappa)^(1/gamma)
    v1 = ((0.01*sin((x[1]-1/2)*2*pi)) / kappa)
  end
  v2 = 0.0

  return prim2cons(SVector(rho, v1, v2), equations)
end

function initial_condition_wave_polytropic(x, t, equations::PolytropicEulerEquations2D)
  gamma = 2.0
  kappa = 1.0

  rho = 1.0
  v1 = 0.0
  if x[1] > 0.0
    rho = ((1.0 + 0.01*sin(x[1]*2*pi)) / kappa)^(1/gamma)
    v1 = ((0.01*sin((x[1]-1/2)*2*pi)) / kappa)
  end
  v2 = 0.0

  return prim2cons(SVector(rho, v1, v2), equations)
end

###############################################################################
# semidiscretization of the linear advection equation

Trixi.wrap_array(u_ode::AbstractVector, mesh::Trixi.AbstractMesh, equations, dg::DGSEM, cache) = Trixi.wrap_array_native(u_ode, mesh, equations, dg, cache)

# Define the equations involved.
gamma_A = 1.001
kappa_A = 1.0
equations_A = PolytropicEulerEquations2D(gamma_A, kappa_A)
gamma_B = 2.0
kappa_B = 1.0
equations_B = PolytropicEulerEquations2D(gamma_B, kappa_B)

equations_coupling = CouplingPolytropicEuler2D(0.01)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
volume_flux = flux_ranocha
solver = DGSEM(polydeg=2, surface_flux=flux_hll,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min_A = (-2.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max_A = ( 0.0,  1.0) # maximum coordinates (max(x), max(y))
coordinates_min_B = ( 0.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max_B = ( 2.0,  1.0) # maximum coordinates (max(x), max(y))

cells_per_dimension = (32, 32)

mesh_A = StructuredMesh(cells_per_dimension,
                        coordinates_min_A,
                        coordinates_max_A)
mesh_B = StructuredMesh(cells_per_dimension,
                        coordinates_min_B,
                        coordinates_max_B)

# A semidiscretization collects data structures and functions for the spatial discretization.
semi_A = SemidiscretizationHyperbolic(mesh_A, equations_A,
                                      initial_condition_wave_isotropic, solver,
                                      boundary_conditions=(
#                                       x_neg=BoundaryConditionCoupledAB(2, (:end, :i_forward), Float64, equations_B, equations_coupling),
                                      x_neg=BoundaryConditionCoupled(2, (:end, :i_forward), Float64),
#                                       x_neg=boundary_condition_periodic,
#                                       x_pos=BoundaryConditionCoupledAB(2, (:begin, :i_forward), Float64, equations_B, equations_coupling),
                                      x_pos=BoundaryConditionCoupled(2, (:begin, :i_forward), Float64),
#                                       x_pos=boundary_condition_periodic,
                                      y_neg=boundary_condition_periodic,
                                      y_pos=boundary_condition_periodic))

semi_B = SemidiscretizationHyperbolic(mesh_B, equations_B,
                                      initial_condition_wave_polytropic, solver,
                                      boundary_conditions=(
#                                       x_neg=BoundaryConditionCoupledAB(1, (:end, :i_forward), Float64, equations_A, equations_coupling),
                                      x_neg=BoundaryConditionCoupled(1, (:end, :i_forward), Float64),
#                                       x_neg=boundary_condition_periodic,
#                                       x_pos=BoundaryConditionCoupledAB(1, (:begin, :i_forward), Float64, equations_A, equations_coupling),
                                      x_pos=BoundaryConditionCoupled(1, (:begin, :i_forward), Float64),
#                                       x_pos=boundary_condition_periodic,
                                      y_neg=boundary_condition_periodic,
                                      y_pos=boundary_condition_periodic))

# Define the indices of the 'other' system.
# This is not ideal.
other_list = [2, 1]

# Create a semidiscretization that bundles semi1 and semi2
semi_coupled = SemidiscretizationCoupled((semi_A, semi_B), other_list)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 30.0
ode = semidiscretize(semi_coupled, (0.0, 3.0))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

resid_tol = 5.0e-12
steady_state_callback = SteadyStateCallback(abstol=resid_tol, reltol=0.0)

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi_coupled, interval=analysis_interval)
alive_callback = AliveCallback(analysis_interval=analysis_interval)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=5,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.0)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback,
                        alive_callback,
#                         analysis_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# Convert the snapshots into vtk format.
trixi2vtk("out/solution_*.h5")
