
using OrdinaryDiffEq
using Trixi


@inline function initial_condition_viscous_shock_solved(x, t, equations::HyperbolicNavierStokesEquations1D)
  tol = 1e-10
  ind = 0
  while ind == 0
    for i in 1:size(x_sol)[1]
      if isapprox(x_sol[i], x[1], atol=tol)
        ind = i
        break
      end
    end
    tol *= 10
  end
  return SVector(rho_sol[ind], rho_v1_sol[ind], rho_e_sol[ind], tau_sol[ind], q_sol[ind])
end

###############################################################################
# viscous shock example for the Navier-Stokes equations

equations = HyperbolicNavierStokesEquations1D()

polydeg = 3
refinement_level = 4

# include the solution computed with the fortran code taken from http://www.cfdbooks.com/cfdcodes.html
# this solution is used as the initial condition for the DG-Solver
x_sol, rho_sol, rho_v1_sol, rho_e_sol, tau_sol, q_sol = calc_viscous_shock_solution(polydeg, refinement_level)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg, flux_lax_friedrichs)

coordinates_min = -1 # minimum coordinate
coordinates_max = 1 # maximum coordinate

# Create a uniformely refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=refinement_level,
                n_cells_max=30_000,
                periodicity=false) # set maximum capacity of tree data structure


initial_condition = initial_condition_viscous_shock_solved

boundary_conditions = boundary_condition_viscous_shock

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions,
                                    source_terms=source_terms_harmonic)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 3.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

resid_tol = 5.0e-12
steady_state_callback = SteadyStateCallback(abstol=resid_tol, reltol=0.0)

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100,
                                     extra_analysis_integrals=(entropy, energy_total))

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.9)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, steady_state_callback,
                        analysis_callback, alive_callback,
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


"""
This section can be uncommanded to get the correct l_inf errors of the obtained solution calculated and printed out.
The errors can not be calculated correctly by Trixis callback function due to the initial condition which is once computed by fortran code and then fixed for the runtime of the elixir.
"""
# rho = sol.u[2][[5*i-4 for i in 1:Int(size(sol.u[2])[1]/5)]]
# rho_v1 = sol.u[2][[5*i-3 for i in 1:Int(size(sol.u[2])[1]/5)]]
# rho_e = sol.u[2][[5*i-2 for i in 1:Int(size(sol.u[2])[1]/5)]]
# tau = sol.u[2][[5*i-1 for i in 1:Int(size(sol.u[2])[1]/5)]]
# q = sol.u[2][[5*i for i in 1:Int(size(sol.u[2])[1]/5)]]
#
# println("l_inf error rho:    ", maximum(abs, rho-rho_sol))
# println("l_inf error rho_v1: ", maximum(abs, rho_v1-rho_v1_sol))
# println("l_inf error rho_e:  ", maximum(abs, rho_e-rho_e_sol))
# println("l_inf error tau:    ", maximum(abs, tau-tau_sol))
# println("l_inf error q:      ", maximum(abs, q-q_sol))
