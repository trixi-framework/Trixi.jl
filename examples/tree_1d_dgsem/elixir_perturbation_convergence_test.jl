using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations

tau = 0.5
vxr = 0.1
equations = PerturbationMomentSystem1D(vxr, 2/3, tau)
initial_condition = initial_condition_convergence_test


solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = 0
coordinates_max = 6

mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level=4, n_cells_max=10_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms=source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

t = 0.5
tspan = (0.0, t)
ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100, save_final_restart=true)

save_solution = SaveSolutionCallback(interval=100, solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.1)

callbacks = CallbackSet(summary_callback,analysis_callback, alive_callback, save_restart, save_solution, stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=0.001, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);


summary_callback()


pd = PlotData1D(sol; solution_variables=cons2prim)
#plot(pd.x, pd.data[:,2], xlims = (coordinates_min, coordinates_max), title ="vx", label = "M t = " *string(t), markersize=3)
plot(pd, size=(1000,500))

