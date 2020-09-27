# TODO: Taal refactor, rename to
# - linear_advection_mortar.jl
# - advection_mortar.jl
# or something similar?

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0)
# advectionvelocity = (0.2, -0.3)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

initial_conditions = initial_conditions_convergence_test

surface_flux = flux_lax_friedrichs
solver = DGSEM(3, surface_flux)

coordinates_min = (-1, -1)
coordinates_max = ( 1,  1)
refinement_patches = (
  (type="box", coordinates_min=(0.0, -1.0), coordinates_max=(1.0, 1.0)),
)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                refinement_patches=refinement_patches,
                n_cells_max=10_000,)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_conditions, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, analysis_interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

save_solution = SaveSolutionCallback(solution_interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

# TODO: Taal, restart
# restart_interval = 10

stepsize_callback = StepsizeCallback(cfl=2.0)

callbacks = CallbackSet(summary_callback, stepsize_callback, analysis_callback, save_solution, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
