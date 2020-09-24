# TODO: Taal refactor, rename to
# - linear_advection_amr.jl
# - advection_amr.jl
# or something similar?

using OrdinaryDiffEq
using Trixi

advectionvelocity = (1.0, 1.0)
# advectionvelocity = (0.2, -0.3)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

initial_conditions = initial_conditions_gauss

surface_flux = flux_lax_friedrichs
solver = DGSEM(3, surface_flux)

coordinates_min = (-5, -5)
coordinates_max = ( 5,  5)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_conditions, solver)

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

# TODO: Taal implement, printing stuff (Logo etc.) at the beginning (optionally)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, analysis_interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

save_solution = SaveSolutionCallback(solution_interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

# TODO: Taal, IO
# restart_interval = 10

# TODO: Taal, AMR
# amr_interval = 5
# amr_callback = AMRCallback(amr_interval=5, amr_indicator=amr_indicator_gauss)

stepsize_callback = StepsizeCallback(cfl=1.6)

callbacks = CallbackSet(stepsize_callback, analysis_callback, save_solution, alive_callback)


sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);

nothing
