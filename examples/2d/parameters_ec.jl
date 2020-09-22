# TODO: Taal refactor, rename to
# - euler_ec.jl
# or something similar?

using DiffEqCallbacks
using OrdinaryDiffEq
using Trixi

equations = CompressibleEulerEquations2D(1.4)

initial_conditions = initial_conditions_weak_blast_wave

surface_flux = flux_chandrashekar
volume_flux  = flux_chandrashekar
solver = DGSEM(3, surface_flux, VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-2, -2)
coordinates_max = ( 2,  2)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=10_000)


semi = Semidiscretization(mesh, equations, initial_conditions, solver)

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

# TODO: Taal implement, printing stuff (Logo etc.) at the beginning (optionally)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.0)

# TODO: Taal, IO
# # save_initial_solution = false
# solution_interval = 100
# solution_variables = "primitive"
# restart_interval = 10

# TODO: Taal, restart
# restart = true
# restart_filename = "out/restart_000100.h5"
callbacks = CallbackSet(stepsize_callback, analysis_callback, alive_callback)


sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);
# sol = solve(ode, Tsit5(), save_everystep=false, callback=callbacks);

nothing
