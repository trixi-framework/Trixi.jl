
using OrdinaryDiffEq
using Trixi

advectionvelocity = (1.0, 1.0)
# advectionvelocity = (0.2, -0.3)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

initial_conditions = initial_conditions_convergence_test

surface_flux = flux_lax_friedrichs
solver = DGSEM(3, surface_flux)

coordinates_min = (-1, -1)
coordinates_max = ( 1,  1)
# refinement_patches = (
#   (type="box", coordinates_min=(-0.5, -0.5), coordinates_max=(0.5, 0.5)),
#   (type="box", coordinates_min=(-0.1, -0.1), coordinates_max=(0.1, 0.1)),
# )
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000)


semi = Semidiscretization(mesh, equations, initial_conditions, solver)

# TODO: Taal implement, printing stuff at the beginning (optionally)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

sol = solve(ode, Tsit5(), save_everystep=false)

# TODO: Taal, restart
# restart = true
# restart_filename = "out/restart_000100.h5"

# TODO: Taal, CFL
# cfl = 0.8
# n_steps_max = 10000

# TODO: Taal, analysis
# analysis_interval = 100
# extra_analysis_quantities = ["entropy", "energy_total"]

# TODO: Taal, IO
# # save_initial_solution = false
# solution_interval = 100
# solution_variables = "primitive"
# restart_interval = 10
