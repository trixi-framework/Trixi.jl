
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_constant

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

default_mesh_file = joinpath(@__DIR__, "cube_boundaries.inp")
isfile(default_mesh_file) ||
    download("https://gist.github.com/DanielDoehring/eefe73ae5d113bc3944a518b6e88e663/raw/359a58a808790f3c3efc050273270eb1cc8ee353/cube_boundaries.inp",
             default_mesh_file)
mesh_file = default_mesh_file

boundary_symbols = [:PhysicalSurface1]

# Map the unstructured mesh with the mapping above
mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:PhysicalSurface1 => BoundaryConditionDirichlet(initial_condition))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
