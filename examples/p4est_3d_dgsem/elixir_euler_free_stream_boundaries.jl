
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

default_mesh_file = joinpath(@__DIR__, "mesh_cube_with_boundaries.inp")
isfile(default_mesh_file) ||
    Trixi.download("https://gist.githubusercontent.com/DanielDoehring/710eab379fe3042dc08af6f2d1076e49/raw/38e9803bc0dab9b32a61d9542feac5343c3e6f4b/mesh_cube_with_boundaries.inp",
                   default_mesh_file)
mesh_file = default_mesh_file

boundary_symbols = [:PhysicalSurface1, :PhysicalSurface2]

mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:PhysicalSurface1 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface2 => BoundaryConditionDirichlet(initial_condition))

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
