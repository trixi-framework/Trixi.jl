using OrdinaryDiffEqStabilizedRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

function initial_condition_const(x, t, equations)
    RealT = eltype(x)
    rho = 1
    rho_v1 = convert(RealT, 0.1)
    rho_v2 = convert(RealT, -0.2)
    rho_v3 = convert(RealT, 0.7)
    rho_e = 10
    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e)
end
initial_condition = initial_condition_const

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)
solver_parabolic = ViscousFormulationBassiRebay1()

mu() = 0.5
prandtl_number() = 0.72
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

default_mesh_file = joinpath(@__DIR__, "mesh_cube_with_boundaries.inp")
isfile(default_mesh_file) ||
    Trixi.download("https://gist.githubusercontent.com/DanielDoehring/710eab379fe3042dc08af6f2d1076e49/raw/38e9803bc0dab9b32a61d9542feac5343c3e6f4b/mesh_cube_with_boundaries.inp",
                   default_mesh_file)
mesh_file = default_mesh_file

boundary_symbols = [:PhysicalSurface1, :PhysicalSurface2]

mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, initial_refinement_level = 0,
                    boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:PhysicalSurface1 => BoundaryConditionDirichlet(initial_condition),
                           :PhysicalSurface2 => BoundaryConditionDirichlet(initial_condition))

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition,
                                             solver; solver_parabolic,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, ROCK4(max_stages = 8); adaptive = false, dt = 1e-3,
            ode_default_options()..., callback = callbacks);
