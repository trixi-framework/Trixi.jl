using OrdinaryDiffEqStabilizedRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

function initial_condition_const(x, t, equations)
    RealT = eltype(x)
    rho = 1
    rho_v1 = convert(RealT, 0.1)
    rho_v2 = convert(RealT, -0.2)
    rho_e = 10
    return SVector(rho, rho_v1, rho_v2, rho_e)
end
initial_condition = initial_condition_const

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))
solver_parabolic = ViscousFormulationLocalDG()

mu() = 0.5
prandtl_number() = 0.72
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

# Mapping as described in https://arxiv.org/abs/2012.12040 but reduced to 2D
function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5

    y = eta + 3 / 8 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
                       cos(0.5 * pi * (2 * eta - 3) / 3))

    x = xi + 3 / 8 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
                      cos(2 * pi * (2 * y - 3) / 3))

    return SVector(x, y)
end

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

# Unstructured mesh with 48 cells of the square domain [-1, 1]^n
mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/a075f8ec39a67fa9fad8f6f84342cbca/raw/a7206a02ed3a5d3cadacd8d9694ac154f9151db7/square_unstructured_1.inp",
                           joinpath(@__DIR__, "square_unstructured_1.inp"))

# Map the unstructured mesh with the mapping above
mesh = P4estMesh{2}(mesh_file, mapping = mapping, polydeg = polydeg)

boundary_conditions = Dict(:all => BoundaryConditionDirichlet(initial_condition))

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition,
                                             solver; solver_parabolic,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, ROCK4(max_stages = 10); adaptive = false, dt = 1e-3,
            ode_default_options()..., callback = callbacks);
