using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

diffusivity() = 1e-3
equations_parabolic = LaplaceDiffusion3D(diffusivity(), equations)

# Solver with polydeg=4 to ensure free stream preservation (FSP) on non-conforming meshes.
# The polydeg of the solver must be at least twice as big as the polydeg of the mesh.
# See https://doi.org/10.1007/s10915-018-00897-9, Section 6.
solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

function Trixi.initial_condition_gauss(x, t, equations_parabolic::LaplaceDiffusion3D)
    # Store translated coordinate for easy use of exact solution
    @unpack equations_hyperbolic = equations_parabolic
    x_trans = x - equations_hyperbolic.advection_velocity * t

    scalar = exp(-(x_trans[1]^2 + x_trans[2]^2 + x_trans[3]^2))
    return SVector(scalar)
end

initial_condition = initial_condition_gauss
boundary_condition = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = (; all = boundary_condition)

# Mapping as described in https://arxiv.org/abs/2012.12040 but with less warping.
# The mapping will be interpolated at tree level, and then refined without changing
# the geometry interpolant. The original mapping applied to this unstructured mesh
# causes some Jacobians to be negative, which makes the mesh invalid.
function mapping(xi, eta, zeta)
    y = eta +
        1 / 4 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
         cos(0.5 * pi * (2 * eta - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    x = xi +
        1 / 4 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
         cos(2 * pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    z = zeta +
        1 / 4 * (cos(0.5 * pi * (2 * x - 3) / 3) *
         cos(pi * (2 * y - 3) / 3) *
         cos(0.5 * pi * (2 * zeta - 3) / 3))

    # Transform the weird deformed cube to be approximately the size of [-5,5]^3 to match IC
    return SVector(5 * x, 5 * y, 5 * z)
end

# Unstructured mesh with 48 cells of the cube domain [-1, 1]^3
mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/b8df0033798e4926dec515fc045e8c2c/raw/b9254cde1d1fb64b6acc8416bc5ccdd77a240227/cube_unstructured_2.inp",
                           joinpath(@__DIR__, "cube_unstructured_2.inp"))

# Mesh polydeg of 2 (half the solver polydeg) to ensure FSP (see above).
mesh = P4estMesh{3}(mesh_file, polydeg = 2,
                    mapping = mapping,
                    initial_refinement_level = 1)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ViscousFormulationLocalDG(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 4.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 1,
                                      med_level = 2, med_threshold = 0.1,
                                      max_level = 3, max_threshold = 0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
