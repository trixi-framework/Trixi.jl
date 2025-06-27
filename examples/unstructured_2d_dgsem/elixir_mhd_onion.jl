using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Vertical magnetic streamlines with a bend in the center.
# This makes the streamlines look like a section of an onion.

# Define the initial condition as a vertical field with a bend in the center.
function initial_condition_onion(x, t, equations::IdealGlmMhdEquations2D)
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    p = rho^equations.gamma
    B1 = -2 * x[1] * x[2] * exp(-x[1]^2 - x[2]^2)
    B2 = (2 * x[1]^2 - 1) * exp(-x[1]^2 - x[2]^2) + 1
    B3 = 0.0
    psi = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

initial_condition = initial_condition_onion

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
mesh_file = Trixi.download("https://gist.githubusercontent.com/patrickersing/8fe7e61bf1e79eeafa031ec2e355803d/raw/7b26d86f15a419152a505934ae742457826d99b5/mesh_mhd_onion.mesh",
                           joinpath(@__DIR__, "mesh_mhd_onion.mesh"))
mesh = UnstructuredMesh2D(mesh_file, periodicity = false)

surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

boundary_conditions = Dict(:Bottom => BoundaryConditionDirichlet(initial_condition),
                           :Top => BoundaryConditionDirichlet(initial_condition),
                           :Right => BoundaryConditionDirichlet(initial_condition),
                           :Left => BoundaryConditionDirichlet(initial_condition))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

analysis_callback = AnalysisCallback(semi, interval = 100)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 1.0

stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 0.01, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
