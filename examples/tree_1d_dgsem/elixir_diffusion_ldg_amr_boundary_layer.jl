using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Adaptive semidiscretization of the pure diffusion equation with mixed 
# Dirichlet-Neumann BCs and an initial condition with a boundary layer.

diffusivity = 0.25
amplitude = 1.0
boundary_layer_thickness = 0.01

equations = LinearDiffusionEquation1D(diffusivity)

solver = DGSEM(polydeg = 3)
solver_parabolic = ParabolicFormulationLocalDG()

mesh = TreeMesh((0.0,), (1.0,),
                initial_refinement_level = 0,
                periodicity = false,
                n_cells_max = 30_000)

function initial_condition_boundary_layer(x, t, equations)
    return SVector(amplitude * (1 - exp(-x[1] / boundary_layer_thickness)) /
                   (1 - exp(-1 / boundary_layer_thickness)))
end

initial_condition = initial_condition_boundary_layer
boundary_condition_dirichlet = BoundaryConditionDirichlet((x, t, equations) -> SVector(0.0))
boundary_condition_neumann = BoundaryConditionNeumann((x, t, equations) -> SVector(0.0))

boundary_conditions = (; x_neg = boundary_condition_dirichlet,
                       x_pos = boundary_condition_neumann)

semi = SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                   solver_parabolic = solver_parabolic,
                                   boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 200)

alive_callback = AliveCallback(analysis_interval = 200)

amr_indicator = IndicatorLöhner(semi, variable = first)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 3,
                                      med_threshold = 0.01,
                                      max_level = 6, max_threshold = 0.175)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 200,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = false)

stepsize_callback = StepsizeCallback(cfl_parabolic = 0.05)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), adaptive = false,
            ode_default_options()..., callback = callbacks, maxiters = 500_000)
