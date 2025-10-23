using OrdinaryDiffEqLowStorageRK
using Trixi

advection_velocity() = 1.5
equations = LinearScalarAdvectionEquation1D(advection_velocity())
diffusivity() = 1e-2
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_godunov)

initial_condition_zero(x, t, equations::LinearScalarAdvectionEquation1D) = SVector(0.0)
initial_condition = initial_condition_zero

coordinates_min = (-1.0,)
coordinates_max = (1.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = false,
                n_cells_max = 30_000)

# BC types
boundary_condition_left = BoundaryConditionDirichlet((x, t, equations_parabolic) -> 1.0)
boundary_condition_neumann_zero = BoundaryConditionNeumann((x, t, equations_parabolic) -> 0.0)

# define inviscid boundary conditions
boundary_conditions = (; x_neg = boundary_condition_left,
                       x_pos = boundary_condition_do_nothing)

# define viscous boundary conditions
boundary_conditions_parabolic = (; x_neg = boundary_condition_left,
                                 x_pos = boundary_condition_neumann_zero)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_indicator = IndicatorLöhner(semi, variable = first)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 3,
                                      med_level = 4, med_threshold = 0.1,
                                      max_level = 6, max_threshold = 0.15)
amr_interval = 50
amr_callback = AMRCallback(semi, amr_controller,
                           interval = amr_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        amr_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
