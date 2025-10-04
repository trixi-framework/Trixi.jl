using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation

advection_velocity() = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity())
diffusivity() = 1e-2
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_godunov)

coordinates_min = (-1.0, )
coordinates_max = (0.0, )
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = false,
                n_cells_max = 30_000)

# Example setup taken from
# - Truman Ellis, Jesse Chan, and Leszek Demkowicz (2016).
#   Robust DPG methods for transient convection-diffusion.
#   In: Building bridges: connections and challenges in modern approaches
#   to numerical partial differential equations.
#   [DOI](https://doi.org/10.1007/978-3-319-41640-3_6).
function initial_condition_eriksson_johnson(x, t, equations)
    l = 4
    epsilon = diffusivity() # TODO: this requires epsilon < .6 due to sqrt
    lambda_1 = (-1 + sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
    lambda_2 = (-1 - sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
    u = exp(-l * t) * (exp(lambda_1 * x[1]) - exp(lambda_2 * x[1]))
    return SVector{1}(u)
end
initial_condition = initial_condition_eriksson_johnson

boundary_conditions = (; x_neg = BoundaryConditionDirichlet(initial_condition),
                       x_pos = boundary_condition_do_nothing)

boundary_conditions_parabolic = BoundaryConditionDirichlet(initial_condition)

semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ViscousFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_indicator = IndicatorLöhner(semi, variable = first)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 3,
                                      med_level = 4, med_threshold = 0.1,
                                      max_level = 6, max_threshold = 0.2)
amr_interval = 50
amr_callback = AMRCallback(semi, amr_controller,
                           interval = amr_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        amr_callback)

###############################################################################
# run the simulation

time_int_tol = 1.0e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
