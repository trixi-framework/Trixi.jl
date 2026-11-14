using OrdinaryDiffEqLowStorageRK
using Trixi
using SummationByParts # for the SBP operators for the Tetrahedra

###############################################################################
# semidiscretization of the linear advection-diffusion equation

diffusivity() = 5.0e-2
advection_velocity = (1.0, 0.0, 0.0)
equations = LinearScalarAdvectionEquation3D(advection_velocity)
equations_parabolic = LaplaceDiffusion3D(diffusivity(), equations)

dg = DGMulti(element_type = Tet(),
             approximation_type = SBP(),
             surface_flux = flux_lax_friedrichs,
             polydeg = 3)

coordinates_min = (-1.0, -0.5, -0.5)
coordinates_max = (0.0, 0.5, 0.5)

# tag the different boundary segments
x_neg(x, tol = 50 * eps()) = abs(x[1] - coordinates_min[1]) < tol
x_pos(x, tol = 50 * eps()) = abs(x[1] - coordinates_max[1]) < tol
y_neg(x, tol = 50 * eps()) = abs(x[2] - coordinates_min[2]) < tol
y_pos(x, tol = 50 * eps()) = abs(x[2] - coordinates_max[2]) < tol
z_neg(x, tol = 50 * eps()) = abs(x[3] - coordinates_min[3]) < tol
z_pos(x, tol = 50 * eps()) = abs(x[3] - coordinates_max[3]) < tol
is_on_boundary = (; x_neg, x_pos, y_neg, y_pos, z_neg, z_pos)

cells_per_dimension = (6, 4, 3)
mesh = DGMultiMesh(dg, cells_per_dimension;
                   coordinates_min, coordinates_max,
                   periodicity = false,
                   is_on_boundary)

# Example setup taken from
# - Truman Ellis, Jesse Chan, and Leszek Demkowicz (2016).
#   Robust DPG methods for transient convection-diffusion.
#   In: Building bridges: connections and challenges in modern approaches
#   to numerical partial differential equations.
#   [DOI](https://doi.org/10.1007/978-3-319-41640-3_6).
function initial_condition_eriksson_johnson(x, t, equations)
    l = 4
    epsilon = diffusivity() # NOTE: this requires epsilon <= 1/16 due to sqrt
    lambda_1 = (-1 + sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
    lambda_2 = (-1 - sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
    r1 = (1 + sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
    s1 = (1 - sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
    u = exp(-l * t) * (exp(lambda_1 * x[1]) - exp(lambda_2 * x[1])) +
        cos(pi * x[2]) * (exp(s1 * x[1]) - exp(r1 * x[1])) / (exp(-s1) - exp(-r1))
    return SVector{1}(u)
end
initial_condition = initial_condition_eriksson_johnson

boundary_conditions = (; x_neg = BoundaryConditionDirichlet(initial_condition),
                       y_neg = BoundaryConditionDirichlet(initial_condition),
                       z_neg = boundary_condition_do_nothing,
                       y_pos = BoundaryConditionDirichlet(initial_condition),
                       x_pos = boundary_condition_do_nothing,
                       z_pos = boundary_condition_do_nothing)

# A single boundary condition is applied to all (tagged) boundary segments
boundary_conditions_parabolic = BoundaryConditionDirichlet(initial_condition)

semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, dg;
                                             solver_parabolic = ParabolicFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1.0e-8
sol = solve(ode, RDPK3SpFSAL49(thread = Trixi.Threaded());
            abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
