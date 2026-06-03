using OrdinaryDiffEqLowStorageRK
using Trixi
using Plots

# Advection test case following
# Randall J. LeVeque
# High-Resolution Conservative Algorithms for Advection in Incompressible Flow
# https://doi.org/10.1137/0733033

############################################################################################
# initial condition

equations = LinearVariableScalarAdvectionEquation2D()

function initial_condition_advected_objects(x, t,
                                            equations::LinearVariableScalarAdvectionEquation2D)
    RealT = eltype(x)

    # smooth hump
    x_0, y_0, r_0 = 0.25f0, 0.5f0, convert(RealT, 0.15)
    r = sqrt((x[1] - x_0)^2 + (x[2] - y_0)^2)
    r = min(r, r_0) / r_0
    hump = 0.25f0 * (1 + cospi(r))

    # cone
    x_1, y_1, r_1 = 0.5f0, 0.25f0, convert(RealT, 0.15)
    r = sqrt((x[1] - x_1)^2 + (x[2] - y_1)^2)
    cone = 1.0f0 - min(r, r_1) / r_1

    # slotted disc
    x_2, y_2, r_2 = 0.5f0, 0.75f0, convert(RealT, 0.15)
    w, l = 0.05f0, convert(RealT, 0.25)
    r = sqrt((x[1] - x_2)^2 + (x[2] - y_2)^2)
    disc = 0
    if r <= r_2 && (x[2] >= y_2 - r_2 + l || abs(x[1] - x_2) >= w)
        disc = 1.0f0
    end

    return SVector(hump + cone + disc)
end

# velocity field at time t = 0 (see reference)
@inline function velocity_swirling(x, equations)
    u = sinpi(x[1])^2 * sinpi(2 * x[2])
    v = -sinpi(x[2])^2 * sinpi(2 * x[1])
    return SVector(u, v)
end

############################################################################################
# semidiscretization
polydeg = 3

initial_condition = initial_condition_advected_objects

surface_flux = flux_lax_friedrichs
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                periodicity = false)

# boundary conditions
boundary_condition_dirichlet = BoundaryConditionDirichlet(initial_condition_advected_objects)
boundary_conditions = (; x_neg = boundary_condition_dirichlet,
                       x_pos = boundary_condition_dirichlet,
                       y_neg = boundary_condition_dirichlet,
                       y_pos = boundary_condition_dirichlet)

# the velocity is passed as auxiliary_field into the cache
semi = SemidiscretizationHyperbolic(mesh,
                                    equations,
                                    initial_condition,
                                    solver,
                                    boundary_conditions = boundary_conditions,
                                    aux_field = velocity_swirling)

##############################################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
solution_variables = cons2prim

analysis_callback = AnalysisCallback(semi,
                                     interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 10,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     output_directory = "out_swirling",
                                     solution_variables = solution_variables)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 2,
                                      med_level = 4, med_threshold = 0.3,
                                      max_level = 6, max_threshold = 0.8)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 20,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

visualization = VisualizationCallback(semi; interval = 20) #, show_mesh = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        amr_callback,
                        #visualization,
                        save_solution)

###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, ode_default_options()..., callback = callbacks);
