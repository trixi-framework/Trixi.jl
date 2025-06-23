using Trixi

###############################################################################
# This example shows that one can restart a hyperbolic-parabolic simulation from 
# a purely hyperbolic simulation/restart file.

base_elixir = "elixir_euler_NACA0012airfoil_mach085.jl"
trixi_include(@__MODULE__, joinpath(@__DIR__, base_elixir),
              tspan = (0.0, 0.005))

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

Re() = 50000.0
airfoil_cord_length() = 1.0
mu() = rho_inf() * u_inf(equations) * airfoil_cord_length() / Re()

prandtl_number() = 0.72

equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

###############################################################################

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

boundary_conditions_hyp = Dict(:Left => boundary_condition_free_stream,
                               :Right => boundary_condition_free_stream,
                               :Top => boundary_condition_free_stream,
                               :Bottom => boundary_condition_free_stream,
                               :AirfoilBottom => boundary_condition_slip_wall,
                               :AirfoilTop => boundary_condition_slip_wall)

boundary_conditions_para = Dict(:Left => boundary_condition_free_stream,
                                :Right => boundary_condition_free_stream,
                                :Top => boundary_condition_free_stream,
                                :Bottom => boundary_condition_free_stream,
                                :AirfoilBottom => boundary_condition_airfoil,
                                :AirfoilTop => boundary_condition_airfoil)

restart_file = "restart_000000252.h5"
restart_filename = joinpath("out", restart_file)
mesh = load_mesh(restart_filename)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions_hyp,
                                                                    boundary_conditions_para))

###############################################################################

tspan = (0.0, 10.0)
dt_restart = load_dt(restart_filename)
ode = semidiscretize(semi, tspan, restart_filename)

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient))

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 1,
                                      med_level = 3, med_threshold = 0.05,
                                      max_level = 4, max_threshold = 0.1)

amr_interval = 100
amr_callback = AMRCallback(semi, amr_controller,
                           interval = amr_interval,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution, save_restart,
                        stepsize_callback, amr_callback)

###############################################################################

sol = solve(ode, SSPRK54(thread = Trixi.True());
            dt = dt_restart,
            ode_default_options()..., callback = callbacks);
