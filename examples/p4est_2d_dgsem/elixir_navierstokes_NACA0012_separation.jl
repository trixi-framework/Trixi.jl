using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

gamma() = 1.4
equations = CompressibleEulerEquations2D(gamma())

Re = 10^6
airfoil_cord_length = 1.0

U_inf() = 0.85
rho_inf() = gamma() # => p_inf = 1.0

mu() = rho_inf() * U_inf() * airfoil_cord_length / Re
prandtl_number() = 0.72
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

p_inf() = 1.0
mach_inf() = U_inf()
aoa() = deg2rad(14.0) # 14 Degree angle of attack

@inline function initial_condition_mach085_flow(x, t, equations)
    v1 = 0.824751367334597   # 0.85 * cos(aoa())
    v2 = 0.20563361125971757 # 0.85 * sin(aoa())

    prim = SVector(1.4, v1, v2, 1.0)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach085_flow

surface_flux = flux_hll
volume_flux = flux_ranocha

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 1.0,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral_stabilized = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

volume_integral = VolumeIntegralAdaptive(volume_integral_default = VolumeIntegralWeakForm(),
                                         volume_integral_stabilized = volume_integral_stabilized,
                                         indicator = nothing) # Indicator taken from `volume_integral_stabilized`

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral_stabilized) # Just using FD actually crashes for this configuration!

#mesh_file = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/C1.2_quadgrids/naca_ref2_quadr_relabel.inp"
mesh_file = "/storage/home/daniel/Meshes/HighOrderCFDWorkshop/C1.2_quadgrids/naca_ref2_quadr_relabel.inp"

boundary_symbols = [:Airfoil, :Inflow, :Outflow]
mesh = P4estMesh{2}(mesh_file, boundary_symbols = boundary_symbols, initial_refinement_level = 1)

bc_farfield = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(:Inflow => bc_farfield,
                           :Outflow => bc_farfield,
                           :Airfoil => boundary_condition_slip_wall)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

boundary_conditions_para = Dict(:Inflow => bc_farfield,
                                :Outflow => bc_farfield,
                                :Airfoil => boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_para))

###############################################################################
# ODE solvers

t_c = airfoil_cord_length / U_inf()
tspan = (0.0, 25.0 * t_c)

ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

save_sol_interval = 5000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true)
#=
l_inf = 1.0 # Length of airfoil
force_boundary_names = (:Airfoil,)
drag_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           DragCoefficientPressure2D(aoa(), rho_inf(),
                                                                   U_inf(), l_inf))

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure2D(aoa(), rho_inf(),
                                                                   U_inf(), l_inf))
pressure_coefficient = AnalysisSurfacePointwise(force_boundary_names,
                                                SurfacePressureCoefficient(p_inf(), rho_inf(),
                                                                           U_inf(), l_inf))
=#
analysis_interval = 400                                                                     
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (),
                                     #=
                                     save_analysis = true,
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient),
                                     analysis_pointwise = (pressure_coefficient,)
                                     =#
                                    )

alive_callback = AliveCallback(alive_interval = 50)

amr_indicator = shock_indicator
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.05, # 1
                                      max_level = 3, max_threshold = 0.1)  # 3

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 40)

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, 
                        alive_callback,
                        save_solution,
                        save_restart,
                        amr_callback,
                        )

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(thread = Trixi.True());
            ode_default_options()..., callback = callbacks)