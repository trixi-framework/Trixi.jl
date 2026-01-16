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

volume_integral_stabilized = VolumeIntegralShockCapturingRRG(basis, shock_indicator;
                                                             volume_flux_dg = volume_flux,
                                                             volume_flux_fv = surface_flux,
                                                             slope_limiter = minmod)

volume_integral = VolumeIntegralAdaptive(volume_integral_default = VolumeIntegralWeakForm(),
                                         volume_integral_stabilized = volume_integral_stabilized,
                                         indicator = nothing) # Indicator taken from `volume_integral_stabilized`

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral) # Just using FD actually crashes for this configuration!

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

# Get quadratic meshfile
mesh_file_name = "SD7003_2D_Quadratic.inp"

mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/bd2aa20f7e6839848476a0e87ede9f69/raw/1bc8078b4a57634819dc27010f716e68a225c9c6/SD7003_2D_Quadratic.inp",
                           joinpath(@__DIR__, mesh_file_name))

# There is also a linear mesh file available at
# https://gist.githubusercontent.com/DanielDoehring/375df933da8a2081f58588529bed21f0/raw/18592aa90f1c86287b4f742fd405baf55c3cf133/SD7003_2D_Linear.inp

boundary_symbols = [:Airfoil, :FarField]
mesh = P4estMesh{2}(mesh_file, boundary_symbols = boundary_symbols)

restart_filename = "out/restart_000960622.h5"
mesh = load_mesh(restart_filename)

boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

boundary_conditions_hyp = Dict(:FarField => boundary_condition_free_stream,
                               :Airfoil => boundary_condition_slip_wall)

boundary_conditions_para = Dict(:FarField => boundary_condition_free_stream,
                                :Airfoil => boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions_hyp,
                                                                    boundary_conditions_para))

###############################################################################
# ODE solvers, callbacks etc.

t_c = airfoil_cord_length / U_inf()

tspan = (0.0, 50 * t_c) # Non-AMR
#ode = semidiscretize(semi, tspan)

tspan = (load_time(restart_filename), 65 * t_c) # 65 at restart file
ode = semidiscretize(semi, tspan, restart_filename)


summary_callback = SummaryCallback()

save_sol_interval = 100_000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = true,
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

alive_callback = AliveCallback(alive_interval = 100)

amr_indicator = shock_indicator

@inline function v1(u, equations::CompressibleEulerEquations2D)
    rho, rho_v1, _, _ = u
    return rho_v1 / rho
end
amr_indicator = IndicatorLöhner(semi, variable = v1)

amr_controller = ControllerThreeLevelCombined(semi, amr_indicator, shock_indicator,
                                              base_level = 0,
                                              med_level = 1, med_threshold = 0.1,
                                              max_level = 2, max_threshold = 0.3,
                                              max_threshold_secondary = shock_indicator.alpha_max)                                      

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 100,
                           adapt_initial_condition = false)

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, 
                        alive_callback,
                        amr_callback,
                        #save_solution,
                        #save_restart
                        )

###############################################################################
# run the simulation

ode_algorithm = SSPRK43(thread = Trixi.True())

tols = 5e-5 # Not sure if low or high tols lead to better performance here
sol = solve(ode, ode_algorithm;
            abstol = tols, reltol = tols, dt = 4e-6,
            maxiters = Inf,
            ode_default_options()..., callback = callbacks)