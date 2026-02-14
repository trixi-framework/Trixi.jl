using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

gamma() = 1.4
equations = CompressibleEulerEquations2D(gamma())

Re = 6.5 * 10^6
airfoil_cord_length = 1.0

# See https://www1.grc.nasa.gov/wp-content/uploads/case_c2.1.pdf or
# https://cfd.ku.edu/hiocfd/case_c2.2.html
U_inf() = 0.734 # Mach_inf = 1.0
rho_inf() = gamma() # => p_inf = 1.0

mu() = rho_inf() * U_inf() * airfoil_cord_length / Re
prandtl_number() = 0.71
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

aoa() = deg2rad(2.79) # 2.79 Degree angle of attack

@inline function initial_condition_mach085_flow(x, t, equations)
    v1 = 0.73312995164809
    v2 = 0.03572777625978245

    prim = SVector(1.4, v1, v2, 1.0)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach085_flow

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 1.0,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)

# In non-blended/limited regions, we use the cheaper weak form volume integral
volume_integral_default = VolumeIntegralWeakForm()

surface_flux = flux_hllc
volume_flux = flux_ranocha
# For the blended/limited regions, we need to supply compatible high-order and low-order volume integrals.
volume_integral_blend_high_order = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_blend_low_order = VolumeIntegralPureLGLFiniteVolumeO2(basis;
                                                                      volume_flux_fv = surface_flux,
                                                                      reconstruction_mode = reconstruction_O2_inner,
                                                                      slope_limiter = minmod)

volume_integral = VolumeIntegralShockCapturingHGType(shock_indicator;
                                                     volume_integral_default = volume_integral_default,
                                                     volume_integral_blend_high_order = volume_integral_blend_high_order,
                                                     volume_integral_blend_low_order = volume_integral_blend_low_order)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

###############################################################################

# mesh downloaded from https://cfd.ku.edu/hiocfd/rae2822/
mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/373727b8bc43e4aaeb63a6fcea77f098/raw/99cfd7c6b35df1a28d11db71be4b7702522cc84f/rae2822_level3.inp",
                           joinpath(@__DIR__, "rae2822_level3.inp"))

boundary_symbols = [:WallBoundary, :FarfieldBoundary]
mesh = P4estMesh{2}(mesh_file, boundary_symbols = boundary_symbols)

boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

boundary_conditions_hyp = (; FarfieldBoundary = boundary_condition_free_stream,
                           WallBoundary = boundary_condition_slip_wall)

boundary_conditions_para = (; FarfieldBoundary = boundary_condition_free_stream,
                            WallBoundary = boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions_hyp,
                                                                    boundary_conditions_para))

###############################################################################
# ODE solvers, callbacks etc.

t_c = airfoil_cord_length / U_inf() # convective time
tspan = (0.0, 25 * t_c)
tspan = (0.0, 5e-5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

save_sol_interval = 50_000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true)

force_boundary_names = (:WallBoundary,)
drag_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           DragCoefficientPressure2D(aoa(), rho_inf(),
                                                                     U_inf(),
                                                                     airfoil_cord_length))
lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure2D(aoa(), rho_inf(),
                                                                     U_inf(),
                                                                     airfoil_cord_length))

analysis_callback = AnalysisCallback(semi, interval = save_sol_interval,
                                     output_directory = "out",
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient))

alive_callback = AliveCallback(alive_interval = 500)

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, save_restart)

###############################################################################
# run the simulation

ode_algorithm = SSPRK43(thread = Trixi.True())

tols = 1e-4
sol = solve(ode, ode_algorithm;
            abstol = tols, reltol = tols, dt = 1e-6,
            maxiters = Inf,
            ode_default_options()..., callback = callbacks)
