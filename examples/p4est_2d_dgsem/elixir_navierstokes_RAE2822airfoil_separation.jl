using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

gamma() = 1.4
equations = CompressibleEulerEquations2D(gamma())

Re = 6.5 * 10^6
airfoil_cord_length = 1.0

# See https://www1.grc.nasa.gov/wp-content/uploads/case_c2.1.pdf
# Also https://arrow.utias.utoronto.ca/~myano/papers/yd_2012_how_c22_rae.pdf for Mach number plot
U_inf() = 0.734
rho_inf() = gamma() # => p_inf = 1.0

mu() = rho_inf() * U_inf() * airfoil_cord_length / Re
prandtl_number() = 0.71
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

p_inf() = 1.0
mach_inf() = U_inf()
aoa() = deg2rad(2.79) # 2.79 Degree angle of attack

@inline function initial_condition_mach085_flow(x, t, equations)
    v1 = 0.73312995164809
    v2 = 0.03572777625978245

    prim = SVector(1.4, v1, v2, 1.0)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach085_flow

surface_flux = flux_hllc
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
               volume_integral = volume_integral_stabilized)

###############################################################################
mesh_file = "/home/daniel/git/Paper_AdaptiveVolTerm/Data/RAE2822_Transonic/HiCFD_Meshes/rae2822_level3.inp"

boundary_symbols = [:WallBoundary, :FarfieldBoundary]
mesh = P4estMesh{2}(mesh_file, boundary_symbols = boundary_symbols)

restart_filename = "/home/daniel/git/Paper_AdaptiveVolTerm/Data/RAE2822_Transonic/restart_tc26_VTA.h5"

boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

boundary_conditions_hyp = Dict(:FarfieldBoundary => boundary_condition_free_stream,
                               :WallBoundary => boundary_condition_slip_wall)

boundary_conditions_para = Dict(:FarfieldBoundary => boundary_condition_free_stream,
                                :WallBoundary => boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions_hyp,
                                                                    boundary_conditions_para))

###############################################################################
# ODE solvers, callbacks etc.

t_c = airfoil_cord_length / U_inf()

tspan = (0.0, 25 * t_c)
ode = semidiscretize(semi, tspan)

tspan = (load_time(restart_filename), 26 * t_c) # timings for one tc
ode = semidiscretize(semi, tspan, restart_filename)

summary_callback = SummaryCallback()

save_sol_interval = 500_000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true)

l_inf = 1.0 # Length of airfoil
force_boundary_names = (:WallBoundary,)
drag_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           DragCoefficientPressure2D(aoa(), rho_inf(),
                                                                     U_inf(), l_inf))
lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure2D(aoa(), rho_inf(),
                                                                     U_inf(), l_inf))

analysis_interval = 100_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient))

alive_callback = AliveCallback(alive_interval = 500)

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, 
                        alive_callback,
                        #save_solution,
                        #save_restart
                        )

###############################################################################
# run the simulation

ode_algorithm = SSPRK43(thread = Trixi.True())

tols = 1e-4 # Not sure if low or high tols lead to better performance here
sol = solve(ode, ode_algorithm;
            abstol = tols, reltol = tols, dt = 3e-6,
            maxiters = Inf,
            ode_default_options()..., callback = callbacks)