# adapted elixir_euler_NACA0012airfoil_mach085

using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

p_inf() = 1.0
rho_inf() = p_inf() / (1.0 * 287.87) # p_inf = 1.0,  T = 1, R = 287.87
mach_inf() = 0.85
aoa() = pi / 180.0 # 1 Degree angle of attack
c_inf(equations) = sqrt(equations.gamma * p_inf() / rho_inf())
u_inf(equations) = mach_inf() * c_inf(equations)

# Leave `equations` unspecified here to enable usage of `BoundaryConditionDirichlet(initial_condition)`
# in the "elixir_navierstokes_NACA0012airfoil_mach085_restart.jl" which includes this elixir to
# demonstrate restarting/initializing a hyperbolic-parabolic simulation from a purely hyperbolic simulation.
@inline function initial_condition_mach085_flow(x, t, equations)
    v1 = u_inf(equations) * cos(aoa())
    v2 = u_inf(equations) * sin(aoa())

    prim = SVector(rho_inf(), v1, v2, p_inf())
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach085_flow

volume_flux = flux_ranocha
surface_flux = flux_lax_friedrichs

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)

#               volume_integral = volume_integral)

mesh_file = "airfoil_step.msh"

mesh = T8codeMesh(mesh_file, 2; polydeg = 1,
                  initial_refinement_level = 1)

@inline function boundary_condition_subsonic_constant(u_inner,
                                                      normal_direction::AbstractVector, x,
                                                      t,
                                                      surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_mach085_flow(x, t, equations)

    return flux_hll(u_inner, u_boundary, normal_direction, equations)
end

boundary_conditions = (; all = boundary_condition_subsonic_constant)

# TODO: somehow get the boundary symbols and set slip wall conditions for the airfoil

#                      Right = boundary_condition_subsonic_constant,
#                      Top = boundary_condition_subsonic_constant,
#                      Bottom = boundary_condition_subsonic_constant,
#                      AirfoilBottom = boundary_condition_slip_wall,
#                      AirfoilTop = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers

# Run for a long time to reach a steady state
# TODO: 20?
tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 2000

l_inf = 1.0 # Length of airfoil

force_boundary_names = (:AirfoilBottom, :AirfoilTop)
drag_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           DragCoefficientPressure2D(aoa(), rho_inf(),
                                                                     u_inf(equations),
                                                                     l_inf))

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure2D(aoa(), rho_inf(),
                                                                     u_inf(equations),
                                                                     l_inf))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true)
                                     #analysis_integrals = (drag_coefficient,
                                     #                      lift_coefficient))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 500,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out_airfoil")

stepsize_callback = StepsizeCallback(cfl = 1.0)

mesh_deformation_callback = MeshDeformationCallback(interval = 50)

# TODO
# visualization

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution, stepsize_callback, mesh_deformation_callback)

###############################################################################
# run the simulation
sol = solve(ode, SSPRK54(thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
