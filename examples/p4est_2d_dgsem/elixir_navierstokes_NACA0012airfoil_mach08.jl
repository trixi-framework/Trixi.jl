
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

# Laminar transonic flow around a NACA0012 airfoil.

# This test is taken from the paper of Swanson and Langer. The values for the drag and lift coefficients
# from Case 5 in Table 3 are used to validate the scheme and computation of surface forces.

# References:
# - Roy Charles Swanson, Stefan Langer (2016)
#   Structured and Unstructured Grid Methods (2016)
#   [https://ntrs.nasa.gov/citations/20160003623] (https://ntrs.nasa.gov/citations/20160003623)
# - Deep Ray, Praveen Chandrashekar (2017)
#   An entropy stable finite volume scheme for the
#   two dimensional Navier–Stokes equations on triangular grids
#   [DOI:10.1016/j.amc.2017.07.020](https://doi.org/10.1016/j.amc.2017.07.020)

equations = CompressibleEulerEquations2D(1.4)

prandtl_number() = 0.72
mu() = 0.0031959974968701088
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

rho_inf() = 1.0
p_inf() = 2.85
aoa() = 10.0 * pi / 180.0 # 10 degree angle of attack
l_inf() = 1.0
mach_inf() = 0.8
u_inf(equations) = mach_inf() * sqrt(equations.gamma * p_inf() / rho_inf())
@inline function initial_condition_mach08_flow(x, t, equations)
    # set the freestream flow parameters
    gamma = equations.gamma
    u_inf = mach_inf() * sqrt(gamma * p_inf() / rho_inf())

    v1 = u_inf * cos(aoa())
    v2 = u_inf * sin(aoa())

    prim = SVector(rho_inf(), v1, v2, p_inf())
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach08_flow

surface_flux = flux_lax_friedrichs

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)

mesh_file = Trixi.download("https://gist.githubusercontent.com/Arpit-Babbar/339662b4b46164a016e35c81c66383bb/raw/8bf94f5b426ba907ace87405cfcc1dcc2ef7cbda/NACA0012.inp",
                           joinpath(@__DIR__, "NACA0012.inp"))

mesh = P4estMesh{2}(mesh_file, initial_refinement_level = 1)

# The boundary values across outer boundary are constant but subsonic, so we cannot compute the
# boundary flux from the external information alone. Thus, we use the numerical flux to distinguish
# between inflow and outflow characteristics
@inline function boundary_condition_subsonic_constant(u_inner,
                                                      normal_direction::AbstractVector, x,
                                                      t,
                                                      surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_mach08_flow(x, t, equations)

    return Trixi.flux_hll(u_inner, u_boundary, normal_direction, equations)
end

boundary_conditions = Dict(:Left => boundary_condition_subsonic_constant,
                           :Right => boundary_condition_subsonic_constant,
                           :Top => boundary_condition_subsonic_constant,
                           :Bottom => boundary_condition_subsonic_constant,
                           :AirfoilBottom => boundary_condition_slip_wall,
                           :AirfoilTop => boundary_condition_slip_wall)

velocity_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))

heat_airfoil = Adiabatic((x, t, equations) -> 0.0)

boundary_conditions_airfoil = BoundaryConditionNavierStokesWall(velocity_airfoil,
                                                                heat_airfoil)

function momenta_initial_condition_mach08_flow(x, t, equations)
    u = initial_condition_mach08_flow(x, t, equations)
    momenta = SVector(u[2], u[3])
end
velocity_bc_square = NoSlip((x, t, equations) -> momenta_initial_condition_mach08_flow(x, t,
                                                                                       equations))

heat_bc_square = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_square = BoundaryConditionNavierStokesWall(velocity_bc_square,
                                                              heat_bc_square)

boundary_conditions_parabolic = Dict(:Left => boundary_condition_square,
                                     :Right => boundary_condition_square,
                                     :Top => boundary_condition_square,
                                     :Bottom => boundary_condition_square,
                                     :AirfoilBottom => boundary_conditions_airfoil,
                                     :AirfoilTop => boundary_conditions_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers

# Run for a long time to reach a state where forces stabilize up to 3 digits
tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 2000

force_boundary_names = (:AirfoilBottom, :AirfoilTop)
drag_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           DragCoefficientPressure(aoa(), rho_inf(),
                                                                   u_inf(equations),
                                                                   l_inf()))

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure(aoa(), rho_inf(),
                                                                   u_inf(equations),
                                                                   l_inf()))

drag_coefficient_shear_force = AnalysisSurfaceIntegral(force_boundary_names,
                                                       DragCoefficientShearStress(aoa(),
                                                                                  rho_inf(),
                                                                                  u_inf(equations),
                                                                                  l_inf()))

lift_coefficient_shear_force = AnalysisSurfaceIntegral(force_boundary_names,
                                                       LiftCoefficientShearStress(aoa(),
                                                                                  rho_inf(),
                                                                                  u_inf(equations),
                                                                                  l_inf()))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient,
                                                           drag_coefficient_shear_force,
                                                           lift_coefficient_shear_force))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 500,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(thread = OrdinaryDiffEq.True()); abstol = 1e-8,
            reltol = 1e-8,
            ode_default_options()..., callback = callbacks)
summary_callback() # print the timer summary
