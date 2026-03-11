using Trixi
using OrdinaryDiffEqSSPRK
using LinearAlgebra: norm
using Downloads

###############################################################################
# semidiscretization of the compressible Euler equations

### Inviscid transonic flow over the 3rd Drag Prediction Workshop (DPW) wing ###

# For details, see the adaptation for the 1st International Workshop on High-Order CFD Methods
# https://cfd.ku.edu/hiocfd/case_c3.2.html

gamma() = 1.4
equations = CompressibleEulerEquations3D(gamma())

rho_inf() = 1.293 # [kg/m^3]
p_inf() = 101325 # [Pa]

c_inf() = sqrt(gamma() * p_inf()/rho_inf()) # [m/s]

Ma_inf() = 0.76
U_inf() = Ma_inf() * c_inf() # [m/s]

alpha() = deg2rad(0.5)
Ux_inf() = U_inf() * cos(alpha()) # [m/s]
Uz_inf() = U_inf() * sin(alpha()) # [m/s]

@inline function initial_condition(x, t, equations::CompressibleEulerEquations3D)
    prim = SVector(rho_inf(), Ux_inf(), 0.0, Uz_inf(), p_inf())
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

# Ensure that rho and p are the same across symmetry line and allow only
# tangential velocity.
# Somewhat naive implementation of `boundary_condition_slip_wall`.
# Used here to avoid confusion between wing (wall) and symmetry plane.
@inline function bc_symmetry(u_inner, normal_direction::AbstractVector, x, t,
                             surface_flux_function,
                             equations::CompressibleEulerEquations3D)
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute the primitive variables
    rho, v1, v2, v3, p = cons2prim(u_inner, equations)

    v_normal = normal[1] * v1 + normal[2] * v2 + normal[3] * v3

    u_mirror = prim2cons(SVector(rho,
                                 v1 - 2 * v_normal * normal[1],
                                 v2 - 2 * v_normal * normal[2],
                                 v3 - 2 * v_normal * normal[3],
                                 p), equations)

    flux = surface_flux_function(u_inner, u_mirror, normal, equations) * norm_

    return flux
end

polydeg = 2
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)

volume_integral_default = VolumeIntegralWeakForm()

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

volume_integral_blend_high_order = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_blend_low_order = VolumeIntegralPureLGLFiniteVolume(surface_flux)

volume_integral = VolumeIntegralShockCapturingHGType(shock_indicator;
                                                     volume_integral_default = volume_integral_default,
                                                     volume_integral_blend_high_order = volume_integral_blend_high_order,
                                                     volume_integral_blend_low_order = volume_integral_blend_low_order)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

# Mesh taken from https://cfd.ku.edu/hiocfd/dpw/dpw_w1_cgrid_q1.msh.gz, adapted for Trixi & P4est by Daniel Doehring
mesh_file = joinpath(@__DIR__, "DPW3_Wing.inp")
Downloads.download("https://zenodo.org/records/18953457/files/DPW3_Wing.inp?download=1",
                   mesh_file)

# Boundary symbols follow from nodesets in the mesh file
boundary_symbols = [:Symmetry, :Farfield, :Wing]
mesh = P4estMesh{3}(mesh_file; polydeg = 1, boundary_symbols = boundary_symbols)

boundary_conditions = (; Symmetry = bc_symmetry, # Could use `boundary_condition_slip_wall` here as well
                       Farfield = bc_farfield,
                       Wing = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

t_convective = 100.0 # simulate for 100 convective time units
chord_length = 0.197556 # [m]
t_end = t_convective * chord_length / U_inf() # [s]

tspan = (0.0, t_end)

ode = semidiscretize(semi, tspan)

###############################################################################

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[])

alive_callback = AliveCallback(alive_interval = 2)

save_sol_interval = 1000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out/")

save_restart = SaveRestartCallback(interval = save_sol_interval)

stepsize = StepsizeCallback(cfl = 3.1)

callbacks = CallbackSet(summary_callback,
                        stepsize,
                        alive_callback, analysis_callback,
                        save_solution, save_restart)

###############################################################################

sol = solve(ode, SSPRK43(thread = Trixi.True());
            adaptive = false, dt = 1.0, # needed for CFL-based timestepping
            ode_default_options()..., callback = callbacks);
