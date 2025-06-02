using Trixi
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

### Inviscid transonic flow over the ONERA M6 wing ###
# See for reference
# 
# https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html
# https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing01/m6wing01.html
#
# which are test cases for the viscous flow.
# 
# A tutorial for the invscid case can be found for the SU2 Code (https://github.com/su2code/SU2):
# https://su2code.github.io/tutorials/Inviscid_ONERAM6/

@inline function initial_condition(x, t, equations::CompressibleEulerEquations3D)
    # set the freestream flow parameters
    rho_freestream = 1.4

    # v_total = 0.84 = Mach (for c = 1)

    # AoA = 3.06
    v1 = 0.8388023121403883
    v2 = 0.0448406193973588
    v3 = 0.0

    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

# Ensure that rho and p are the same across symmetry line and allow only 
# tangential velocity.
# Somewhat naive implementation of `boundary_condition_slip_wall`.
# Used here to avoid confusion between wing (body) and symmetry plane.
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
surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

# Flux Differencing is required, shock capturing not!
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

# This is a sanitized mesh obtained from the original mesh available at
# https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing01/m6wing01.html
#
# which has been merged into a single gmsh mesh file by the HiSA team, 
# see the case description (for the viscous case, but the mesh is the same)
# https://hisa.gitlab.io/archive/nparc/oneraM6/notes/oneraM6.html
#
# The base mesh is available at
# https://gitlab.com/hisa/hisa/-/blob/master/examples/oneraM6/mesh/p3dMesh/m6wing.msh?ref_type=heads
#
# The sanitized, i.e., higher-order ready mesh was subsequently created by Daniel Doehring
# and is available at
mesh_file = Trixi.download("https://github.com/DanielDoehring/AerodynamicMeshes/raw/refs/heads/main/ONERA_M6_Wing/ONERA_M6_Wing_sanitized.inp",
                           joinpath(@__DIR__, "ONERA_M6_sanitized.inp"))

# Boundary symbols follow from nodesets in the mesh file
boundary_symbols = [:Symmetry, :FarField, :BottomWing, :TopWing]
mesh = P4estMesh{3}(mesh_file; polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:Symmetry => bc_symmetry, # Could use `boundary_condition_slip_wall` here as well
                           :FarField => bc_farfield,
                           :BottomWing => boundary_condition_slip_wall,
                           :TopWing => boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

# This is an extremely long (weeks!) simulation
tspan = (0.0, 6.0)
ode = semidiscretize(semi, tspan)

###############################################################################

summary_callback = SummaryCallback()

force_boundary_names = (:BottomWing, :TopWing)

aoa() = deg2rad(3.06)

rho_inf() = 1.4
u_inf(equations) = 0.84

### Wing projected area calculated from geometry information provided at ###
### https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html ###

#height = 1.1963
height = 1.0 # Mesh we use normalizes wing height to one

g_I = tan(deg2rad(30)) * height

#base = 0.8059
base = 0.8059 / 1.1963 # Mesh we use normalizes wing height to one

g_II = base - g_I
g_III = tan(deg2rad(15.8)) * height
A = height * (0.5 * (g_I + g_III) + g_II)

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure3D(aoa(), rho_inf(),
                                                                     u_inf(equations), A))
drag_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           DragCoefficientPressure3D(aoa(), rho_inf(),
                                                                     u_inf(equations), A))

analysis_interval = 100_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[], # Turn off error computation
                                     analysis_integrals = (lift_coefficient,
                                                           drag_coefficient))

alive_callback = AliveCallback(alive_interval = 50)

save_sol_interval = 50_000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out/")

save_restart = SaveRestartCallback(interval = save_sol_interval)

callbacks = CallbackSet(summary_callback,
                        alive_callback, analysis_callback,
                        save_solution, save_restart)

###############################################################################

sol = solve(ode, SSPRK43(); abstol = 1.0e-6, reltol = 1.0e-6,
            dt = 1e-8,
            ode_default_options()..., callback = callbacks);
