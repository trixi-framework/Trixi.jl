# Channel flow around a cylinder at Mach 3
#
# Boundary conditions are supersonic Mach 3 inflow at the left portion of the domain
# and supersonic outflow at the right portion of the domain. The top and bottom of the
# channel as well as the cylinder are treated as Euler slip wall boundaries.
# This flow results in strong shock reflections / interactions as well as Kelvin-Helmholtz
# instabilities at later times as two Mach stems form above and below the cylinder.
#
# For complete details on the problem setup see Section 5.7 of the paper:
# - Jean-Luc Guermond, Murtazo Nazarov, Bojan Popov, and Ignacio Tomas (2018)
#   Second-Order Invariant Domain Preserving Approximation of the Euler Equations using Convex Limiting.
#   [DOI: 10.1137/17M1149961](https://doi.org/10.1137/17M1149961)
#
# Keywords: supersonic flow, shock capturing, AMR, unstructured curved mesh, positivity preservation, compressible Euler, 2D

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_mach3_flow(x, t, equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 3.0
    v2 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach3_flow

# Supersonic inflow boundary condition.
# Calculate the boundary flux entirely from the external solution state, i.e., set
# external solution state values for everything entering the domain.
@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t, surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_mach3_flow(x, t, equations)
    flux = Trixi.flux(u_boundary, normal_direction, equations)

    return flux
end

# For subcell limiting, the calculation of local bounds for non-periodic domains requires the
# boundary outer state. Those functions return the boundary value for a specific boundary condition
# at time `t`, for the node with spatial indices `indices` and the given `normal_direction`.
# only for P4estMesh{2}
@inline function Trixi.get_boundary_outer_state(u_inner, t,
                                                boundary_condition::typeof(boundary_condition_supersonic_inflow),
                                                normal_direction::AbstractVector,
                                                equations, dg, cache,
                                                indices...)
    x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, indices...)

    return initial_condition_mach3_flow(x, t, equations)
end

# Supersonic outflow boundary condition.
# Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# except all the solution state values are set from the internal solution as everything leaves the domain
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function,
                                            equations::CompressibleEulerEquations2D)
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end

# only for P4estMesh{2}
@inline function Trixi.get_boundary_outer_state(u_inner, t,
                                                boundary_condition::typeof(boundary_condition_outflow),
                                                normal_direction::AbstractVector,
                                                equations, dg, cache,
                                                indices...)
    return u_inner
end

# only for P4estMesh{2}
@inline function Trixi.get_boundary_outer_state(u_inner, t,
                                                boundary_condition::typeof(boundary_condition_slip_wall),
                                                normal_direction::AbstractVector,
                                                equations::CompressibleEulerEquations2D,
                                                dg, cache, indices...)
    factor = (normal_direction[1] * u_inner[2] + normal_direction[2] * u_inner[3])
    u_normal = (factor / sum(normal_direction .^ 2)) * normal_direction

    return SVector(u_inner[1],
                   u_inner[2] - 2 * u_normal[1],
                   u_inner[3] - 2 * u_normal[2],
                   u_inner[4])
end

boundary_conditions = Dict(:Bottom => boundary_condition_slip_wall,
                           :Circle => boundary_condition_slip_wall,
                           :Top => boundary_condition_slip_wall,
                           :Right => boundary_condition_outflow,
                           :Left => boundary_condition_supersonic_inflow)

volume_flux = flux_ranocha_turbo
surface_flux = flux_lax_friedrichs
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                local_twosided_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                local_onesided_variables_nonlinear = [(Trixi.entropy_guermond_etal,
                                                                       min)],
                                max_iterations_newton = 50) # Default value of 10 iterations is too low to fulfill bounds.

volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/a08f78f6b185b63c3baeff911a63f628/raw/addac716ea0541f588b9d2bd3f92f643eb27b88f/abaqus_cylinder_in_channel.inp",
                           joinpath(@__DIR__, "abaqus_cylinder_in_channel.inp"))

mesh = P4estMesh{2}(mesh_file, initial_refinement_level = 0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
summary_callback() # print the timer summary
