using OrdinaryDiffEqSSPRK
using Trixi
using LinearAlgebra: norm

###############################################################################
## Semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_subsonic(x_, t, equations::CompressibleEulerEquations2D)
    rho, v1, v2, p = (0.5313, 0.0, 0.0, 0.4)

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_subsonic

# Calculate the boundary flux from the inner state while using the pressure from the outer state
# when the flow is subsonic (which is always the case in this example).

# If the naive approach of only using the inner state is used, the errors increase with the
# increase of refinement level, see https://github.com/trixi-framework/Trixi.jl/issues/2530
# These errors arise from the corner points in this test.

# See the reference below for a discussion on inflow/outflow boundary conditions. The subsonic
# outflow boundary conditions are discussed in Section 2.3.
#
# - Jan-Reneé Carlson (2011)
#   Inflow/Outflow Boundary Conditions with Application to FUN3D.
#   [NASA TM 20110022658](https://ntrs.nasa.gov/citations/20110022658)
@inline function boundary_condition_outflow_general(u_inner,
                                                    normal_direction::AbstractVector, x, t,
                                                    surface_flux_function,
                                                    equations::CompressibleEulerEquations2D)

    # This would be for the general case where we need to check the magnitude of the local Mach number
    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # Rotate the internal solution state
    u_local = Trixi.rotate_to_x(u_inner, normal, equations)

    # Compute the primitive variables
    rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

    # Compute local Mach number
    a_local = sqrt(equations.gamma * p_local / rho_local)
    Mach_local = abs(v_normal / a_local)
    if Mach_local <= 1.0 # The `if` is not needed in this elixir but kept for generality
        # In general, `p_local` need not be available from the initial condition
        p_local = pressure(initial_condition_subsonic(x, t, equations), equations)
    end

    # Create the `u_surface` solution state where the local pressure is possibly set from an external value
    prim = SVector(rho_local, v_normal, v_tangent, p_local)
    u_boundary = prim2cons(prim, equations)
    u_surface = Trixi.rotate_from_x(u_boundary, normal, equations)

    # Compute the flux using the appropriate mixture of internal / external solution states
    return flux(u_surface, normal_direction, equations)
end

boundary_conditions = Dict(:x_neg => boundary_condition_outflow_general,
                           :x_pos => boundary_condition_outflow_general,
                           :y_neg => boundary_condition_outflow_general,
                           :y_pos => boundary_condition_outflow_general)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

trees_per_dimension = (1, 1)


mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 3,
                 periodicity = false)

surface_flux = flux_lax_friedrichs

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
## ODE solvers, callbacks etc.

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = 100)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
## Run the simulation
sol = solve(ode, SSPRK54();
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
