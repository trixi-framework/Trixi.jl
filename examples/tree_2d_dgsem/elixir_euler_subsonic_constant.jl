using OrdinaryDiffEqSSPRK
using Trixi

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
@inline function boundary_condition_outflow_general(u_inner, orientation::Integer,
                                                    direction, x, t,
                                                    surface_flux_function,
                                                    equations::CompressibleEulerEquations2D)
    rho_local, vx_local, vy_local, p_local = cons2prim(u_inner, equations)
    a_local = sqrt(equations.gamma * p_local / rho_local)
    v_mag = sqrt(vx_local^2 + vy_local^2)
    Mach_local = abs(v_mag / a_local)
    if Mach_local <= 1 # The `if` is not needed in this elixir but kept for generality
        # In general, `p_local` need not be available from the initial condition
        p_local = pressure(initial_condition_subsonic(x, t, equations), equations)
    end

    prim = SVector(rho_local, vx_local, vy_local, p_local)
    u_surface = prim2cons(prim, equations)

    return flux(u_surface, orientation, equations)
end

boundary_conditions = (x_neg = boundary_condition_outflow_general,
                       x_pos = boundary_condition_outflow_general,
                       y_neg = boundary_condition_outflow_general,
                       y_pos = boundary_condition_outflow_general)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                periodicity = false, n_cells_max = 512^2 * 16)

surface_flux = flux_lax_friedrichs

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_integral = VolumeIntegralWeakForm()

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

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
