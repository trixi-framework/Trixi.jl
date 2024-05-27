
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the Lattice-Boltzmann equations for the D2Q9 scheme

equations = LatticeBoltzmannEquations2D(Ma = 0.05, Re = 2000)

"""
    initial_condition_couette_unsteady(x, t, equations::LatticeBoltzmannEquations2D)

Initial state for an *unsteady* Couette flow setup, which is also the exact solution for the
incompressible Navier-Stokes equations. To be used in combination with
[`boundary_condition_couette`](@ref) and [`boundary_condition_noslip_wall`](@ref). In the limit,
this setup will converge to the state set in [`initial_condition_couette_steady`](@ref).
"""
function initial_condition_couette_unsteady(x, t, equations::LatticeBoltzmannEquations2D)
    @unpack L, u0, rho0, nu = equations

    x1, x2 = x
    v1 = u0 * x2 / L
    for m in 1:100
        lambda_m = m * pi / L
        v1 += 2 * u0 * (-1)^m / (lambda_m * L) * exp(-nu * lambda_m^2 * t) *
              sin(lambda_m * x2)
    end

    rho = 1
    v2 = 0

    return equilibrium_distribution(rho, v1, v2, equations)
end
initial_condition = initial_condition_couette_unsteady

"""
    boundary_condition_couette(u_inner, orientation, direction, x, t,
                               surface_flux_function,
                               equations::LatticeBoltzmannEquations2D)

Moving *upper* wall boundary condition for a Couette flow setup. To be used in combination with
[`boundary_condition_noslip_wall`](@ref) for the lower wall and
[`boundary_condition_periodic`](@ref) for the lateral boundaries.
"""
function boundary_condition_couette(u_inner, orientation, direction, x, t,
                                    surface_flux_function,
                                    equations::LatticeBoltzmannEquations2D)
    return boundary_condition_moving_wall_ypos(u_inner, orientation, direction, x, t,
                                               surface_flux_function, equations)
end

function boundary_condition_moving_wall_ypos(u_inner, orientation, direction, x, t,
                                             surface_flux_function,
                                             equations::LatticeBoltzmannEquations2D)
    @assert direction==4 "moving wall assumed in +y direction"

    @unpack rho0, u0, weights, c_s = equations
    cs_squared = c_s^2

    pdf1 = u_inner[3] + 2 * weights[1] * rho0 * u0 / cs_squared
    pdf2 = u_inner[2] # outgoing
    pdf3 = u_inner[1] + 2 * weights[3] * rho0 * (-u0) / cs_squared
    pdf4 = u_inner[2]
    pdf5 = u_inner[5] # outgoing
    pdf6 = u_inner[6] # outgoing
    pdf7 = u_inner[5] + 2 * weights[7] * rho0 * (-u0) / cs_squared
    pdf8 = u_inner[6] + 2 * weights[8] * rho0 * u0 / cs_squared
    pdf9 = u_inner[9]

    u_boundary = SVector(pdf1, pdf2, pdf3, pdf4, pdf5, pdf6, pdf7, pdf8, pdf9)

    # Calculate boundary flux (u_inner is "left" of boundary, u_boundary is "right" of boundary)
    return surface_flux_function(u_inner, u_boundary, orientation, equations)
end
boundary_conditions = (x_neg = boundary_condition_periodic,
                       x_pos = boundary_condition_periodic,
                       y_neg = boundary_condition_noslip_wall,
                       y_pos = boundary_condition_couette)

solver = DGSEM(polydeg = 3, surface_flux = flux_godunov)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = (true, false),
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 40.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Custom solution variables: normalize velocities by reference speed `u0`
@inline function macroscopic_normalized(u, equations::LatticeBoltzmannEquations2D)
    macroscopic = cons2macroscopic(u, equations)
    rho, v1, v2, p = macroscopic

    # Use `typeof(macroscopic)` to avoid having to explicitly add `using StaticArrays`
    convert(typeof(macroscopic), (rho, v1 / equations.u0, v2 / equations.u0, p))
end
function Trixi.varnames(::typeof(macroscopic_normalized),
                        equations::LatticeBoltzmannEquations2D)
    ("rho", "v1_normalized", "v2_normalized", "p")
end

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = macroscopic_normalized)

stepsize_callback = StepsizeCallback(cfl = 1.0)

collision_callback = LBMCollisionCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        collision_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
