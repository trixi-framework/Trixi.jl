using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_double_mach_reflection(x, t, equations::CompressibleEulerEquations2D)

Compressible Euler setup for a double Mach reflection problem.
Involves strong shock interactions as well as steady / unsteady flow structures.
Also exercises special boundary conditions along the bottom of the domain that is a mixture of
Dirichlet and slip wall.
See Section IV c on the paper below for details.

- Paul Woodward and Phillip Colella (1984)
  The Numerical Simulation of Two-Dimensional Fluid Flows with Strong Shocks.
  [DOI: 10.1016/0021-9991(84)90142-6](https://doi.org/10.1016/0021-9991(84)90142-6)
"""
@inline function initial_condition_double_mach_reflection(x, t,
                                                          equations::CompressibleEulerEquations2D)
    if x[1] < 1 / 6 + (x[2] + 20 * t) / sqrt(3)
        phi = pi / 6
        sin_phi, cos_phi = sincos(phi)

        rho = 8.0
        v1 = 8.25 * cos_phi
        v2 = -8.25 * sin_phi
        p = 116.5
    else
        rho = 1.4
        v1 = 0.0
        v2 = 0.0
        p = 1.0
    end

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_double_mach_reflection

boundary_condition_inflow_outflow = BoundaryConditionCharacteristic(initial_condition)

# Special mixed boundary condition type for the :y_neg of the domain.
# It is charachteristic-based when x < 1/6 and a slip wall when x >= 1/6
# Note: Only for P4estMesh
@inline function boundary_condition_mixed_characteristic_wall(u_inner,
                                                              normal_direction::AbstractVector,
                                                              x, t, surface_flux_function,
                                                              equations::CompressibleEulerEquations2D)
    if x[1] < 1 / 6
        # From the BoundaryConditionCharacteristic
        # get the external state of the solution
        u_boundary = Trixi.characteristic_boundary_value_function(initial_condition_double_mach_reflection,
                                                                  u_inner,
                                                                  normal_direction,
                                                                  x, t,
                                                                  equations)
        # Calculate boundary flux
        flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
    else # x[1] >= 1 / 6
        # Use the free slip wall BC otherwise
        flux = boundary_condition_slip_wall(u_inner, normal_direction, x, t,
                                            surface_flux_function, equations)
    end

    return flux
end

@inline function Trixi.get_boundary_outer_state(u_inner, t,
                                                boundary_condition::typeof(boundary_condition_mixed_characteristic_wall),
                                                normal_direction::AbstractVector,
                                                mesh::P4estMesh{2},
                                                equations::CompressibleEulerEquations2D,
                                                dg, cache, indices...)
    x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, indices...)
    if x[1] < 1 / 6 # BoundaryConditionCharacteristic
        u_outer = Trixi.characteristic_boundary_value_function(initial_condition_double_mach_reflection,
                                                               u_inner,
                                                               normal_direction,
                                                               x, t, equations)

    else # if x[1] >= 1 / 6 # boundary_condition_slip_wall
        factor = (normal_direction[1] * u_inner[2] + normal_direction[2] * u_inner[3])
        u_normal = (factor / sum(normal_direction .^ 2)) * normal_direction

        u_outer = SVector(u_inner[1],
                          u_inner[2] - 2.0 * u_normal[1],
                          u_inner[3] - 2.0 * u_normal[2],
                          u_inner[4])
    end
    return u_outer
end

boundary_conditions = Dict(:y_neg => boundary_condition_mixed_characteristic_wall,
                           :y_pos => boundary_condition_inflow_outflow,
                           :x_pos => boundary_condition_inflow_outflow,
                           :x_neg => boundary_condition_inflow_outflow)

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the
# `StepsizeCallback` (CFL-Condition) and less diffusion.
surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
volume_flux = flux_ranocha
polydeg = 4
basis = LobattoLegendreBasis(polydeg)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                local_twosided_variables_cons = ["rho"],
                                local_onesided_variables_nonlinear = [(Trixi.entropy_guermond_etal,
                                                                       min)],
                                positivity_correction_factor = 0.1,
                                max_iterations_newton = 100,
                                bar_states = true)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

initial_refinement_level = 4
trees_per_dimension = (4 * 2^initial_refinement_level, 2^initial_refinement_level)
coordinates_min = (0.0, 0.0)
coordinates_max = (4.0, 1.0)
mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 0,
                 periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback,
                        save_solution)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
