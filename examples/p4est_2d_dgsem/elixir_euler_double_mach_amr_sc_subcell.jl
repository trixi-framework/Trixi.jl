using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

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

# Using with StructuredMesh{2}
@inline function characteristic_boundary_value_function(outer_boundary_value_function,
                                                        u_inner,
                                                        normal_direction::AbstractVector,
                                                        direction, x, t,
                                                        equations::CompressibleEulerEquations2D)
    # Get inverse of density
    srho = 1 / u_inner[1]

    # Get normal velocity
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        factor = 1
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        factor = -1
    end
    vn = factor * srho *
         (normal_direction[1] * u_inner[2] + normal_direction[2] * u_inner[3]) /
         sqrt(normal_direction[1]^2 + normal_direction[2]^2)

    return calc_characteristic_boundary_value_function(outer_boundary_value_function,
                                                       u_inner, srho, vn, x, t,
                                                       equations)
end

# Using with P4estMesh{2}
@inline function characteristic_boundary_value_function(outer_boundary_value_function,
                                                        u_inner,
                                                        normal_direction::AbstractVector,
                                                        x, t,
                                                        equations::CompressibleEulerEquations2D)
    # Get inverse of density
    srho = 1 / u_inner[1]

    # Get normal velocity
    vn = srho * (normal_direction[1] * u_inner[2] + normal_direction[2] * u_inner[3]) /
         sqrt(normal_direction[1]^2 + normal_direction[2]^2)

    return calc_characteristic_boundary_value_function(outer_boundary_value_function,
                                                       u_inner, srho, vn, x, t,
                                                       equations)
end

# Function to compute the outer state of the characteristics-based boundary condition.
# This function is called by all mesh types.
@inline function calc_characteristic_boundary_value_function(outer_boundary_value_function,
                                                             u_inner, srho, vn, x, t,
                                                             equations::CompressibleEulerEquations2D)
    # get pressure and Mach from state
    p = pressure(u_inner, equations)
    a = sqrt(equations.gamma * p * srho)
    normalMachNo = abs(vn / a)

    if vn < 0 # inflow
        if normalMachNo < 1.0
            # subsonic inflow: All variables from outside but pressure
            cons = outer_boundary_value_function(x, t, equations)

            prim = cons2prim(cons, equations)
            prim = SVector(view(prim, 1:3)..., p)
            cons = prim2cons(prim, equations)
        else
            # supersonic inflow: All variables from outside
            cons = outer_boundary_value_function(x, t, equations)
        end
    else # outflow
        if normalMachNo < 1.0
            # subsonic outflow: All variables from inside but pressure
            cons = outer_boundary_value_function(x, t, equations)

            prim = cons2prim(u_inner, equations)
            prim = SVector(view(prim, 1:3)..., pressure(cons, equations))
            cons = prim2cons(prim, equations)
        else
            # supersonic outflow: All variables from inside
            cons = u_inner
        end
    end

    return cons
end

"""
    BoundaryConditionCharacteristic(outer_boundary_value_function)
Characteristic-based boundary condition.
!!! warning "Experimental code"
  This numerical flux is experimental and may change in any future release.
"""
struct BoundaryConditionCharacteristic{B, C}
    outer_boundary_value_function::B
    boundary_value_function::C
end

function BoundaryConditionCharacteristic(outer_boundary_value_function)
    BoundaryConditionCharacteristic{typeof(outer_boundary_value_function),
                                    typeof(characteristic_boundary_value_function)}(outer_boundary_value_function,
                                                                                    characteristic_boundary_value_function)
end

# Characteristic-based boundary condition for use with TreeMesh or StructuredMesh
@inline function (boundary_condition::BoundaryConditionCharacteristic)(u_inner,
                                                                       orientation_or_normal,
                                                                       direction,
                                                                       x, t,
                                                                       surface_flux_function,
                                                                       equations)
    u_boundary = boundary_condition.boundary_value_function(boundary_condition.outer_boundary_value_function,
                                                            u_inner,
                                                            orientation_or_normal,
                                                            direction, x, t, equations)

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation_or_normal,
                                     equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation_or_normal,
                                     equations)
    end

    return flux
end

# Characteristic-based boundary condition for use with P4estMesh
@inline function (boundary_condition::BoundaryConditionCharacteristic)(u_inner,
                                                                       normal_direction,
                                                                       x, t,
                                                                       surface_flux_function,
                                                                       equations)
    u_boundary = boundary_condition.boundary_value_function(boundary_condition.outer_boundary_value_function,
                                                            u_inner, normal_direction,
                                                            x, t, equations)

    # Calculate boundary flux
    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

    return flux
end
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
        u_boundary = characteristic_boundary_value_function(initial_condition_double_mach_reflection,
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
                                                boundary_condition::BoundaryConditionCharacteristic,
                                                normal_direction::AbstractVector,
                                                mesh::P4estMesh{2},
                                                equations::CompressibleEulerEquations2D,
                                                dg, cache, indices...)
    x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, indices...)
    u_outer = characteristic_boundary_value_function(initial_condition_double_mach_reflection,
                                                     u_inner,
                                                     normal_direction,
                                                     x, t, equations)

    return u_outer
end

@inline function Trixi.get_boundary_outer_state(u_inner, t,
                                                boundary_condition::typeof(boundary_condition_mixed_characteristic_wall),
                                                normal_direction::AbstractVector,
                                                mesh::P4estMesh{2},
                                                equations::CompressibleEulerEquations2D,
                                                dg, cache, indices...)
    x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, indices...)
    if x[1] < 1 / 6 # BoundaryConditionCharacteristic
        u_outer = characteristic_boundary_value_function(initial_condition_double_mach_reflection,
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

boundary_conditions = (; Bottom = boundary_condition_mixed_characteristic_wall,
                       Top = boundary_condition_inflow_outflow,
                       Right = boundary_condition_inflow_outflow,
                       Left = boundary_condition_inflow_outflow)

surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
volume_flux = flux_ranocha

polydeg = 4
basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                local_twosided_variables_cons = ["rho"],
                                local_onesided_variables_nonlinear = [(entropy_guermond_etal,
                                                                       min)],
                                max_iterations_newton = 100,
                                bar_states = false)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)

mortar = MortarIDP(equations, basis, limiter_idp)
solver = DGSEM(basis, surface_flux, volume_integral, mortar)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/a0806ef0d03cf5ea221af523167b6e32/raw/61ed0eb017eb432d996ed119a52fb041fe363e8c/abaqus_double_mach.inp",
                           joinpath(@__DIR__, "abaqus_double_mach.inp"))

mesh = P4estMesh{2}(mesh_file)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

save_restart = SaveRestartCallback(interval = 10000,
                                   save_final_restart = true)

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 3, med_threshold = 0.05,
                                      max_level = 5, max_threshold = 0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true,
                           limiter! = stage_limiter!)

stepsize_callback = StepsizeCallback(cfl = 0.4, bar_states = false)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback,
                        stepsize_callback)

###############################################################################

stage_callbacks = (SubcellLimiterIDPCorrection(),
                   BoundsCheckCallback(save_errors = false))

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
