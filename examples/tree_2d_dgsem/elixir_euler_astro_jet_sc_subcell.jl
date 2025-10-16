using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5 / 3
equations = CompressibleEulerEquations2D(gamma)

# Initial condition adopted from
# - Yong Liu, Jianfang Lu, and Chi-Wang Shu
#   An oscillation free discontinuous Galerkin method for hyperbolic systems
#   https://tinyurl.com/c76fjtx4
# Mach = 2000 jet
function initial_condition_astro_jet(x, t, equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    @unpack gamma = equations
    rho = 0.5f0
    v1 = 0
    v2 = 0
    p = convert(RealT, 0.4127)
    # add inflow for t>0 at x=-0.5
    # domain size is [-0.5,+0.5]^2
    if (t > 0) && (x[1] ≈ -0.5f0) && (abs(x[2]) < RealT(0.05))
        rho = 5.0f0
        v1 = 800 # about Mach number Ma = 2000
        v2 = 0
        p = convert(RealT, 0.4127)
    end
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_astro_jet

# TODO: Add docstring when about to merge.
# Using with TreeMesh{2}
@inline function characteristic_boundary_value_function(outer_boundary_value_function,
                                                        u_inner, orientation::Integer,
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
    if orientation == 1
        vn = factor * u_inner[2] * srho
    else
        vn = factor * u_inner[3] * srho
    end

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

@inline function Trixi.get_boundary_outer_state(u_inner, t,
                                                boundary_condition::BoundaryConditionCharacteristic,
                                                orientation_or_normal, direction,
                                                mesh::TreeMesh{2}, equations, dg, cache,
                                                indices...)
    (; node_coordinates) = cache.elements

    x = Trixi.get_node_coords(node_coordinates, equations, dg, indices...)
    u_outer = boundary_condition.boundary_value_function(boundary_condition.outer_boundary_value_function,
                                                         u_inner, orientation_or_normal,
                                                         direction, x, t, equations)

    return u_outer
end

boundary_conditions = (x_neg = BoundaryConditionCharacteristic(initial_condition_astro_jet),
                       x_pos = BoundaryConditionCharacteristic(initial_condition_astro_jet),
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

surface_flux = flux_lax_friedrichs # HLLC needs more shock capturing (alpha_max)
volume_flux = flux_chandrashekar # works with Chandrashekar flux as well
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

# shock capturing necessary for this tough example
limiter_idp = SubcellLimiterIDP(equations, basis;
                                # local_twosided_variables_cons = ["rho"],
                                # local_onesided_variables_nonlinear = [(Trixi.entropy_guermond_etal,
                                #                                        min)],
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                positivity_correction_factor = 0.1,
                                max_iterations_newton = 200)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
mortar = MortarIDP(equations, basis; pure_low_order = false,
                   basis_function = :piecewise_constant,
                   # basis_function = :piecewise_linear,
                   positivity_variables_cons = ["rho"],
                   positivity_variables_nonlinear = [pressure])
solver = DGSEM(basis, surface_flux, volume_integral, mortar)

coordinates_min = (-0.5, -0.5)
coordinates_max = (0.5, 0.5)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 8,
                periodicity = (false, true),
                n_cells_max = 100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.001)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

restart_interval = 21800
save_restart = SaveRestartCallback(interval = restart_interval,
                                   save_final_restart = true)

positivity_limiter = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-10, 1.0e-10),
                                                         variables = (Trixi.density,
                                                                      pressure))

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 1.0,
                                          alpha_min = 0.0001,
                                          alpha_smooth = true,
                                          variable = Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 2,
                                      med_level = 0, med_threshold = 0.0003, # med_level = current level
                                      max_level = 8, max_threshold = 0.003)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true,
                           limiter! = positivity_limiter)

function cfl_amr(t)
    if t < 4.5e-7
        return 0.001
    else
        return 0.5
    end
end
stepsize_callback = StepsizeCallback(cfl = cfl_amr)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        amr_callback,
                        stepsize_callback)

stage_callbacks = (SubcellLimiterIDPCorrection(),
                   BoundsCheckCallback(save_errors = false, interval = 1))

###############################################################################
# run the simulation
sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  maxiters = 1_000_000, callback = callbacks);
