using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the nonconservative linear advection equation
#   ∂ₜ u + a·∇ u = 0
# with a spatially varying velocity `a`
# on a statically refined, non-periodic mesh using a manufactured solution
#
# In this elixir, we employ the specialization that combines conservative and
# nonconservative fluxes in a single function. The single function must return the
# two contributions
#
#   flux_cons(u_ll, u_rr, aux_ll, aux_rr, normal_direction, equations)
#     + 0.5f0 * flux_noncons(u_ll, u_rr, aux_ll, aux_rr, normal_direction, equations)
#
# and
#
#   flux_cons(u_ll, u_rr, aux_ll, aux_rr, normal_direction, equations)
#     + 0.5f0 * flux_noncons(u_rr, u_ll, aux_rr, aux_ll, normal_direction, equations),
#
# i.e., the flux for the element on the left side and the one for the element on the
# right side of the interface. The GPU kernels support nonconservative terms only with
# such combined fluxes. This elixir is equivalent to
# `elixir_advectionnonconservative_source_terms_nonconforming.jl`, which uses tuples of
# conservative and nonconservative fluxes.

# The following function is equivalent to `(flux_lax_friedrichs, flux_nonconservative_central)`.
@inline function flux_lax_friedrichs_nonconservative_central(u_ll, u_rr, aux_ll, aux_rr,
                                                             normal_direction::AbstractVector,
                                                             equations::NonconservativeLinearScalarAdvectionEquation2D)
    f = flux_lax_friedrichs(u_ll, u_rr, aux_ll, aux_rr, normal_direction, equations)
    g_left = flux_nonconservative_central(u_ll, u_rr, aux_ll, aux_rr,
                                          normal_direction, equations)
    g_right = flux_nonconservative_central(u_rr, u_ll, aux_rr, aux_ll,
                                           normal_direction, equations)
    flux_left = f + 0.5f0 * g_left
    flux_right = f + 0.5f0 * g_right
    return flux_left, flux_right
end

@inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_lax_friedrichs_nonconservative_central),
::NonconservativeLinearScalarAdvectionEquation2D) = Trixi.True()

# The following function is equivalent to `(flux_central, flux_nonconservative_central)`.
@inline function flux_central_nonconservative_central(u_ll, u_rr, aux_ll, aux_rr,
                                                      normal_direction::AbstractVector,
                                                      equations::NonconservativeLinearScalarAdvectionEquation2D)
    f = flux_central(u_ll, u_rr, aux_ll, aux_rr, normal_direction, equations)
    g_left = flux_nonconservative_central(u_ll, u_rr, aux_ll, aux_rr,
                                          normal_direction, equations)
    g_right = flux_nonconservative_central(u_rr, u_ll, aux_rr, aux_ll,
                                           normal_direction, equations)
    flux_left = f + 0.5f0 * g_left
    flux_right = f + 0.5f0 * g_right
    return flux_left, flux_right
end

@inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_central_nonconservative_central),
::NonconservativeLinearScalarAdvectionEquation2D) = Trixi.True()

# For nonperiodic boundary conditions, the boundary flux must also be specialized
# such that it returns only the first contribution of the surface flux function.
@inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner, aux_inner,
                                                                  normal_direction::AbstractVector,
                                                                  x, t,
                                                                  surface_flux_function::typeof(flux_lax_friedrichs_nonconservative_central),
                                                                  equations)
    # Get the external value of the solution.
    u_boundary = boundary_condition.boundary_value_function(x, t, equations)

    # Calculate boundary flux. There are no separate auxiliary variables outside of the
    # boundary, so the inner ones are used on both sides.
    flux, _ = surface_flux_function(u_inner, u_boundary, aux_inner, aux_inner,
                                    normal_direction, equations)
    return flux
end

equations = NonconservativeLinearScalarAdvectionEquation2D()

# Advection velocity with non-vanishing divergence, prescribed at every node of the mesh.
# The auxiliary variables of `NonconservativeLinearScalarAdvectionEquation2D` are the components
# `(a1, a2)` of the advection velocity.
function velocity_variable(x)
    a1 = 1 + 0.5f0 * sinpi(x[2])
    a2 = 0.5f0 + 0.25f0 * cospi(x[2])
    return SVector(a1, a2)
end

function aux_field(x, equations::NonconservativeLinearScalarAdvectionEquation2D)
    return velocity_variable(x)
end

function initial_condition_manufactured(x, t,
                                        equations::NonconservativeLinearScalarAdvectionEquation2D)
    return SVector(1 + 0.5f0 * sinpi(x[1] + x[2] - t))
end

# Source terms for the manufactured solution
#   ∂ₜu + a⋅∇u = s.
# In contrast to the conservative form, the divergence of `a` does not appear.
# The advection velocity `a` is taken from the auxiliary variables.
@inline function source_terms_manufactured(u, aux, x, t,
                                           equations::NonconservativeLinearScalarAdvectionEquation2D)
    RealT = eltype(x)
    a1, a2 = aux
    # Derivative of the manufactured solution with respect to each spatial coordinate
    du_dx = 0.5f0 * convert(RealT, pi) * cospi(x[1] + x[2] - t)
    du_dt = -du_dx

    return SVector(du_dt + (a1 + a2) * du_dx)
end

initial_condition = initial_condition_manufactured
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

surface_flux = flux_lax_friedrichs_nonconservative_central
volume_flux = flux_central_nonconservative_central
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
trees_per_dimension = (1, 1)

mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min,
                 coordinates_max = coordinates_max,
                 initial_refinement_level = 2,
                 periodicity = false)

# Refine the bottom left quadrant of each tree to level 3 to create mortars
function refine_fn(p4est, which_tree, quadrant)
    quadrant_obj = unsafe_load(quadrant)
    if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.level < 3
        # return true (refine)
        return Cint(1)
    else
        # return false (don't refine)
        return Cint(0)
    end
end

# Refine recursively until each bottom left quadrant of a tree has level 3
# The mesh will be rebalanced before the simulation starts
refine_fn_c = @cfunction(refine_fn, Cint,
                         (Ptr{Trixi.p4est_t}, Ptr{Trixi.p4est_topidx_t},
                          Ptr{Trixi.p4est_quadrant_t}))
Trixi.refine_p4est!(mesh.p4est, true, refine_fn_c, C_NULL)

# The `aux_field` is required since `equations` has auxiliary variables. They are evaluated
# once at all nodes.
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms = source_terms_manufactured,
                                    boundary_conditions = boundary_conditions,
                                    aux_field = aux_field)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
# Setting `real_type` allows to change the real number type, e.g., to `Float32`.
# This is particularly useful when changing the `storage_type` to a GPU array
# type such as `ROCArray` (AMD) or `CuArray` (NVIDIA CUDA).
semi = trixi_adapt(semi; real_type = nothing, storage_type = nothing)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
