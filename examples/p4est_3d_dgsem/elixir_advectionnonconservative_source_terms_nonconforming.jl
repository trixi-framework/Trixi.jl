using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the nonconservative linear advection equation
#   ∂ₜ u + a·∇ u = 0
# with a spatially varying velocity `a`
# on a statically refined, non-periodic mesh using a manufactured solution

equations = NonconservativeLinearScalarAdvectionEquation3D()

# Advection velocity with non-vanishing divergence, prescribed at every node of the mesh.
# The auxiliary variables of `NonconservativeLinearScalarAdvectionEquation3D` are the components
# `(a1, a2, a3)` of the advection velocity.
function velocity_variable(x)
    a1 = 1 + 0.5f0 * sinpi(x[2])
    a2 = 0.5f0 + 0.25f0 * sinpi(x[1])
    a3 = 0.25f0 * cospi(x[3])
    return SVector(a1, a2, a3)
end

function aux_field(x, equations::NonconservativeLinearScalarAdvectionEquation3D)
    velocity_variable(x)
end

function initial_condition_manufactured(x, t,
                                        equations::NonconservativeLinearScalarAdvectionEquation3D)
    return SVector(1 + 0.5f0 * sinpi(x[1] + x[2] + x[3] - t))
end

# Source terms for the manufactured solution
#   ∂ₜu + a⋅∇u = s.
# In contrast to the conservative form, the divergence of `a` does not appear.
# The advection velocity `a` is taken from the auxiliary variables.
function source_terms_manufactured(u, aux, x, t,
                                   equations::NonconservativeLinearScalarAdvectionEquation3D)
    RealT = eltype(x)
    a1, a2, a3 = aux
    # Derivative of the manufactured solution with respect to each spatial coordinate
    du_dx = 0.5f0 * convert(RealT, pi) * cospi(x[1] + x[2] + x[3] - t)
    du_dt = -du_dx

    return SVector(du_dt + (a1 + a2 + a3) * du_dx)
end

initial_condition = initial_condition_manufactured
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

# The physical flux vanishes, so the advection term is discretized only with the
# nonconservative flux. The local Lax-Friedrichs flux adds the upwind dissipation.
surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)
volume_flux = (flux_central, flux_nonconservative_central)
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)
trees_per_dimension = (1, 1, 1)

mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min,
                 coordinates_max = coordinates_max,
                 initial_refinement_level = 2,
                 periodicity = false)

# Refine the bottom left quadrant of each tree to level 3 to create mortars. The auxiliary
# variables are continuous, so the values of the small elements are used on both sides of
# the mortars.
function refine_fn(p8est, which_tree, quadrant)
    quadrant_obj = unsafe_load(quadrant)
    if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.z == 0 &&
       quadrant_obj.level < 3
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
                         (Ptr{Trixi.p8est_t}, Ptr{Trixi.p4est_topidx_t},
                          Ptr{Trixi.p8est_quadrant_t}))
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
