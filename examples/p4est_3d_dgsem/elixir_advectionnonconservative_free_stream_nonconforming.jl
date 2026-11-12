using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the nonconservative linear advection equation
#   ∂ₜ u + a·∇ u = 0
# with a discontinuous velocity `a` on a statically refined, non-periodic mesh.
# Since a·∇ u = 0 for a constant state `u`, the constant state is preserved up to
# round-off errors also for discontinuous velocities and at mortars.

equations = NonconservativeLinearScalarAdvectionEquation3D()

# Smooth velocities on both sides of the discontinuity at x = -0.5
function velocity_left(x)
    a1 = 1 + 0.5f0 * sinpi(x[2])
    a2 = 0.5f0 + 0.25f0 * sinpi(x[1])
    a3 = 0.25f0 * cospi(x[3])
    return SVector(a1, a2, a3)
end

function velocity_right(x)
    a1 = 0.5f0 - 0.25f0 * cospi(x[3])
    a2 = -0.5f0 + 0.25f0 * sinpi(x[1])
    a3 = 0.75f0 + 0.25f0 * sinpi(x[2])
    return SVector(a1, a2, a3)
end

# The velocity is overwritten by a discontinuous one below
function aux_field(x, equations::NonconservativeLinearScalarAdvectionEquation3D)
    return velocity_left(x)
end

function initial_condition_constant(x, t,
                                    equations::NonconservativeLinearScalarAdvectionEquation3D)
    return SVector(one(eltype(x)))
end

initial_condition = initial_condition_constant
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

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

# Refine the bottom left quadrant of each tree to level 3 to create mortars, in particular
# at the discontinuity x = -0.5
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

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    aux_field = aux_field)

# Workaround to set a discontinuous velocity for testing, similar to the discontinuous
# bottom topographies in TrixiShallowWater.jl. Since `aux_field` only gets the node
# coordinates, nodes on an interface get the same values from both sides. Thus, we choose
# the velocity based on the center of each element and overwrite the auxiliary variables.
# Afterwards, the auxiliary variables at the interfaces and mortars have to be updated.
# Note that `remake` evaluates `aux_field` again and thus removes the discontinuity.
(; aux) = semi.cache.aux_vars
(; node_coordinates) = semi.cache.elements
for element in eachelement(solver, semi.cache)
    x_center = sum(node_coordinates[1, :, :, :, element]) / nnodes(solver)^3
    for k in eachnode(solver), j in eachnode(solver), i in eachnode(solver)
        x_node = Trixi.get_node_coords(node_coordinates, equations, solver,
                                       i, j, k, element)
        aux_node = x_center < -0.5 ? velocity_left(x_node) : velocity_right(x_node)
        Trixi.set_aux_node_vars!(aux, aux_node, equations, solver, i, j, k, element)
    end
end
Trixi.init_aux_surface_node_vars!(semi.cache.aux_vars, mesh, equations, solver,
                                  semi.cache)

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
