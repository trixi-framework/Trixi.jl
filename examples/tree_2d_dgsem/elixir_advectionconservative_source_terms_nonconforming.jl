using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the conservative linear advection equation
#   ∂ₜ u + ∇·(a u) = 0
# with a spatially varying velocity `a`
# on a statically refined, non-periodic mesh using a manufactured solution

equations = ConservativeLinearScalarAdvectionEquation2D()

# Advection velocity with non-vanishing divergence, prescribed at every node of the mesh.
# The auxiliary variables of `ConservativeLinearScalarAdvectionEquation2D` are the components
# `(a1, a2)` of the advection velocity.
function velocity_variable(x)
    a1 = 1 + 0.5f0 * sinpi(x[2])
    a2 = 0.5f0 + 0.25f0 * cospi(x[2])
    return SVector(a1, a2)
end

# Divergence of `velocity_variable`, which is not stored as auxiliary variable
function divergence_velocity_variable(x)
    return -0.25f0 * convert(eltype(x), pi) * sinpi(x[2])
end

aux_field(x, equations::ConservativeLinearScalarAdvectionEquation2D) = velocity_variable(x)

function initial_condition_manufactured(x, t,
                                        equations::ConservativeLinearScalarAdvectionEquation2D)
    return SVector(1 + 0.5f0 * sinpi(x[1] + x[2] - t))
end

# Source terms for the manufactured solution
#   ∂ₜu + ∇⋅(a u) = ∂ₜu + a⋅∇u + (∇⋅a) u = s.
# The advection velocity `a` is taken from the auxiliary variables.
@inline function source_terms_manufactured(u, aux, x, t,
                                           equations::ConservativeLinearScalarAdvectionEquation2D)
    RealT = eltype(x)
    a1, a2 = aux
    u_exact = initial_condition_manufactured(x, t, equations)[1]
    # Derivative of `u_exact` with respect to each spatial coordinate
    du_dx = 0.5f0 * convert(RealT, pi) * cospi(x[1] + x[2] - t)
    du_dt = -du_dx

    return SVector(du_dt + (a1 + a2) * du_dx +
                   divergence_velocity_variable(x) * u_exact)
end

initial_condition = initial_condition_manufactured
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
# Refine the bottom left element to create mortars. This matches the static refinement
# of the corresponding `P4estMesh` elixir.
refinement_patches = ((type = "box", coordinates_min = (-1.0, -1.0),
                       coordinates_max = (-0.5, -0.5)),)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                refinement_patches = refinement_patches,
                periodicity = false)

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
