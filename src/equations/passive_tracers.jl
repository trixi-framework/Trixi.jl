# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    PassiveTracerEquations(flow_equations; n_tracers)

Adds passive tracers to the `flow_equations`. The tracers are advected by the flow velocity.
These work for arbitrary dimensions with arbitrary numbers of tracers `n_tracers`, for conservative and non-conservative equations. For one dimension, with 
one tracer ``\chi`` and flow with density and velocity ``\rho, v`` respectively, the equation of the
passive tracer is
```math
\frac{\partial \rho \chi}{\partial t} + \frac{\partial}{\partial x} \left( \rho v \chi \right) = 0
```
"""
struct PassiveTracerEquations{NDIMS, NVARS, NTracers,
                              FlowEquations <: AbstractEquations} <:
       AbstractEquations{NDIMS, NVARS}
    flow_equations::FlowEquations
end

function PassiveTracerEquations(flow_equations::AbstractEquations; n_tracers::Int)
    return PassiveTracerEquations{ndims(flow_equations),
                                  nvariables(flow_equations) + n_tracers, n_tracers,
                                  typeof(flow_equations)}(flow_equations)
end

# Get the number of passive tracers
@inline ntracers(::PassiveTracerEquations{NDIMS, NVARS, NTracers, FlowEquations}) where {NDIMS, NVARS,
NTracers, FlowEquations} = NTracers

have_nonconservative_terms(equations::PassiveTracerEquations) = have_nonconservative_terms(equations.flow_equations)
have_aux_node_vars(equations::PassiveTracerEquations) = have_aux_node_vars(equations.flow_equations)
n_aux_node_vars(equations::PassiveTracerEquations) = n_aux_node_vars(equations.flow_equations)

function varnames(variables::typeof(cons2cons),
                  tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    flow_varnames = varnames(variables, flow_equations)
    n_tracers = ntracers(tracer_equations)
    return (flow_varnames..., ntuple(i -> "rho_chi_$i", Val(n_tracers))...)
end

function varnames(variables::typeof(cons2prim),
                  tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    flow_varnames = varnames(variables, flow_equations)
    n_tracers = ntracers(tracer_equations)
    return (flow_varnames..., ntuple(i -> "chi_$i", Val(n_tracers))...)
end

# Calculate flux for a single point
@inline function flux(u, orientation_or_normal,
                      tracer_equations::PassiveTracerEquations)
    n_flow_variables = nvariables_flow(tracer_equations)

    u_flow = flow_variables(u, tracer_equations)

    v_normal = velocity(u_flow, orientation_or_normal, tracer_equations.flow_equations)

    flux_tracer = SVector(ntuple(@inline(v->u[v + n_flow_variables] * v_normal),
                                 Val(ntracers(tracer_equations))))

    flux_flow = flux(u_flow, orientation_or_normal, tracer_equations.flow_equations)

    return SVector(flux_flow..., flux_tracer...)
end

"""
    initial_condition_density_wave(x, t, equations::PassiveTracerEquations)

Takes the [`initial_condition_density_wave`](@ref) for the flow equations and
takes its translated first coordinates as the initial condition for the tracers.
"""
function initial_condition_density_wave(x, t,
                                        equations::PassiveTracerEquations)
    # Store translated coordinate for easy use of exact solution
    u_flow = initial_condition_density_wave(x, t, equations.flow_equations)
    # Obtain `u_tracers` by translating `u_flow`
    xc = SVector(ntuple(_ -> 0.1f0 * one(eltype(x)), Val(ndims(equations))))

    tracers = SVector((initial_condition_density_wave(x + i * xc, t,
                                                      equations.flow_equations)[1] for i in 1:ntracers(equations))...)

    u_tracers = u_flow[1] * tracers
    return SVector(u_flow..., u_tracers...)
end

# Calculate the number of variables for the flow equations
@inline function nvariables_flow(tracer_equations::PassiveTracerEquations)
    return nvariables(tracer_equations.flow_equations)
end

# Obtain the flow variables from the conservative variables. The tracers are not included.
@inline function flow_variables(u, tracer_equations::PassiveTracerEquations)
    n_flow_variables = nvariables_flow(tracer_equations)
    return SVector(ntuple(@inline(v->u[v]), Val(n_flow_variables)))
end

# Obtain the tracers which advect by the flow velocity by dividing the respective conservative
# variable by the density.
function tracers(u, tracer_equations::PassiveTracerEquations)
    n_flow_variables = nvariables_flow(tracer_equations)

    rho = density(u, tracer_equations)
    return SVector(ntuple(@inline(v->u[v + n_flow_variables] / rho),
                          Val(ntracers(tracer_equations))))
end
function tracers(u, aux, tracer_equations::PassiveTracerEquations)
    n_flow_variables = nvariables_flow(tracer_equations)

    rho = density(u, aux, tracer_equations)
    return SVector(ntuple(@inline(v->u[v + n_flow_variables] / rho),
                          Val(ntracers(tracer_equations))))
end

# Obtain rho * tracers which are the conservative variables for the tracer equations.
function rho_tracers(u, tracer_equations::PassiveTracerEquations)
    n_flow_variables = nvariables_flow(tracer_equations)

    return SVector(ntuple(@inline(v->u[v + n_flow_variables]),
                          Val(ntracers(tracer_equations))))
end

# Primitives for the flow equations and tracers. For a tracer, the primitive variable is obtained
# by dividing by density.
@inline function cons2prim(u, tracer_equations::PassiveTracerEquations)
    return SVector(cons2prim(flow_variables(u, tracer_equations),
                             tracer_equations.flow_equations)...,
                   tracers(u, tracer_equations)...)
end

# Conservative variables for the flow equations and tracers. For tracers, the conservative variable
# is obtained by multiplying by the density
@inline function prim2cons(u, tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    u_flow = flow_variables(u, tracer_equations)

    n_flow_variables = nvariables_flow(tracer_equations)

    rho = density(u, tracer_equations)
    cons_flow = prim2cons(u_flow, flow_equations)
    cons_tracer = SVector(ntuple(@inline(v->rho * u[v + n_flow_variables]),
                                 Val(ntracers(tracer_equations))))
    return SVector(cons_flow..., cons_tracer...)
end

# Entropy for tracers is the L2 norm of the tracers
@inline function entropy(cons, tracer_equations::PassiveTracerEquations)
    flow_entropy = entropy(flow_variables(cons, tracer_equations),
                           tracer_equations.flow_equations)
    tracer_entropy = density(cons, tracer_equations) *
                     sum(abs2, tracers(cons, tracer_equations))
    return flow_entropy + tracer_entropy
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    flow_entropy = cons2entropy(flow_variables(u, tracer_equations), flow_equations)
    tracers_ = tracers(u, tracer_equations)

    flow_entropy_after_density = SVector(ntuple(@inline(v->flow_entropy[v + 1]),
                                                Val(nvariables_flow(tracer_equations) -
                                                    1)))
    variables = SVector(flow_entropy[1] - sum(tracers_ .^ 2),
                        flow_entropy_after_density...,
                        2 * tracers_...) # factor of 2 because of the L2 norm
    return variables
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, aux, tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    flow_entropy = cons2entropy(flow_variables(u, tracer_equations), aux, flow_equations)
    tracers_ = tracers(u, aux, tracer_equations)

    flow_entropy_after_density = SVector(ntuple(@inline(v->flow_entropy[v + 1]),
                                                Val(nvariables_flow(tracer_equations) -
                                                    1)))
    variables = SVector(flow_entropy[1] - sum(tracers_ .^ 2),
                        flow_entropy_after_density...,
                        2 * tracers_...) # factor of 2 because of the L2 norm
    return variables
end

# Works if the method exists for flow equations
@inline function velocity(u, orientation_or_normal,
                          tracer_equations::PassiveTracerEquations)
    return velocity(flow_variables(u, tracer_equations), orientation_or_normal,
                    tracer_equations.flow_equations)
end

# Works if the method exists for flow equations
@inline function density(u, tracer_equations::PassiveTracerEquations)
    return density(flow_variables(u, tracer_equations), tracer_equations.flow_equations)
end

@inline function density(u, aux, tracer_equations::PassiveTracerEquations)
    return density(flow_variables(u, tracer_equations), aux, tracer_equations.flow_equations)
end

# Works if the method exists for flow equations
@inline function pressure(u, tracer_equations::PassiveTracerEquations)
    return pressure(flow_variables(u, tracer_equations),
                    tracer_equations.flow_equations)
end

# Works if the method exists for flow equations
@inline function density_pressure(u, tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    u_flow = flow_variables(u, tracer_equations)
    return density_pressure(u_flow, flow_equations)
end

# Used for local Lax-Friedrichs type dissipation, and uses only the flow equations
# This assumes that the `velocity` is always bounded by the estimate of the
# wave speed for the wrapped equations.
@inline function max_abs_speed_naive(u_ll, u_rr, orientation_or_normal,
                                     tracer_equations::PassiveTracerEquations)
    u_flow_ll = flow_variables(u_ll, tracer_equations)
    u_flow_rr = flow_variables(u_rr, tracer_equations)

    @unpack flow_equations = tracer_equations
    return max_abs_speed_naive(u_flow_ll, u_flow_rr, orientation_or_normal,
                               flow_equations)
end

@inline function max_abs_speeds(u, tracer_equations::PassiveTracerEquations)
    u_flow = flow_variables(u, tracer_equations)

    return max_abs_speeds(u_flow, tracer_equations.flow_equations)
end

"""
    FluxTracerEquationsCentral(flow_flux)

Get an entropy conserving flux for the equations with tracers corresponding to the given entropy conserving flux `flow_flux` for the flow equations. The study of this flux is part of ongoing 
research.
"""
struct FluxTracerEquationsCentral{FlowFlux}
    flow_flux::FlowFlux
end

# Entropy conserving (EC) flux for the tracer equations using EC `flow_flux` for the flow equations
@inline function (f::FluxTracerEquationsCentral)(u_ll, u_rr,
                                                 orientation_or_normal_direction,
                                                 tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    u_flow_ll = flow_variables(u_ll, tracer_equations)
    u_flow_rr = flow_variables(u_rr, tracer_equations)

    flux_flow = f.flow_flux(u_flow_ll, u_flow_rr, orientation_or_normal_direction,
                            flow_equations)
    flux_rho = density(flux_flow, flow_equations)
    tracers_ll = tracers(u_ll, tracer_equations)
    tracers_rr = tracers(u_rr, tracer_equations)
    flux_tracer = 0.5f0 * SVector(ntuple(@inline(v->tracers_ll[v] + tracers_rr[v]),
                                 Val(ntracers(tracer_equations))))
    flux_tracer = flux_rho * flux_tracer
    return SVector(flux_flow..., flux_tracer...)
end

@inline function (f::FluxTracerEquationsCentral)(u_ll, u_rr, aux_ll, aux_rr,
                                                 orientation_or_normal_direction,
                                                 tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    u_flow_ll = flow_variables(u_ll, tracer_equations)
    u_flow_rr = flow_variables(u_rr, tracer_equations)

    flux_flow = f.flow_flux(u_flow_ll, u_flow_rr, aux_ll, aux_rr,
                            orientation_or_normal_direction,
                            flow_equations)
    flux_rho = flux_flow[1]
    tracers_ll = tracers(u_ll, aux_ll, tracer_equations)
    tracers_rr = tracers(u_rr, aux_rr, tracer_equations)
    flux_tracer = 0.5f0 * SVector(ntuple(@inline(v->tracers_ll[v] + tracers_rr[v]),
                                 Val(ntracers(tracer_equations))))
    flux_tracer = flux_rho * flux_tracer
    return SVector(flux_flow..., flux_tracer...)
end

struct FluxTracerEquationsPass{FlowFlux}
    flow_flux::FlowFlux
end

function (f::FluxTracerEquationsPass)(u_ll, u_rr,
                                      orientation_or_normal_direction,
                                      tracer_equations::PassiveTracerEquations)
    (; flow_equations) = tracer_equations
    u_flow_ll = flow_variables(u_ll, tracer_equations)
    u_flow_rr = flow_variables(u_rr, tracer_equations)
    flux_flow = f.flow_flux(u_flow_ll, u_flow_rr, orientation_or_normal_direction,
                            flow_equations)
    flux_tracer = SVector(ntuple(@inline(v->zero(eltype(u_ll))),
                                 Val(ntracers(tracer_equations))))
    return SVector(flux_flow..., flux_tracer...)
end

function (f::FluxTracerEquationsPass)(u_ll, u_rr, aux_ll, aux_rr,
                                      orientation_or_normal_direction,
                                      tracer_equations::PassiveTracerEquations)
    (; flow_equations) = tracer_equations
    u_flow_ll = flow_variables(u_ll, tracer_equations)
    u_flow_rr = flow_variables(u_rr, tracer_equations)
    flux_flow = f.flow_flux(u_flow_ll, u_flow_rr, aux_ll, aux_rr,
                            orientation_or_normal_direction,
                            flow_equations)
    flux_tracer = SVector(ntuple(@inline(v->zero(eltype(u_ll))),
                                 Val(ntracers(tracer_equations))))
    return SVector(flux_flow..., flux_tracer...)
end

@inline function boundary_condition_slip_wall_noncons(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t,
                                                      surface_flux_function,
                                                      tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    u_flow = flow_variables(u_inner, tracer_equations)
    bc_flow, bc_flow_noncons = boundary_condition_slip_wall(u_flow, normal_direction, x,
                                                            t,
                                                            surface_flux_function,
                                                            flow_equations)
    bc_tracer = SVector(ntuple(@inline(v->0), Val(ntracers(tracer_equations))))
    return (vcat(bc_flow, bc_tracer), vcat(bc_flow_noncons, bc_tracer))
end

@inline function boundary_condition_slip_wall_noncons(u_inner, aux_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t,
                                                      surface_flux_function,
                                                      tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    u_flow = flow_variables(u_inner, tracer_equations)
    bc_flow, bc_flow_noncons = boundary_condition_slip_wall(u_flow, aux_inner,
                                                            normal_direction, x, t,
                                                            surface_flux_function,
                                                            flow_equations)
    bc_tracer = SVector(ntuple(@inline(v->0), Val(ntracers(tracer_equations))))
    return (vcat(bc_flow, bc_tracer), vcat(bc_flow_noncons, bc_tracer))
end

@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              x, t,
                                              surface_flux_function,
                                              tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations
    u_flow = flow_variables(u_inner, tracer_equations)
    bc_flow = boundary_condition_slip_wall(u_flow, normal_direction, x, t,
                                           surface_flux_function, flow_equations)
    bc_tracer = SVector(ntuple(@inline(v->0), Val(ntracers(tracer_equations))))
    return vcat(bc_flow, bc_tracer)
end
end # muladd
