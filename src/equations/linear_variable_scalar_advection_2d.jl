@muladd begin
struct LinearVariableScalarAdvectionEquation2D{} <:
        AbstractLinearScalarAdvectionEquation{2, 1} end

varnames(::typeof(cons2cons), ::LinearVariableScalarAdvectionEquation2D) = ("scalar",)
varnames(::typeof(cons2prim), ::LinearVariableScalarAdvectionEquation2D) = ("scalar",)
varnames(::typeof(cons2aux), ::LinearVariableScalarAdvectionEquation2D) = ("v1", "v2")

have_aux_node_vars(::LinearVariableScalarAdvectionEquation2D) = True()
n_aux_node_vars(::LinearVariableScalarAdvectionEquation2D) = 2

@inline function flux(u, aux_vars, orientation::Integer,
                        equations::LinearVariableScalarAdvectionEquation2D)
    a = aux_vars[orientation]
    return a * u
end

@inline function flux(u, aux_vars, normal_direction::AbstractVector,
                        equation::LinearVariableScalarAdvectionEquation2D)
    a = dot(aux_vars, normal_direction) # velocity in normal direction
    return a * u
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr,
                                        normal_direction::AbstractVector,
                                        equation::LinearVariableScalarAdvectionEquation2D)
    # velocity in normal direction
    v_ll = dot(aux_ll, normal_direction)
    v_rr = dot(aux_rr, normal_direction)
    return max(abs(v_ll), abs(v_rr))
end

# Maximum wave speeds in each direction for CFL calculation
@inline function Trixi.max_abs_speeds(u, aux_vars,
                                        equations::LinearVariableScalarAdvectionEquation2D)
    return abs.(aux_vars)
end

@inline cons2entropy(u, aux, equations::LinearVariableScalarAdvectionEquation2D) = u
@inline cons2prim(u, aux, equations::LinearVariableScalarAdvectionEquation2D) = u
@inline cons2prim(u, equations::LinearVariableScalarAdvectionEquation2D) = u
end
