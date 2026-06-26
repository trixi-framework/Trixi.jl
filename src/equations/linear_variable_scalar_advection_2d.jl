# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct LinearVariableScalarAdvectionEquation2D{} <:
       AbstractLinearScalarAdvectionEquation{2} end

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

function flux_godunov(u_ll, u_rr, aux_ll, aux_rr, normal_direction::AbstractVector,
                      equation::LinearVariableScalarAdvectionEquation2D)
    # velocity in normal direction
    v_ll = dot(aux_ll, normal_direction)
    v_rr = dot(aux_rr, normal_direction)

    a_normal = 0.5f0 * (v_ll + v_rr)
    if a_normal >= 0
        return v_ll * u_ll
    else
        return v_rr * u_rr
    end
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr,
                               orientation::Integer,
                               equation::LinearVariableScalarAdvectionEquation2D)
    v_ll = aux_ll[orientation]
    v_rr = aux_rr[orientation]
    return max(abs(v_ll), abs(v_rr))
end

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

@inline cons2entropy(u, aux, ::LinearVariableScalarAdvectionEquation2D) = u
@inline cons2prim(u, aux, ::LinearVariableScalarAdvectionEquation2D) = SVector(u[1],
                                                                               aux[1],
                                                                               aux[2])
@inline cons2aux(u, aux, ::LinearVariableScalarAdvectionEquation2D) = SVector(aux[1],
                                                                              aux[2])

varnames(::typeof(cons2cons), ::LinearVariableScalarAdvectionEquation2D) = ("scalar",)
varnames(::typeof(cons2prim), ::LinearVariableScalarAdvectionEquation2D) = ("scalar",
                                                                            "v1", "v2")
varnames(::typeof(cons2aux), ::LinearVariableScalarAdvectionEquation2D) = ("v1", "v2")
end
