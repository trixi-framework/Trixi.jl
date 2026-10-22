# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearizedEulerEquations3DAuxVars()

Linearized Euler equations in three space dimensions with a spatially varying mean flow.
The equations are the same as for [`LinearizedEulerEquations3D`](@ref), but the mean flow
variables
```math
(\bar{\rho}, \bar{v}_1, \bar{v}_2, \bar{v}_3, \bar{c})
```
are not constant. Instead, they are stored as auxiliary variables at every node, see
[`n_aux_node_vars`](@ref). Their values are given by the function `aux_field` that
has to be passed to [`SemidiscretizationHyperbolic`](@ref).
"""
struct LinearizedEulerEquations3DAuxVars <: AbstractLinearizedEulerEquations{3, 5} end

@inline n_aux_node_vars(::LinearizedEulerEquations3DAuxVars) = 5

# Together with our specialization of `Adapt.adapt_structure`, this allows to adapt the
# floating point type of semidiscretizations. Since all mean flow variables are stored as
# auxiliary variables, the equations do not store any data depending on the real type.
function Base.similar(equations::LinearizedEulerEquations3DAuxVars, ::Type)
    return equations
end

function varnames(::typeof(cons2cons), ::LinearizedEulerEquations3DAuxVars)
    return ("rho_prime", "v1_prime", "v2_prime", "v3_prime", "p_prime")
end

function varnames(::typeof(cons2prim), ::LinearizedEulerEquations3DAuxVars)
    return ("rho_prime", "v1_prime", "v2_prime", "v3_prime", "p_prime")
end

function varnames(::typeof(cons2aux), ::LinearizedEulerEquations3DAuxVars)
    return ("rho_mean", "v1_mean", "v2_mean", "v3_mean", "c_mean")
end

# Calculate 1D flux for a single point using the mean flow from the auxiliary variables
@inline function flux(u, aux, orientation::Integer,
                      equations::LinearizedEulerEquations3DAuxVars)
    rho_mean, v1_mean, v2_mean, v3_mean, c_mean = aux
    rho_prime, v1_prime, v2_prime, v3_prime, p_prime = u

    v_mean = SVector(v1_mean, v2_mean, v3_mean)[orientation]
    v_prime = SVector(v1_prime, v2_prime, v3_prime)[orientation]

    f1 = v_mean * rho_prime + rho_mean * v_prime
    f2 = v_mean * v1_prime + (orientation == 1) * p_prime / rho_mean
    f3 = v_mean * v2_prime + (orientation == 2) * p_prime / rho_mean
    f4 = v_mean * v3_prime + (orientation == 3) * p_prime / rho_mean
    f5 = v_mean * p_prime + c_mean^2 * rho_mean * v_prime

    return SVector(f1, f2, f3, f4, f5)
end

# Calculate 1D flux for a single point in the normal direction using the mean flow from
# the auxiliary variables
@inline function flux(u, aux, normal_direction::AbstractVector,
                      equations::LinearizedEulerEquations3DAuxVars)
    rho_mean, v1_mean, v2_mean, v3_mean, c_mean = aux
    rho_prime, v1_prime, v2_prime, v3_prime, p_prime = u

    v_mean_normal = v1_mean * normal_direction[1] + v2_mean * normal_direction[2] +
                    v3_mean * normal_direction[3]
    v_prime_normal = v1_prime * normal_direction[1] + v2_prime * normal_direction[2] +
                     v3_prime * normal_direction[3]

    f1 = v_mean_normal * rho_prime + rho_mean * v_prime_normal
    f2 = v_mean_normal * v1_prime + normal_direction[1] * p_prime / rho_mean
    f3 = v_mean_normal * v2_prime + normal_direction[2] * p_prime / rho_mean
    f4 = v_mean_normal * v3_prime + normal_direction[3] * p_prime / rho_mean
    f5 = v_mean_normal * p_prime + c_mean^2 * rho_mean * v_prime_normal

    return SVector(f1, f2, f3, f4, f5)
end

"""
    have_constant_speed(::LinearizedEulerEquations3DAuxVars)

Indicates whether the characteristic speeds are constant, i.e., independent of the solution.
Queried in the timestep computation [`StepsizeCallback`](@ref) and [`linear_structure`](@ref).
The characteristic speeds depend on the mean flow, which varies in space.

# Returns
- `False()`
"""
@inline have_constant_speed(::LinearizedEulerEquations3DAuxVars) = False()

@inline function max_abs_speeds(u, aux, equations::LinearizedEulerEquations3DAuxVars)
    rho_mean, v1_mean, v2_mean, v3_mean, c_mean = aux
    return abs(v1_mean) + c_mean, abs(v2_mean) + c_mean, abs(v3_mean) + c_mean
end

@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                               equations::LinearizedEulerEquations3DAuxVars)
    v_mean_ll = aux_ll[orientation + 1]
    v_mean_rr = aux_rr[orientation + 1]
    return max(abs(v_mean_ll) + aux_ll[5], abs(v_mean_rr) + aux_rr[5])
end

@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr,
                               normal_direction::AbstractVector,
                               equations::LinearizedEulerEquations3DAuxVars)
    norm_ = norm(normal_direction)
    v_mean_normal_ll = aux_ll[2] * normal_direction[1] +
                       aux_ll[3] * normal_direction[2] +
                       aux_ll[4] * normal_direction[3]
    v_mean_normal_rr = aux_rr[2] * normal_direction[1] +
                       aux_rr[3] * normal_direction[2] +
                       aux_rr[4] * normal_direction[3]
    return max(abs(v_mean_normal_ll) + aux_ll[5] * norm_,
               abs(v_mean_normal_rr) + aux_rr[5] * norm_)
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearizedEulerEquations3DAuxVars) = u
@inline cons2entropy(u, ::LinearizedEulerEquations3DAuxVars) = u
end # muladd
