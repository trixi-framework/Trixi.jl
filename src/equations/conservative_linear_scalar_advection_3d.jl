# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    ConservativeLinearScalarAdvectionEquation3D()

The linear scalar advection equation in conservative form
```math
\partial_t u + \nabla \cdot \bigl( \mathbf{a}(\mathbf{x}) u \bigr) = 0
```
in three space dimensions with a spatially varying advection velocity
``\mathbf{a}(\mathbf{x}) = (a_1, a_2, a_3)``. In contrast to
[`LinearScalarAdvectionEquation3D`](@ref), the advection velocity is not constant.
Instead, it is stored as auxiliary variables at every node, see
[`nauxvariables`](@ref). Its values are given by the function `aux_field` that has to be
passed to [`SemidiscretizationHyperbolic`](@ref).

See also [`NonconservativeLinearScalarAdvectionEquation3D`](@ref) for the nonconservative
form.
"""
struct ConservativeLinearScalarAdvectionEquation3D <:
       AbstractLinearScalarAdvectionEquation{3} end

@inline nauxvariables(::ConservativeLinearScalarAdvectionEquation3D) = 3

# Together with our specialization of `Adapt.adapt_structure`, this allows to adapt the
# floating point type of semidiscretizations. Since the advection velocity is stored as
# auxiliary variables, the equations do not store any data depending on the real type.
function Base.similar(equations::ConservativeLinearScalarAdvectionEquation3D, ::Type)
    return equations
end

varnames(::typeof(cons2cons), ::ConservativeLinearScalarAdvectionEquation3D) = ("scalar",)
varnames(::typeof(cons2prim), ::ConservativeLinearScalarAdvectionEquation3D) = ("scalar",)
auxvarnames(::ConservativeLinearScalarAdvectionEquation3D) = ("a1", "a2", "a3")

# Calculate 1D flux for a single point using the advection velocity from the auxiliary
# variables
@inline function flux(u, aux, orientation::Integer,
                      equations::ConservativeLinearScalarAdvectionEquation3D)
    a = aux[orientation]
    return a * u
end

# Calculate 1D flux for a single point in the normal direction using the advection velocity
# from the auxiliary variables
@inline function flux(u, aux, normal_direction::AbstractVector,
                      equations::ConservativeLinearScalarAdvectionEquation3D)
    # Advection velocity in the (not necessarily normalized) normal direction
    a = dot(aux, normal_direction)
    return a * u
end

"""
    have_constant_speed(::ConservativeLinearScalarAdvectionEquation3D)

Indicates whether the characteristic speeds are constant, i.e., independent of the solution.
Queried in the timestep computation [`StepsizeCallback`](@ref) and [`linear_structure`](@ref).
The characteristic speeds are given by the advection velocity, which varies in space.

# Returns
- `False()`
"""
@inline have_constant_speed(::ConservativeLinearScalarAdvectionEquation3D) = False()

# Maximum wave speeds in each direction for the CFL condition
@inline function max_abs_speeds(u, aux,
                                equations::ConservativeLinearScalarAdvectionEquation3D)
    return abs(aux[1]), abs(aux[2]), abs(aux[3])
end

@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                               equations::ConservativeLinearScalarAdvectionEquation3D)
    return max(abs(aux_ll[orientation]), abs(aux_rr[orientation]))
end

@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr,
                               normal_direction::AbstractVector,
                               equations::ConservativeLinearScalarAdvectionEquation3D)
    return max(abs(dot(aux_ll, normal_direction)), abs(dot(aux_rr, normal_direction)))
end

# Convert conservative variables to primitive and entropy variables
@inline cons2prim(u, ::ConservativeLinearScalarAdvectionEquation3D) = u
@inline cons2entropy(u, ::ConservativeLinearScalarAdvectionEquation3D) = u
end # @muladd
