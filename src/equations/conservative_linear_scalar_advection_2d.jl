# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    ConservativeLinearScalarAdvectionEquation2D()

The linear scalar advection equation in conservative form
```math
\partial_t u + \nabla \cdot \bigl( \mathbf{a}(\mathbf{x}) u \bigr) = 0
```
in two space dimensions with a spatially varying advection velocity
``\mathbf{a}(\mathbf{x}) = (a_1, a_2)``. In contrast to
[`LinearScalarAdvectionEquation2D`](@ref), the advection velocity is not constant.
Instead, it is stored as auxiliary variables at every node, see
[`nauxvariables`](@ref). Its values are given by the function `aux_field` that has to be
passed to [`SemidiscretizationHyperbolic`](@ref).

See also [`NonconservativeLinearScalarAdvectionEquation2D`](@ref) for the nonconservative
form.
"""
struct ConservativeLinearScalarAdvectionEquation2D <:
       AbstractLinearScalarAdvectionEquation{2} end

@inline nauxvariables(::ConservativeLinearScalarAdvectionEquation2D) = 2

# Together with our specialization of `Adapt.adapt_structure`, this allows to adapt the
# floating point type of semidiscretizations. Since the advection velocity is stored as
# auxiliary variables, the equations do not store any data depending on the real type.
function Base.similar(equations::ConservativeLinearScalarAdvectionEquation2D, ::Type)
    return equations
end

varnames(::typeof(cons2cons), ::ConservativeLinearScalarAdvectionEquation2D) = ("scalar",)
varnames(::typeof(cons2prim), ::ConservativeLinearScalarAdvectionEquation2D) = ("scalar",)
auxvarnames(::ConservativeLinearScalarAdvectionEquation2D) = ("a1", "a2")

# Calculate 1D flux for a single point using the advection velocity from the auxiliary
# variables
@inline function flux(u, aux, orientation::Integer,
                      equations::ConservativeLinearScalarAdvectionEquation2D)
    a = aux[orientation]
    return a * u
end

# Calculate 1D flux for a single point in the normal direction using the advection velocity
# from the auxiliary variables
@inline function flux(u, aux, normal_direction::AbstractVector,
                      equations::ConservativeLinearScalarAdvectionEquation2D)
    # Advection velocity in the (not necessarily normalized) normal direction
    a = dot(aux, normal_direction)
    return a * u
end

# TODO: Right now, this is somewhat confusing. The speeds are indeed constant
#       in time, but not constant in space. This trait is used, e.g., to make
#       the calculation of the time step size for CFL-based step size control
#       more efficient. We could compute the step size factor only once and
#       adapt it when the mesh changes during AMR.
@inline have_constant_speed(::ConservativeLinearScalarAdvectionEquation2D) = False()

# Maximum wave speeds in each direction for the CFL condition
@inline function max_abs_speeds(u, aux,
                                equations::ConservativeLinearScalarAdvectionEquation2D)
    return abs(aux[1]), abs(aux[2])
end

@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                               equations::ConservativeLinearScalarAdvectionEquation2D)
    return max(abs(aux_ll[orientation]), abs(aux_rr[orientation]))
end

@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr,
                               normal_direction::AbstractVector,
                               equations::ConservativeLinearScalarAdvectionEquation2D)
    return max(abs(dot(aux_ll, normal_direction)), abs(dot(aux_rr, normal_direction)))
end

# Convert conservative variables to primitive and entropy variables
@inline cons2prim(u, ::ConservativeLinearScalarAdvectionEquation2D) = u
@inline cons2entropy(u, ::ConservativeLinearScalarAdvectionEquation2D) = u
end # @muladd
