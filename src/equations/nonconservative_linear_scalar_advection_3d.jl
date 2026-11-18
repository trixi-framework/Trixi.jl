# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    NonconservativeLinearScalarAdvectionEquation3D()

The linear scalar advection equation in nonconservative form
```math
\partial_t u + \mathbf{a}(\mathbf{x}) \cdot \nabla u = 0
```
in three space dimensions with a spatially varying advection velocity
``\mathbf{a}(\mathbf{x}) = (a_1, a_2, a_3)``. The advection velocity is stored as
auxiliary variables at every node, see [`nauxvariables`](@ref). Its values are given by
the function `aux_field` that has to be passed to [`SemidiscretizationHyperbolic`](@ref).

The physical flux vanishes and the advection term is discretized with the nonconservative
flux [`flux_nonconservative_central`](@ref), e.g., using
`volume_integral = VolumeIntegralFluxDifferencing((flux_central, flux_nonconservative_central))`
and `surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)`.

See also [`ConservativeLinearScalarAdvectionEquation3D`](@ref) for the conservative form.
For a divergence-free advection velocity, both PDE forms are equivalent
(but their discretizations are not necessarily equivalent).
"""
struct NonconservativeLinearScalarAdvectionEquation3D <:
       AbstractLinearScalarAdvectionEquation{3} end

@inline nauxvariables(::NonconservativeLinearScalarAdvectionEquation3D) = 3

have_nonconservative_terms(::NonconservativeLinearScalarAdvectionEquation3D) = True()

# Together with our specialization of `Adapt.adapt_structure`, this allows to adapt the
# floating point type of semidiscretizations. Since the advection velocity is stored as
# auxiliary variables, the equations do not store any data depending on the real type.
function Base.similar(equations::NonconservativeLinearScalarAdvectionEquation3D, ::Type)
    return equations
end

function varnames(::typeof(cons2cons), ::NonconservativeLinearScalarAdvectionEquation3D)
    return ("scalar",)
end
function varnames(::typeof(cons2prim), ::NonconservativeLinearScalarAdvectionEquation3D)
    return ("scalar",)
end
auxvarnames(::NonconservativeLinearScalarAdvectionEquation3D) = ("a1", "a2", "a3")

# The physical flux of the nonconservative form vanishes. It is still required, e.g., for
# the dissipation of `flux_lax_friedrichs`.
@inline function flux(u, aux, orientation_or_normal_direction,
                      equations::NonconservativeLinearScalarAdvectionEquation3D)
    return zero(u)
end

"""
    flux_nonconservative_central(u_ll, u_rr, aux_ll, aux_rr,
                                 normal_direction::AbstractVector,
                                 equations::NonconservativeLinearScalarAdvectionEquation3D)

Central nonconservative two-point "flux" for the advection term ``\\mathbf{a} \\cdot \\nabla u``,
written as the product of the local advection velocity and the sum of the states.
The use of this term together with [`flux_central`](@ref) with
[`VolumeIntegralFluxDifferencing`](@ref) yields a "standard" (weak-form) DGSEM
discretization. Together with `surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)`,
it results in a standard local Lax-Friedrichs flux.
"""
@inline function flux_nonconservative_central(u_ll, u_rr, aux_ll, aux_rr,
                                              normal_direction::AbstractVector,
                                              equations::NonconservativeLinearScalarAdvectionEquation3D)
    # Advection velocity of the element receiving the flux in the (not necessarily
    # normalized) normal direction
    a_ll = dot(aux_ll, normal_direction)
    # For flux differencing using nonconservative terms, we return the nonconservative
    # flux scaled by 2. This cancels with a factor of 0.5 in the arithmetic average of {u}.
    return a_ll * (u_ll + u_rr)
end

@inline function flux_nonconservative_central(u_ll, u_rr, aux_ll, aux_rr,
                                              orientation::Integer,
                                              equations::NonconservativeLinearScalarAdvectionEquation3D)
    a_ll = aux_ll[orientation]
    # For flux differencing using nonconservative terms, we return the nonconservative
    # flux scaled by 2. This cancels with a factor of 0.5 in the arithmetic average of {u}.
    return a_ll * (u_ll + u_rr)
end

# TODO: Right now, this is somewhat confusing. The speeds are indeed constant
#       in time, but not constant in space. This trait is used, e.g., to make
#       the calculation of the time step size for CFL-based step size control
#       more efficient. We could compute the step size factor only once and
#       adapt it when the mesh changes during AMR.
@inline have_constant_speed(::NonconservativeLinearScalarAdvectionEquation3D) = False()

# Maximum wave speeds in each direction for the CFL condition
@inline function max_abs_speeds(u, aux,
                                equations::NonconservativeLinearScalarAdvectionEquation3D)
    return abs(aux[1]), abs(aux[2]), abs(aux[3])
end

@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                               equations::NonconservativeLinearScalarAdvectionEquation3D)
    return max(abs(aux_ll[orientation]), abs(aux_rr[orientation]))
end

@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr,
                               normal_direction::AbstractVector,
                               equations::NonconservativeLinearScalarAdvectionEquation3D)
    return max(abs(dot(aux_ll, normal_direction)), abs(dot(aux_rr, normal_direction)))
end

# Convert conservative variables to primitive and entropy variables
@inline cons2prim(u, ::NonconservativeLinearScalarAdvectionEquation3D) = u
@inline cons2entropy(u, ::NonconservativeLinearScalarAdvectionEquation3D) = u
end # @muladd
