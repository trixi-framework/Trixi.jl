# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Nonlinear (WENO) moment inversion of the `MacroCellFV` solver. The smoothness
# indicators are shared with the WENO5 reconstructions (`reconstruction_weno5.jl`).

"""
    MomentInversionWENO(; linear_weight = 1/6, epsilon = nothing)

Nonlinear (WENO) moment inversion: the point value at the center of each subcell is
the combination of the three third-order values from the subcell averages of the
stencils `(i-2, i-1, i)`, `(i-1, i, i+1)`, and `(i, i+1, i+2)`, with the WENO-Z weights
(Borges et al., 2008, see [`ReconstructionWENO5Z`](@ref)) for the symmetric linear
weights `(linear_weight, 1 - 2 linear_weight, linear_weight)` and the smoothness
indicators of Jiang and Shu (1996). The symmetric linear weights make the combination
exact for cubic polynomials (fourth order); the linear weights of the fifth-order value
would be negative. Next to discontinuities, the smooth one-sided stencil is selected,
so that the point values of a uniform state next to a jump (e.g., at a material
interface at constant pressure and velocity) keep the uniform state. The moment
inversion reads 2 halo subcells on each side. The default `epsilon = nothing` is the
same as for [`ReconstructionWENO5Z`](@ref).

See [`MomentInversionCentral`](@ref) for the use of the point values.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct MomentInversionWENO{RealT, Epsilon}
    linear_weight::RealT
    epsilon::Epsilon
end

function MomentInversionWENO(; linear_weight = 1 / 6, epsilon = nothing)
    0 < linear_weight < 0.5 ||
        throw(ArgumentError("`linear_weight` must be in (0, 1/2)"))
    return MomentInversionWENO(linear_weight, epsilon)
end

function Base.show(io::IO, inversion::MomentInversionWENO)
    print(io, "MomentInversionWENO(linear_weight = ", inversion.linear_weight,
          ", epsilon = ", inversion.epsilon, ")")
end

@inline nhalo(::MomentInversionWENO) = 2

@inline has_point_values(::MomentInversionWENO) = True()

@inline function center_value(inversion::MomentInversionWENO, stencil)
    return weno_center_value(inversion, stencil(-2), stencil(-1), stencil(0),
                             stencil(1), stencil(2))
end

# WENO combination of the third-order center values `c - Δ²/24` of the stencils
# `(a, b, c)`, `(b, c, d)`, and `(c, d, e)`
@inline function weno_center_value(inversion::MomentInversionWENO, a, b, c, d, e)
    RealT = typeof(c)
    beta0, beta1, beta2 = weno5_smoothness(a, b, c, d, e)
    epsilon = weno5z_epsilon(inversion.epsilon, RealT)
    d_outer = convert(RealT, inversion.linear_weight)
    tau = abs(beta0 - beta2)
    alpha0 = d_outer * (1 + tau / (beta0 + epsilon))
    alpha1 = (1 - 2 * d_outer) * (1 + tau / (beta1 + epsilon))
    alpha2 = d_outer * (1 + tau / (beta2 + epsilon))
    # (mirror-symmetric operations, see `weno5_smoothness`)
    second_difference = ((alpha0 * (a - 2 * b + c) + alpha2 * (e - 2 * d + c)) +
                         alpha1 * ((b + d) - 2 * c)) / ((alpha0 + alpha2) + alpha1)
    return c - second_difference / 24
end
end # @muladd
