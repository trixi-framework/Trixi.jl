# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Nonlinear (WENO) moment inversion of the `MacroCellFV` solver. The smoothness
# indicators are shared with the WENO5 reconstructions (`reconstruction_weno5.jl`).

"""
    MomentInversionWENO(; linear_weight = 1/6, epsilon = 1.0e-40)

Nonlinear (WENO) moment inversion: the point value at the center of each subcell is
the combination of the three third-order values from the subcell averages of the
stencils `(i-2, i-1, i)`, `(i-1, i, i+1)`, and `(i, i+1, i+2)` along each coordinate
direction, with the WENO-Z weights (Borges et al., 2008) for the symmetric positive
linear weights `(linear_weight, 1 - 2 linear_weight, linear_weight)` and the
smoothness indicators of Jiang and Shu (1996). The symmetric linear weights make the
combination exact for cubic polynomials (fourth order); the linear weights of the
fifth-order value would be negative. Next to discontinuities, the smooth one-sided
stencil is selected, so that the point values of a uniform state next to a jump (e.g.,
at a material interface at constant pressure and velocity) keep the uniform state.
Where the stencil does not fit into the available subcells (ends of the padded array),
the central inversion of [`MomentInversionCentral`](@ref) is used.

See [`MomentInversionCentral`](@ref) for the use of the point values.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct MomentInversionWENO{RealT}
    linear_weight::RealT
    epsilon::RealT
end

function MomentInversionWENO(; linear_weight = 1 / 6, epsilon = 1.0e-40)
    0 < linear_weight < 0.5 ||
        throw(ArgumentError("`linear_weight` must be in (0, 1/2)"))
    linear_weight, epsilon = promote(linear_weight, epsilon)
    return MomentInversionWENO(linear_weight, epsilon)
end

@inline function center_correction(inversion::MomentInversionWENO, line, lo, hi)
    if lo <= -2 && hi >= 2
        return map(@inline((a, b, c, d, e)->weno_center_correction(inversion, a, b, c,
                                                                   d, e)),
                   line(-2), line(-1), line(0), line(1), line(2))
    else
        return center_correction(MomentInversionCentral(), line, lo, hi)
    end
end

# WENO combination of the third-order center values `c - Δ²/24` of the stencils
# `(a, b, c)`, `(b, c, d)`, and `(c, d, e)`, minus the average `c`
@inline function weno_center_correction(inversion::MomentInversionWENO, a, b, c, d, e)
    RealT = typeof(c)
    beta0, beta1, beta2 = weno5_smoothness(a, b, c, d, e)
    epsilon = convert(RealT, inversion.epsilon)
    d_outer = convert(RealT, inversion.linear_weight)
    tau = abs(beta0 - beta2)
    alpha0 = d_outer * (1 + tau / (beta0 + epsilon))
    alpha1 = (1 - 2 * d_outer) * (1 + tau / (beta1 + epsilon))
    alpha2 = d_outer * (1 + tau / (beta2 + epsilon))
    # (mirror-symmetric operations, see `weno5_smoothness`)
    second_difference = ((alpha0 * (a - 2 * b + c) + alpha2 * (e - 2 * d + c)) +
                         alpha1 * ((b + d) - 2 * c)) / ((alpha0 + alpha2) + alpha1)
    return -second_difference / 24
end
end # @muladd
