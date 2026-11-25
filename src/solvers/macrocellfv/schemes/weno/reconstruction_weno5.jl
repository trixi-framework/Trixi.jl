# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Fifth-order WENO face reconstructions of the `MacroCellFV` solver from the subcell
# averages.

"""
    ReconstructionWENO5JS(; epsilon = 1.0e-6)

Fifth-order weighted essentially non-oscillatory (WENO) reconstruction of the face
states of each subcell from the five cell averages centered at the subcell, applied
componentwise to the reconstruction variables. `ReconstructionWENO5JS` uses the
nonlinear weights of

- Guang-Shan Jiang and Chi-Wang Shu (1996)
  Efficient implementation of weighted ENO schemes
  [DOI: 10.1006/jcph.1996.0130](https://doi.org/10.1006/jcph.1996.0130)

and `ReconstructionWENO5Z` the weights of

- Rafael Borges, Monique Carmona, Bruno Costa and Wai Sun Don (2008)
  An improved weighted essentially non-oscillatory scheme for hyperbolic
  conservation laws
  [DOI: 10.1016/j.jcp.2007.11.038](https://doi.org/10.1016/j.jcp.2007.11.038)

(with the exponent 1 of the global smoothness ratio), see
[`ReconstructionWENO5Z`](@ref). `epsilon` avoids divisions by zero in the nonlinear
weights.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionWENO5JS{RealT}
    epsilon::RealT
end

ReconstructionWENO5JS(; epsilon = 1.0e-6) = ReconstructionWENO5JS(epsilon)

"""
    ReconstructionWENO5Z(; epsilon = nothing)

WENO5 reconstruction with the WENO-Z weights of Borges et al. (2008); see
[`ReconstructionWENO5JS`](@ref). The default `epsilon = nothing` uses
`max(1.0e-40, sqrt(floatmin(RealT)))` in the floating point type `RealT` of the
solution, i.e., `1.0e-40` for `Float64` and about `1.1e-19` for `Float32`. A smaller
`epsilon` lets the factor `1 + tau5 / (beta + epsilon)` of a smooth substencil next to
a discontinuity overflow.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionWENO5Z{RealT}
    epsilon::RealT
end

ReconstructionWENO5Z(; epsilon = nothing) = ReconstructionWENO5Z(epsilon)

const ReconstructionWENO5 = Union{ReconstructionWENO5JS, ReconstructionWENO5Z}

function Base.show(io::IO, reconstruction::ReconstructionWENO5)
    print(io, nameof(typeof(reconstruction)), "(epsilon = ", reconstruction.epsilon,
          ")")
end

@inline nhalo(::ReconstructionWENO5) = 3

@inline function reconstruct_cell(reconstruction::ReconstructionWENO5, stencil)
    return weno5_cell(reconstruction, stencil(-2), stencil(-1), stencil(0), stencil(1),
                      stencil(2))
end

# WENO5 values at the faces between the cells `b` and `c` (`w_minus`) and between `c`
# and `d` (`w_plus`) of the cell `c` from the cell averages `a, b, c, d, e` (Jiang and
# Shu, 1996). The face in negative direction is the mirrored reconstruction: its
# smoothness indicators are those of the face in positive direction in reverse order,
# so both faces share the three nonlinear factors. The candidate values are not divided
# by 6 and the linear weights (1, 6, 3) / 10 not by 10; both constants are applied
# once in the normalization. This leaves 5 divisions per cell, which dominate its cost.
@inline function weno5_cell(reconstruction, a, b, c, d, e)
    # Smoothness indicators and nonlinear factors of the three third-order substencils
    beta0, beta1, beta2 = weno5_smoothness(a, b, c, d, e)
    s0, s1, s2 = weno5_factors(reconstruction, beta0, beta1, beta2)

    # 6 times the candidate values at the face in positive direction ...
    p0 = 2 * a - 7 * b + 11 * c
    p1 = -b + 5 * c + 2 * d
    p2 = 2 * c + 5 * d - e
    w_plus = (s0 * p0 + 6 * s1 * p1 + 3 * s2 * p2) / (6 * (s0 + 6 * s1 + 3 * s2))

    # ... and in negative direction (mirrored stencils, written as the exact mirror of
    # the expressions above so that mirrored data give mirrored face values bitwise)
    q0 = 2 * e - 7 * d + 11 * c
    q1 = -d + 5 * c + 2 * b
    q2 = 2 * c + 5 * b - a
    w_minus = (s2 * q0 + 6 * s1 * q1 + 3 * s0 * q2) / (6 * (s2 + 6 * s1 + 3 * s0))

    return w_minus, w_plus
end

# Smoothness indicators of Jiang and Shu (1996) of the substencils `(a, b, c)`,
# `(b, c, d)`, and `(c, d, e)`. The operations are mirror-symmetric: for the reversed
# data `(e, d, c, b, a)`, the indicators are `(beta2, beta1, beta0)` bitwise, so that
# mirrored data (e.g., at reflecting walls) give mirrored face values exactly.
@inline function weno5_smoothness(a, b, c, d, e)
    RealT = typeof(c)
    c_13_12 = convert(RealT, 13 // 12)
    beta0 = c_13_12 * (a - 2 * b + c)^2 + 0.25f0 * (a - 4 * b + 3 * c)^2
    beta1 = c_13_12 * ((b + d) - 2 * c)^2 + 0.25f0 * (b - d)^2
    beta2 = c_13_12 * (e - 2 * d + c)^2 + 0.25f0 * (e - 4 * d + 3 * c)^2
    return beta0, beta1, beta2
end

# Nonlinear factors of the substencils, which multiply the linear weights; the result
# is symmetric: `weno5_factors(r, beta2, beta1, beta0)` gives the factors in reverse order
@inline function weno5_factors(reconstruction::ReconstructionWENO5JS, beta0, beta1,
                               beta2)
    epsilon = convert(typeof(beta0), reconstruction.epsilon)
    return (1 / (epsilon + beta0)^2, 1 / (epsilon + beta1)^2, 1 / (epsilon + beta2)^2)
end

@inline function weno5_factors(reconstruction::ReconstructionWENO5Z, beta0, beta1,
                               beta2)
    epsilon = weno5z_epsilon(reconstruction.epsilon, typeof(beta0))
    tau5 = abs(beta0 - beta2)
    return (1 + tau5 / (beta0 + epsilon), 1 + tau5 / (beta1 + epsilon),
            1 + tau5 / (beta2 + epsilon))
end

# WENO5 reconstruction from the point values at the subcell centers (used with the
# moment inversions to point values, `MomentInversionCentral` and `MomentInversionWENO`,
# set by `point_reconstruction`): the face values are the WENO interpolations of the
# point values to the face midpoints (as in the weighted compact nonlinear schemes of
# Deng and Zhang, 2000, doi:10.1006/jcph.2000.6594), with the candidates of the stencils
# `(i-2, i-1, i)`, `(i-1, i, i+1)`, `(i, i+1, i+2)` and the linear weights
# `(1, 10, 5) / 16`, exact for quartic polynomials. The smoothness indicators and the
# nonlinear factors are those of the reconstruction from the averages.
struct ReconstructionPointValues{Reconstruction}
    reconstruction::Reconstruction
end

function Base.show(io::IO, reconstruction::ReconstructionPointValues)
    print(io, reconstruction.reconstruction, " (point values)")
end

@inline nhalo(reconstruction::ReconstructionPointValues) = nhalo(reconstruction.reconstruction)

@inline function reconstruct_cell(reconstruction::ReconstructionPointValues{<:ReconstructionWENO5},
                                  stencil)
    return weno5_point_cell(reconstruction.reconstruction, stencil(-2), stencil(-1),
                            stencil(0), stencil(1), stencil(2))
end

# WENO interpolation of the point values `a, b, c, d, e` at the centers of the cells
# `i-2, ..., i+2` to the faces of the cell `c`, written as `weno5_cell`: the candidate
# values are not divided by 8 and the linear weights not by 16; the face in negative
# direction is the mirrored interpolation with the same nonlinear factors.
@inline function weno5_point_cell(reconstruction, a, b, c, d, e)
    beta0, beta1, beta2 = weno5_smoothness(a, b, c, d, e)
    s0, s1, s2 = weno5_factors(reconstruction, beta0, beta1, beta2)

    # 8 times the candidate values at the face in positive direction ...
    p0 = 3 * a - 10 * b + 15 * c
    p1 = -b + 6 * c + 3 * d
    p2 = 3 * c + 6 * d - e
    w_plus = (s0 * p0 + 10 * s1 * p1 + 5 * s2 * p2) / (8 * (s0 + 10 * s1 + 5 * s2))

    # ... and in negative direction (exact mirror of the expressions above)
    q0 = 3 * e - 10 * d + 15 * c
    q1 = -d + 6 * c + 3 * b
    q2 = 3 * c + 6 * b - a
    w_minus = (s2 * q0 + 10 * s1 * q1 + 5 * s0 * q2) / (8 * (s2 + 10 * s1 + 5 * s0))

    return w_minus, w_plus
end

# `epsilon` of WENO-Z in the type `RealT` (default: `1.0e-40`, at least
# `sqrt(floatmin(RealT))`, see `ReconstructionWENO5Z`)
@inline weno5z_epsilon(epsilon, RealT) = convert(RealT, epsilon)
@inline function weno5z_epsilon(::Nothing, RealT)
    return max(convert(RealT, 1.0e-40), sqrt(floatmin(RealT)))
end
end # @muladd
