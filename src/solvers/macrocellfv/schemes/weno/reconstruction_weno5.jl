# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Fifth-order WENO face reconstructions of the `MacroCellFV` solver, from the subcell
# averages and from the point values at the subcell centers.

"""
    ReconstructionWENO5JS(; epsilon = 1.0e-6)
    ReconstructionWENO5Z(; epsilon = 1.0e-40)

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

(with the exponent 1 of the global smoothness ratio). `epsilon` avoids divisions by
zero in the nonlinear weights.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionWENO5JS{RealT}
    epsilon::RealT
end

ReconstructionWENO5JS(; epsilon = 1.0e-6) = ReconstructionWENO5JS(epsilon)

"""
    ReconstructionWENO5Z(; epsilon = 1.0e-40)

WENO5 reconstruction with the WENO-Z weights of Borges et al. (2008); see
[`ReconstructionWENO5JS`](@ref).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionWENO5Z{RealT}
    epsilon::RealT
end

ReconstructionWENO5Z(; epsilon = 1.0e-40) = ReconstructionWENO5Z(epsilon)

const ReconstructionWENO5 = Union{ReconstructionWENO5JS, ReconstructionWENO5Z}

@inline nhalo(::ReconstructionWENO5) = 3

@inline function reconstruct_cell(reconstruction::ReconstructionWENO5, stencil)
    u_m2 = stencil(-2)
    u_m1 = stencil(-1)
    u_0 = stencil(0)
    u_p1 = stencil(1)
    u_p2 = stencil(2)

    pairs = map(@inline((a, b, c, d, e)->weno5_cell(reconstruction, a, b, c, d, e)),
                u_m2, u_m1, u_0, u_p1, u_p2)
    return map(first, pairs), map(last, pairs)
end

# WENO5 values at the faces between the cells `b` and `c` (`w_minus`) and between `c`
# and `d` (`w_plus`) of the cell `c` from the cell averages `a, b, c, d, e` (Jiang and
# Shu, 1996). The face in negative direction is the mirrored reconstruction, whose
# smoothness indicators are those of the face in positive direction in reverse order.
@inline function weno5_cell(reconstruction, a, b, c, d, e)
    # Smoothness indicators of the three third-order substencils
    beta0, beta1, beta2 = weno5_smoothness(a, b, c, d, e)

    # Candidate values at the face in positive direction ...
    p0 = (2 * a - 7 * b + 11 * c) / 6
    p1 = (-b + 5 * c + 2 * d) / 6
    p2 = (2 * c + 5 * d - e) / 6
    alpha0, alpha1, alpha2 = weno5_alphas(reconstruction, beta0, beta1, beta2)
    w_plus = (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / (alpha0 + alpha1 + alpha2)

    # ... and in negative direction (mirrored stencils)
    q0 = (2 * e - 7 * d + 11 * c) / 6
    q1 = (-d + 5 * c + 2 * b) / 6
    q2 = (2 * c + 5 * b - a) / 6
    gamma0, gamma1, gamma2 = weno5_alphas(reconstruction, beta2, beta1, beta0)
    w_minus = (gamma0 * q0 + gamma1 * q1 + gamma2 * q2) / (gamma0 + gamma1 + gamma2)

    return w_minus, w_plus
end

# Smoothness indicators of Jiang and Shu (1996) of the substencils `(a, b, c)`,
# `(b, c, d)`, and `(c, d, e)`. The operations are mirror-symmetric: for the reversed
# data `(e, d, c, b, a)`, the indicators are `(beta2, beta1, beta0)` bitwise, so that
# mirrored data (e.g., at reflecting walls) give mirrored face values exactly, also
# through the decisions of the positivity limiter.
@inline function weno5_smoothness(a, b, c, d, e)
    RealT = typeof(c)
    c_13_12 = convert(RealT, 13 // 12)
    beta0 = c_13_12 * (a - 2 * b + c)^2 + 0.25f0 * (a - 4 * b + 3 * c)^2
    beta1 = c_13_12 * ((b + d) - 2 * c)^2 + 0.25f0 * (b - d)^2
    beta2 = c_13_12 * (e - 2 * d + c)^2 + 0.25f0 * (e - 4 * d + 3 * c)^2
    return beta0, beta1, beta2
end

# Unnormalized nonlinear weights with the linear weights (1, 6, 3) / 10
@inline function weno5_alphas(reconstruction::ReconstructionWENO5JS, beta0, beta1,
                              beta2)
    epsilon = convert(typeof(beta0), reconstruction.epsilon)
    return (1 / (10 * (epsilon + beta0)^2), 6 / (10 * (epsilon + beta1)^2),
            3 / (10 * (epsilon + beta2)^2))
end

@inline function weno5_alphas(reconstruction::ReconstructionWENO5Z, beta0, beta1,
                              beta2)
    epsilon = convert(typeof(beta0), reconstruction.epsilon)
    tau5 = abs(beta0 - beta2)
    return ((1 + tau5 / (beta0 + epsilon)) / 10,
            6 * (1 + tau5 / (beta1 + epsilon)) / 10,
            3 * (1 + tau5 / (beta2 + epsilon)) / 10)
end

@inline function reconstruct_cell(reconstruction::ReconstructionPointValues{<:ReconstructionWENO5},
                                  stencil)
    u_m2 = point_part(stencil(-2))
    u_m1 = point_part(stencil(-1))
    u_0 = point_part(stencil(0))
    u_p1 = point_part(stencil(1))
    u_p2 = point_part(stencil(2))
    weno = reconstruction.reconstruction
    pairs = map(@inline((a, b, c, d, e)->weno5_point_cell(weno, a, b, c, d, e)),
                u_m2, u_m1, u_0, u_p1, u_p2)
    return map(first, pairs), map(last, pairs)
end

# WENO interpolation of the point values `a, b, c, d, e` at the centers of the cells
# `i-2, ..., i+2` to the faces of the cell `c` (as `weno5_cell` for averages)
@inline function weno5_point_cell(reconstruction, a, b, c, d, e)
    beta0, beta1, beta2 = weno5_smoothness(a, b, c, d, e)

    p0 = (3 * a - 10 * b + 15 * c) / 8
    p1 = (-b + 6 * c + 3 * d) / 8
    p2 = (3 * c + 6 * d - e) / 8
    alpha0, alpha1, alpha2 = weno5_point_alphas(reconstruction, beta0, beta1, beta2)
    w_plus = (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / (alpha0 + alpha1 + alpha2)

    q0 = (3 * e - 10 * d + 15 * c) / 8
    q1 = (-d + 6 * c + 3 * b) / 8
    q2 = (3 * c + 6 * b - a) / 8
    gamma0, gamma1, gamma2 = weno5_point_alphas(reconstruction, beta2, beta1, beta0)
    w_minus = (gamma0 * q0 + gamma1 * q1 + gamma2 * q2) / (gamma0 + gamma1 + gamma2)

    return w_minus, w_plus
end

# Unnormalized nonlinear weights with the linear weights (1, 10, 5) / 16
@inline function weno5_point_alphas(reconstruction::ReconstructionWENO5JS, beta0,
                                    beta1, beta2)
    epsilon = convert(typeof(beta0), reconstruction.epsilon)
    return (1 / (16 * (epsilon + beta0)^2), 10 / (16 * (epsilon + beta1)^2),
            5 / (16 * (epsilon + beta2)^2))
end

@inline function weno5_point_alphas(reconstruction::ReconstructionWENO5Z, beta0,
                                    beta1, beta2)
    epsilon = convert(typeof(beta0), reconstruction.epsilon)
    tau5 = abs(beta0 - beta2)
    return ((1 + tau5 / (beta0 + epsilon)) / 16,
            10 * (1 + tau5 / (beta1 + epsilon)) / 16,
            5 * (1 + tau5 / (beta2 + epsilon)) / 16)
end
end # @muladd
