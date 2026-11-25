# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Moment inversions of the `MacroCellFV` solver (cell averages -> point values at the
# subcell centers) without nonlinear weights; see `weno/moment_inversion_weno.jl` for
# the WENO moment inversion.
#
# `center_value(inversion, stencil)` returns the point value of one variable at the
# center of a subcell, where `stencil(offset)` returns the average of the subcell
# `offset` cells away along the line. As the reconstructions, the moment inversions are
# applied to each variable separately, so that the loops over the subcells vectorize.

"""
    MomentInversionIdentity()

Moment inversion that uses the cell averages directly as point values at the
cell centers. This is second-order accurate.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct MomentInversionIdentity end

# Number of ghost subcells required by the moment inversion stencil
@inline nhalo(::MomentInversionIdentity) = 0

# `True()` if the moment inversion provides point values at the subcell centers (stored
# in a second pad, see `pad_moment_inversion!`)
@inline has_point_values(::MomentInversionIdentity) = False()

"""
    MomentInversionCentral(; order = 6)

Linear moment inversion of even `order >= 4`: the point value at the center of each
subcell is the combination of the averages of the `order - 1` subcells centered at the
subcell that is exact for polynomials of degree `order - 1`, written as
`ū_0 + Σ_k c_k (ū_{-k} + ū_k - 2 ū_0)` for `k = 1, ..., order/2 - 1`. For `order = 4`,
the coefficient is `c_1 = -1/24`. The moment inversion reads `order/2 - 1` halo subcells
on each side. With the default `order = 6`, the WENO5 interpolation of the point values
limits the order of the scheme to five; `order = 4` needs one halo subcell less.

The point values are converted to the reconstruction variables (`center2recon` of
[`MacroCellFV`](@ref)) pointwise, and the WENO reconstructions interpolate the point
values to the faces with the weights for point values. This removes the second-order
error of the reconstruction of nonlinear functions of the averages, e.g., of the
primitive variables. The order of the scheme is the lower one of the moment inversion
and the reconstruction (fifth order for the WENO5 reconstructions).

Across discontinuities, the central moment inversion can create oscillations. Use
[`MomentInversionWENO`](@ref) for flows with discontinuities. Point values that are not
admissible (e.g., negative density or pressure of the compressible Euler equations) are
replaced by the averages.

Supported with the WENO reconstructions.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct MomentInversionCentral{M}
    order::Int
    # Coefficients `c_1, ..., c_M` of the correction `Σ_k c_k (ū_{-k} + ū_k - 2 ū_0)`
    coefficients::NTuple{M, Float64}
end

function MomentInversionCentral(; order = 6)
    (order >= 4 && iseven(order)) ||
        throw(ArgumentError("`order` must be an even integer of at least 4"))
    m = order ÷ 2 - 1
    coefficients = central_inversion_coefficients(m)
    return MomentInversionCentral{m}(Int(order), Tuple(Float64.(coefficients)))
end

function Base.show(io::IO, inversion::MomentInversionCentral)
    print(io, "MomentInversionCentral(order = ", inversion.order, ")")
end

# Coefficients `c_1, ..., c_m` of the symmetric moment inversion over the subcells
# `-m, ..., m` (exact rational arithmetic): `c_0 = 1 - 2 Σ c_k` and the conditions that
# the averages of `x^p` (even `p <= 2 m`, the odd ones hold by symmetry) over the
# subcells of unit width give the value at the center `x = 0`
function central_inversion_coefficients(m)
    # Average of `x^p` over the subcell `k`
    average(k, p) = ((k + 1 // 2)^(p + 1) - (k - 1 // 2)^(p + 1)) // (p + 1)
    A = zeros(Rational{BigInt}, m + 1, m + 1)
    b = zeros(Rational{BigInt}, m + 1)
    for (row, p) in enumerate(0:2:(2 * m))
        A[row, 1] = average(0, p)
        for k in 1:m
            A[row, k + 1] = 2 * average(k, p)
        end
    end
    b[1] = 1
    c = A \ b
    return c[2:end]
end

@inline nhalo(::MomentInversionCentral{M}) where {M} = M

@inline has_point_values(::MomentInversionCentral) = True()

@inline function center_value(inversion::MomentInversionCentral{M},
                              stencil) where {M}
    @unpack coefficients = inversion
    u_0 = stencil(0)
    RealT = typeof(u_0)
    # (mirror-symmetric operations: mirrored data give the same value up to round-off)
    correction = convert(RealT, coefficients[1]) *
                 ((stencil(-1) + stencil(1)) - 2 * u_0)
    for k in 2:M
        correction = correction +
                     convert(RealT, coefficients[k]) *
                     ((stencil(-k) + stencil(k)) - 2 * u_0)
    end
    return u_0 + correction
end

# `true` if the point value `u` (conservative variables) can replace the average in the
# pointwise transformation to the reconstruction variables; otherwise, the average is
# used as point value of the subcell
@inline macrocell_point_value_admissible(u, equations) = true

@inline function macrocell_point_value_admissible(u,
                                                  equations::CompressibleEulerEquations1D)
    return u[1] > 0 && pressure(u, equations) > 0
end
end # @muladd
