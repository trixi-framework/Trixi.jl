# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Moment inversions of the `MacroCellFV` solver (cell averages -> point values at the
# subcell centers) without nonlinear weights; see `weno/moment_inversion_weno.jl` for
# the WENO moment inversion.

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

"""
    MomentInversionCentral()

Fourth-order moment inversion: the point value at the center of each subcell is
`ū - Σ_d Δ_d² ū / 24` with the second differences `Δ_d²` of the subcell averages `ū`
(conservative variables) along each coordinate direction `d`, exact for cubic
polynomials. Where the neighbors along a direction are missing (at the ends of the
padded work array), the one-sided second difference `2 ū_0 - 5 ū_1 + 4 ū_2 - ū_3` is
used, which keeps the order.

The point values are converted to the reconstruction variables (`cons2recon`)
pointwise, and the WENO reconstructions interpolate the point values to the face
midpoints with the weights for point values (see [`MomentInversionWENO`](@ref)).
This removes the second-order error of the reconstruction of nonlinear functions of
the averages (e.g., primitive variables).

If the central correction is applied across discontinuities, it can create
oscillations; at material interfaces, the oscillations of the conservative variables
perturb the pressure equilibrium. Use [`MomentInversionWENO`](@ref) for flows with
discontinuities. Point values outside of the admissible set (see
[`ReconstructionPositivity`](@ref)) are replaced by the averages.

Supported with the WENO reconstructions (also with the positivity limiter).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct MomentInversionCentral end

# Correction `u_center - ū` of the point value at the center of the subcell `0` of
# `line` (conservative subcell averages `line(offset)` for `lo <= offset <= hi`, with
# `lo <= 0 <= hi`) along one coordinate direction
@inline function center_correction(::MomentInversionCentral, line, lo, hi)
    u_0 = line(0)
    if lo <= -1 && hi >= 1
        # (mirror-symmetric operations, see `weno5_smoothness`)
        return -((line(-1) + line(1)) - 2 * u_0) / 24
    elseif lo == 0
        # One-sided second difference, exact for cubic polynomials
        return -(2 * u_0 - 5 * line(1) + 4 * line(2) - line(3)) / 24
    else
        return -(2 * u_0 - 5 * line(-1) + 4 * line(-2) - line(-3)) / 24
    end
end

# Point value at the subcell center from the conservative average `u_average` and the
# sum of the corrections of all directions; inadmissible point values (see
# `positivity_quantities`) are replaced by the average
@inline function center_point_value(u_average, correction, equations)
    u_center = u_average + correction
    bounds = convert(eltype(u_average), 1.0e-12) *
             positivity_quantities(u_average, equations)
    return positivity_admissible(u_center, bounds, equations) ? u_center : u_average
end
end # @muladd
