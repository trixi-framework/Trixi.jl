# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Face reconstructions of the `MacroCellFV` solver

# `reconstruct_cell(reconstruction, stencil)` returns the reconstructed values of one
# variable of one subcell at its two faces along a coordinate line, where
# `stencil(offset)` returns the value of the subcell `offset` cells away along the line:
# `(w_minus, w_plus)`, the values at the face in negative and in positive line
# direction. The reconstructions are applied to each variable separately, so that the
# loops over the subcells of a line vectorize.

"""
    ReconstructionConstant()

Piecewise constant face reconstruction (first-order finite volume scheme).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionConstant end

@inline nhalo(::ReconstructionConstant) = 1

@inline function reconstruct_cell(::ReconstructionConstant, stencil)
    u = stencil(0)
    return u, u
end

"""
    ReconstructionMUSCL(slope_limiter = minmod)

Piecewise linear (MUSCL) face reconstruction on the equidistant subcells.
The slope of each subcell is computed by `slope_limiter(sl, sr)` from the
left and right one-sided differences `sl` and `sr`, applied componentwise. Any of
the limiter functions [`central_slope`](@ref) (unlimited), [`minmod`](@ref),
[`monotonized_central`](@ref), [`superbee`](@ref), [`vanleer`](@ref), or
[`koren_symmetric`](@ref) can be used.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionMUSCL{SlopeLimiter}
    slope_limiter::SlopeLimiter
end

ReconstructionMUSCL() = ReconstructionMUSCL(minmod)

@inline nhalo(::ReconstructionMUSCL) = 2

@inline function reconstruct_cell(reconstruction::ReconstructionMUSCL, stencil)
    @unpack slope_limiter = reconstruction
    u_m = stencil(-1)
    u_0 = stencil(0)
    u_p = stencil(1)

    # Equidistant subcells: the slope is expressed per subcell width, and the
    # faces are half a subcell away from the cell center.
    slope = slope_limiter(u_0 - u_m, u_p - u_0)
    return u_0 - 0.5f0 * slope, u_0 + 0.5f0 * slope
end
end # @muladd
