# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Face reconstructions of the `MacroCellFV` solver: the common interface, the piecewise
# constant and MUSCL reconstructions, and the wrapper for reconstructions from point
# values. The WENO reconstructions are in `weno/`, the positivity limiter in
# `positivity_constraint.jl`.

# Reconstructed states of one subcell at its two faces along a coordinate line,
# where `stencil(offset)` returns the state of the subcell `offset` cells away
# along the line. Returns `(w_minus, w_plus)`, the states at the face in negative
# and in positive line direction. Each subcell is reconstructed once per direction
# and RHS evaluation; the face states of all subcells are then combined by the
# numerical fluxes.
# Reconstruction with the equations and the conversion of the reconstruction variables
# to the conservative variables (needed by the positivity limiter only)
@inline function reconstruct_cell(reconstruction, stencil, equations, recon2cons)
    return reconstruct_cell(reconstruction, stencil)
end

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
    slope = slope_limiter.(u_0 - u_m, u_p - u_0)
    return u_0 - 0.5f0 * slope, u_0 + 0.5f0 * slope
end

# WENO reconstruction from the point values at the subcell centers (moment inversions
# `MomentInversionPoint`): the pad holds the point values (first half of the
# variables) and the averages (second half), both in reconstruction variables. The
# face values are the WENO interpolations of the point values to the face midpoints
# (as in the weighted compact nonlinear schemes of Deng and Zhang, 2000), with the
# candidates of the stencils `(i-2, i-1, i)`, `(i-1, i, i+1)`, `(i, i+1, i+2)` and the
# linear weights `(1, 10, 5) / 16`, exact for quartic polynomials.
struct ReconstructionPointValues{Reconstruction}
    reconstruction::Reconstruction
end

function Base.show(io::IO, reconstruction::ReconstructionPointValues)
    print(io, reconstruction.reconstruction, " (point values)")
end

@inline nhalo(reconstruction::ReconstructionPointValues) = nhalo(reconstruction.reconstruction)

@inline point_part(x::SVector{M}) where {M} = SVector(ntuple(@inline(i->x[i]),
                                                             Val(M ÷ 2)))
@inline average_part(x::SVector{M}) where {M} = SVector(ntuple(@inline(i->x[M ÷ 2 + i]),
                                                               Val(M ÷ 2)))

# The subcell value used as reference by the positivity limiter: the average
@inline reconstruction_center(reconstruction, stencil) = stencil(0)
@inline function reconstruction_center(::ReconstructionPointValues, stencil)
    return average_part(stencil(0))
end
end # @muladd
