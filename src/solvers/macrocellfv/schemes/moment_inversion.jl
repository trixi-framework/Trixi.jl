# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Moment inversions of the `MacroCellFV` solver (cell averages -> point values at the
# subcell centers)

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

# `True()` if the moment inversion provides point values at the subcell centers (and the
# pads hold point values and averages)
@inline has_point_values(::MomentInversionIdentity) = False()
end # @muladd
