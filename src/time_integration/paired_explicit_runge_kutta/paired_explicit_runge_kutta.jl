# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Basic implementation of the second-order paired explicit Runge-Kutta (PERK) method
include("methods_PERK2.jl")

# Add definitions of functions related to polynomial optimization by Convex and ECOS here
# such that hey can be exported from Trixi.jl and extended in the TrixiConvexECOSExt package
# extension or by the Convex and ECOS-specific code loaded by Requires.jl
function filter_eig_vals end
function undo_normalization! end
function stability_polynomials! end
function bisect_stability_polynomial end
end # @muladd
