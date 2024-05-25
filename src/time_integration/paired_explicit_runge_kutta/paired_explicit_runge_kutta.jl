# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Basic implementation of the second-order paired explicit Runge-Kutta (PERK) method
include("methods_PERK2.jl")
include("methods_PERK3.jl")
# Define all of the functions necessary for polynomial optimizations
include("polynomial_optimizer.jl")

# Add definitions of functions related to polynomial optimization by NLsolve here
# such that hey can be exported from Trixi.jl and extended in the TrixiConvexECOSExt package
# extension or by the NLsolve-specific code loaded by Requires.jl
function solve_a_unknown end
end # @muladd
