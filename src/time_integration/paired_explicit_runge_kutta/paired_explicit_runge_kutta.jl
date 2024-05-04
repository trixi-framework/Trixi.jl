# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Basic implementation of the second-order paired explicit Runge-Kutta (PERK) method
include("methods_PERK2.jl")
# Define all of the functions necessary for polynomial optimizations
include("polynomial_optimizer.jl")
end # @muladd
