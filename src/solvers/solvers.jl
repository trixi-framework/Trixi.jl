# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# define types for parabolic solvers
include("solvers_parabolic.jl")

include("dg.jl")
include("dgmulti.jl")
end # @muladd
