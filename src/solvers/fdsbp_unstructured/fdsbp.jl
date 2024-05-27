# !!! warning "Experimental implementation (curvilinear FDSBP)"
#     This is an experimental feature and may change in future releases.

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# dimension specific curvilinear implementations and data structures
include("containers_2d.jl")
include("fdsbp_2d.jl")
end # @muladd
