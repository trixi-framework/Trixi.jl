# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# Wrapper type for solutions from Trixi.jl's own time integrators, partially mimicking
# SciMLBase.ODESolution
struct TimeIntegratorSolution{tType, uType, P}
  t::tType
  u::uType
  prob::P
end

include("methods_2N.jl")
include("methods_3Sstar.jl")


end # @muladd
