# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See TODO: link-to-my-blog-post
@muladd begin


# Wrapper type for solutions from Trixi's own time integrators, partially mimicking
# DiffEqBase.ODESolution
struct TimeIntegratorSolution{tType, uType, P}
  t::tType
  u::uType
  prob::P
end

include("methods_2N.jl")
include("methods_3Sstar.jl")


end # @muladd
