
# Wrapper type for solutions from Trixi's own time integrators, partially mimicking
# DiffEqBase.ODESolution
struct TimeIntegratorSolution{tType, uType, P}
  t::tType
  u::uType
  prob::P
end

include_optimized("methods_2N.jl")
include_optimized("methods_3Sstar.jl")
