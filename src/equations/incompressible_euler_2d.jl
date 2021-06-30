
@doc raw"""
    IncompressibleEulerEquations2D

better comment goes here
"""
struct IncompressibleEulerEquations2D <: AbstractIncompressibleEulerEquations{2, 3} end


varnames(::typeof(cons2cons), ::IncompressibleEulerEquations2D) = ("v1", "v2", "p")
varnames(::typeof(cons2prim), ::IncompressibleEulerEquations2D) = ("v1", "v2", "p")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::IncompressibleEulerEquations2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::IncompressibleEulerEquations2D)
  return SVector(2.0, -3.5, 10.0)
end


"""
    initial_condition_convergence_test(x, t, equations::IncompressibleEulerEquations2D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_pulse(x, t, equation::IncompressibleEulerEquations2D)

  u = exp(-25.0*(x[1]^2 + x[2]^2))
  v = 0.0
  p = 10.0

  return SVector(u,v,p)
end


# """
#     initial_condition_convergence_test(x, t, equations::IncompressibleEulerEquations2D)
#
# A smooth initial condition used for convergence tests.
# """
# function initial_condition_convergence_test(x, t, equation::IncompressibleEulerEquations2D)
#   c = 2.0
#   A = 1.0
#   L = 1
#   f = 1/L
#   omega = 2 * pi * f
#   scalar = c + A * sin(omega * (x[1] - t))
#
#   return SVector(scalar)
# end


# """
#     source_terms_convergence_test(u, x, t, equations::IncompressibleEulerEquations2D)
#
# Source terms used for convergence tests in combination with
# [`initial_condition_convergence_test`](@ref).
# """
# @inline function source_terms_convergence_test(u, x, t, equations::IncompressibleEulerEquations2D)
#   # Same settings as in `initial_condition`
#   c = 2.0
#   A = 1.0
#   L = 1
#   f = 1/L
#   omega = 2 * pi * f
#   du = omega * cos(omega * (x[1] - t)) * (1 + sin(omega * (x[1] - t)))
#
#   return SVector(du)
# end


# Convert conservative variables to primitive
@inline cons2prim(u, equation::IncompressibleEulerEquations2D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::IncompressibleEulerEquations2D) = u
