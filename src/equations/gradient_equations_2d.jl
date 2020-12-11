
@doc raw"""
    GradientEquations2D

The gradient equations
```math
q^d - \partial_d u = 0
```
in direction `d` in two space dimensions as required for, e.g., the Bassi & Rebay 1 (BR1) or the
local discontinuous Galerkin (LDG) schemes.
"""
struct GradientEquations2D{RealT<:Real, NVARS} <: AbstractGradientEquations{2, NVARS}
  orientation::Int
end

GradientEquations2D(::Type{RealT}, nvars, orientation) where RealT = GradientEquations2D{RealT, nvars}(orientation)
GradientEquations2D(nvars, orientation) = GradientEquations2D(Float64, nvars, orientation)


get_name(::GradientEquations2D) = "GradientEquations2D"
varnames(::typeof(cons2cons), equations::GradientEquations2D) = SVector(ntuple(v -> "gradient_"*string(v), nvariables(equations)))
varnames(::typeof(cons2prim), equations::GradientEquations2D) = varnames(cons2cons, equations)

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::GradientEquations2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::GradientEquations2D)
  return SVector(ntuple(v -> zero(eltype(x)), nvariables(equations)))
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::GradientEquations2D)


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equations::GradientEquations2D)
  if orientation == equations.orientation
    return -u
  else
    return SVector(ntuple(v -> zero(eltype(u)), nvariables(equations)))
  end
end
