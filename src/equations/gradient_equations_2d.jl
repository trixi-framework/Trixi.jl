
@doc raw"""
    GradientEquations2D

The gradient equations
```math
q^d - \partial_d u = 0
```
in direction `d` in two space dimensions as required for, e.g., the Bassi & Rebay 1 (BR1) or the
local discontinuous Galerkin (LDG) schemes.
"""
struct GradientEquations2D{RealT<:Real, NVARS, Orientation} <: AbstractGradientEquations{2, NVARS}
end

GradientEquations2D(::Type{RealT}, nvars, orientation) where RealT = GradientEquations2D{RealT, nvars, orientation}()


get_name(::GradientEquations2D) = "GradientEquations2D"
varnames(::typeof(cons2cons), equations::GradientEquations2D) = SVector(ntuple(v -> "gradient_"*string(v), nvariables(equations)))
varnames(::typeof(cons2prim), equations::GradientEquations2D) = varnames(cons2cons, equations)

orient(::GradientEquations2D{RealT, NVARS, Orientation}) where {RealT, NVARS, Orientation} = Orientation

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::GradientEquations2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::GradientEquations2D)
  return @SVector SVector(ntuple(v -> 0.0, nvariables(equations)))
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::GradientEquations2D)


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::GradientEquations2D)
  if orientation == orient(equation)
    return u
  else
    return @SVector SVector(ntuple(v -> 0.0, nvariables(equations)))
  end
end
