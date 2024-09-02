# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearAcousticAdvectionEquation1D

The linear acoustic advection equation
```math
\partial_t u + a \partial_1 u + b \partial_1 p = 0
\partial_t p + a \partial_1 p + b \partial_1 u = 0
```
in one space dimension with constant velocities `a` and `b`
"""
struct LinearAcousticAdvectionEquation1D{RealT <: Real} <:
       AbstractLinearAcousticAdvectionEquation{1, 2}
    a::RealT
    b::RealT
end

function LinearAcousticAdvectionEquation1D(a::Real,b::Real)
    LinearAcousticAdvectionEquation1D(a,b)
end

function varnames(::typeof(cons2cons), ::LinearAcousticAdvectionEquation1D)
    ("v","p")
end
varnames(::typeof(cons2prim), ::LinearAcousticAdvectionEquation1D) = ("v","p")

# Set initial conditions at physical location `x` for time `t`
function initial_condition_fast_slow(x, t, equation::LinearAcousticAdvectionEquation1D)
    sigma = 0.1
    x0 = 0.75
    x1 = 0.25
    p0 = exp(-((x[1]-x0)/sigma)^2)
    p01 = exp(-((x[1]-x1)/sigma)^2)
    p1 = p01*cos(7.0*2.0*pi*x[1]/sigma)
    p = p0 + p1
    v = p/equation.b[1]

    return SVector(v, p)
end

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer,
                      equation::LinearAcousticAdvectionEquation1D)
    a = equation.a
    b = equation.b
    v, p = u
    return SVector(a * v + b * p, a * p + b * v)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Int,
                                     equation::LinearAcousticAdvectionEquation1D)
    λ_max = abs(equation.b)
end

function flux_rusanov(u_ll, u_rr, orientation::Int,
                      equation::LinearAcousticAdvectionEquation1D)
v_ll, p_ll = u_ll
v_rr, p_rr = u_rr
a = equation.a
b = equation.b
f1 = 0.5*(a*v_ll + b*p_ll + a*v_rr + b*p_rr) - 0.5*(b+a)*(v_rr - v_ll)
f2 = 0.5*(a*p_ll + b*v_ll + a*p_rr + b*v_rr) - 0.5*(b+a)*(p_rr - p_ll)

return SVector(f1,f2)

end

@inline have_constant_speed(::LinearAcousticAdvectionEquation1D) = True()

@inline function max_abs_speeds(equation::LinearAcousticAdvectionEquation1D)
    return abs.(equation.b)
end

# Convert conservative variables to primitive
@inline cons2prim(u, equation::LinearAcousticAdvectionEquation1D) = u

@inline cons2cons(u, equation::LinearAcousticAdvectionEquation1D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::LinearAcousticAdvectionEquation1D) = u

end # @muladd
