# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearAcousticAdvectionFastEquation1D

The linear acoustic advection equation
```math
\partial_t u + a \partial_1 u + b \partial_1 p = 0
\partial_t p + a \partial_1 p + b \partial_1 u = 0
```
in one space dimension with constant velocities `a` and `b`
"""
struct LinearAcousticAdvectionFastEquation1D{RealT <: Real} <:
       AbstractLinearAcousticAdvectionFastEquation{1, 2}
    a::RealT
    b::RealT
end

function LinearAcousticAdvectionFastEquation1D(a::Real, b::Real)
    LinearAcousticAdvectionFastEquation1D(a, b)
end

function varnames(::typeof(cons2cons), ::LinearAcousticAdvectionFastEquation1D)
    ("v", "p")
end
varnames(::typeof(cons2prim), ::LinearAcousticAdvectionFastEquation1D) = ("v", "p")

# Set initial conditions at physical location `x` for time `t`
function initial_condition_fast_slow(x, t,
                                     equation::LinearAcousticAdvectionFastEquation1D)
    sigma = 0.1
    x0 = 0.75
    x1 = 0.25
    p0 = exp(-((x[1] - x0) / sigma)^2)
    p01 = exp(-((x[1] - x1) / sigma)^2)
    p1 = p01 * cos(7.0 * 2.0 * pi * x[1] / sigma)
    p = p0 + p1
    v = p / equation.b[1]

    return SVector(v, p)
end
# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer,
                      equation::LinearAcousticAdvectionFastEquation1D)
    a = equation.a
    b = equation.b
    v, p = u
    return SVector(b * p, b * v)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Int,
                                     equation::LinearAcousticAdvectionFastEquation1D)
    λ_max = abs(equation.b)
end

function flux_rusanov(u_ll, u_rr, orientation::Int,
                      equation::LinearAcousticAdvectionFastEquation1D)
    v_ll, p_ll = u_ll
    v_rr, p_rr = u_rr
    a = equation.a
    b = equation.b
    f1 = 0.5 * (b * p_ll + b * p_rr) - 0.5 * (b) * (v_rr - v_ll)
    f2 = 0.5 * (b * v_ll + b * v_rr) - 0.5 * (b) * (p_rr - p_ll)

    # f1 = 0.5*(b*p_ll + b*p_rr) 
    # f2 = 0.5*(b*v_ll + b*v_rr) 

    return SVector(f1, f2)
end

@inline have_constant_speed(::LinearAcousticAdvectionFastEquation1D) = True()

@inline function max_abs_speeds(equation::LinearAcousticAdvectionFastEquation1D)
    return abs.(equation.b)
end

# Convert conservative variables to primitive
@inline cons2prim(u, equation::LinearAcousticAdvectionFastEquation1D) = u

@inline cons2cons(u, equation::LinearAcousticAdvectionFastEquation1D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::LinearAcousticAdvectionFastEquation1D) = u
end # @muladd
