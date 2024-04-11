# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    MaxwellEquation1D
"""
struct MaxwellEquation1D{RealT <: Real} <:
       AbstractLinearScalarAdvectionEquation{1, 2}
    speed_of_light::SVector{1, RealT}
end

function MaxwellEquation1D(c::Real = 299792458)
    MaxwellEquation1D(SVector(c))
end

function varnames(::typeof(cons2cons), ::MaxwellEquation1D)
  ("E", "B")
end
function varnames(::typeof(cons2prim), ::MaxwellEquation1D)
  ("E", "B")
end

"""
    initial_condition_convergence_test(x, t, equations::MaxwellEquation1D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::MaxwellEquation1D)
    c = equations.speed_of_light[1]
    char_pos = c * t + x[1]

    sin_char_pos = sin(2 * pi * char_pos)

    E = - c * sin_char_pos
    B = sin_char_pos

    return SVector(E, B)
end

# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::MaxwellEquation1D)

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer,
                      equation::MaxwellEquation1D)
    E, B = u
    c = equation.speed_of_light[orientation]
    return SVector(c^2 * B, E)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Int,
                                     equation::MaxwellEquation1D)
    λ_max = equation.speed_of_light[orientation]
end

@inline have_constant_speed(::MaxwellEquation1D) = True()

@inline function max_abs_speeds(equation::MaxwellEquation1D)
    return equation.speed_of_light[1]
end

@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::MaxwellEquation1D)
    min_max_speed_davis(u_ll, u_rr, orientation, equations)
end

@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::MaxwellEquation1D)
    λ_min = -equations.speed_of_light[orientation]
    λ_max = equations.speed_of_light[orientation]

    return λ_min, λ_max
end

# Convert conservative variables to primitive
@inline cons2prim(u, ::MaxwellEquation1D) = u
@inline cons2entropy(u, ::MaxwellEquation1D) = u
end # @muladd