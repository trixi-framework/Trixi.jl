# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    MaxwellEquations1D

The Maxwell equations of electro dynamics
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
E \\ B
\end{pmatrix}
+ 
\frac{\partial}{\partial x}
\begin{pmatrix}
c^2 B \\ E
\end{pmatrix}
=
\begin{pmatrix}
0 \\ 0 
\end{pmatrix}
```
in one dimension with speed of light `c = 299792458 m/s` (in vacuum).
In one dimension the Maxwell equations reduce to a wave equation.
The orthogonal magnetic (e.g.`B_y`) and electric field (`E_z`) propagate as waves 
through the domain in `x`-direction.
"""
struct MaxwellEquations1D{RealT <: Real} <:
       AbstractLinearScalarAdvectionEquation{1, 2}
    speed_of_light::SVector{1, RealT} # c
end

function MaxwellEquations1D(c::Real = 299792458)
    MaxwellEquations1D(SVector(c))
end

function varnames(::typeof(cons2cons), ::MaxwellEquations1D)
    ("E", "B")
end
function varnames(::typeof(cons2prim), ::MaxwellEquations1D)
    ("E", "B")
end

"""
    initial_condition_convergence_test(x, t, equations::MaxwellEquations1D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::MaxwellEquations1D)
    c = equations.speed_of_light[1]
    char_pos = c * t + x[1]

    sin_char_pos = sin(2 * pi * char_pos)

    E = -c * sin_char_pos
    B = sin_char_pos

    return SVector(E, B)
end

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer,
                      equations::MaxwellEquations1D)
    E, B = u
    c = equations.speed_of_light[orientation]
    return SVector(c^2 * B, E)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Int,
                                     equations::MaxwellEquations1D)
    λ_max = equations.speed_of_light[orientation]
end

@inline have_constant_speed(::MaxwellEquations1D) = True()

@inline function max_abs_speeds(equations::MaxwellEquations1D)
    return equations.speed_of_light[1]
end

@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::MaxwellEquations1D)
    min_max_speed_davis(u_ll, u_rr, orientation, equations)
end

@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::MaxwellEquations1D)
    λ_min = -equations.speed_of_light[orientation]
    λ_max = equations.speed_of_light[orientation]

    return λ_min, λ_max
end

# Convert conservative variables to primitive
@inline cons2prim(u, ::MaxwellEquations1D) = u
@inline cons2entropy(u, ::MaxwellEquations1D) = u
end # @muladd
