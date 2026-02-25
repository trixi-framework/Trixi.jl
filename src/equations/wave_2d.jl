# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    WaveEquations2D(c)

The wave equations in two space dimension written as a first order system. The equations are given by
```math
\begin{alignat*}{2}
    \partial_t p &+ c\nabla \cdot{} v &&= 0 \\
    \partial_t v &+ c\nabla p &&= 0
\end{alignat*}
The unknowns are the wave amplitude ``p`` and the wave "flux" ``(v_x, v_y)^T``. 
The parameter ``c`` is the wave speed.
```
"""
struct WaveEquations2D{RealT <: Real} <: AbstractFirstOrderWaveEquations{2, 3}
    c::RealT
end

function varnames(::Union{typeof(cons2cons), typeof(cons2prim)}, ::WaveEquations2D)
    return ("p", "vx", "vy")
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::WaveEquations2D)
    @unpack c = equations
    u, vx, vy = u
    if orientation == 1
        return SVector(c * vx, c * u, zero(vy))
    else
        return SVector(c * vy, zero(vx), c * u)
    end
end

"""
    have_constant_speed(::WaveEquations2D)

Indicates whether the characteristic speeds are constant, i.e., independent of the solution.
Queried in the timestep computation [`StepsizeCallback`](@ref) and [`linear_structure`](@ref).

# Returns
- `True()`
"""
@inline have_constant_speed(::WaveEquations2D) = True()

@inline function max_abs_speeds(equations::WaveEquations2D)
    return equations.c, equations.c
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::WaveEquations2D)
    return equations.c
end

# Convert conservative variables to primitive
@inline cons2prim(u, ::WaveEquations2D) = u
@inline cons2entropy(u, ::WaveEquations2D) = u
end # muladd
