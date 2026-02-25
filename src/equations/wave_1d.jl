# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    WaveEquations1D(c)

The wave equation
The wave equation
```math
u_{tt} - c^2 u_{xx} = 0
```
in one space dimension as a first order system.
The equations are given by
```math
\begin{alignat*}{2}
    \partial_t p &+ c\nabla \cdot v &&= 0 \\
    \partial_t v &+ c\nabla p &&= 0
\end{alignat*}
The unknowns are the wave amplitude ``p`` and the wave flux ``v``. 
The parameter ``c`` is the wave speed.
```
"""
struct WaveEquations1D{RealT <: Real} <: AbstractFirstOrderWaveEquations{1, 2}
    c::RealT
end

function varnames(::Union{typeof(cons2cons), typeof(cons2prim)}, ::WaveEquations1D)
    return ("p", "v")
end

function initial_condition_gauss(x, t, ::WaveEquations1D)
    p = exp(-25 * x[1]^2)
    v = exp(-25 * x[1]^2)
    return SVector(p, v)
end

"""
    boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                equations::WaveEquations1D)

Boundary conditions for a solid wall, corresponding to zero amplitde at the wall.
In some sense this is a mixed boundary condition, with Dirichlet zero for the amplitude and
Neumann zero for the flux.
"""
function boundary_condition_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function,
                                 equations::WaveEquations1D)
    p, v = u_inner
    u_boundary = SVector(zero(p), v)

    # Calculate boundary flux
    if direction == 2  # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
    return flux
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::WaveEquations1D)
    @unpack c = equations
    p, v = u
    return SVector(c * v, c * p)
end

"""
    have_constant_speed(::WaveEquations1D)

Indicates whether the characteristic speeds are constant, i.e., independent of the solution.
Queried in the timestep computation [`StepsizeCallback`](@ref) and [`linear_structure`](@ref).

# Returns
- `True()`
"""
@inline have_constant_speed(::WaveEquations1D) = True()

@inline function max_abs_speeds(equations::WaveEquations1D)
    return equations.c
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::WaveEquations1D)
    return equations.c
end

# Convert conservative variables to primitive
@inline cons2prim(u, ::WaveEquations1D) = u
@inline cons2entropy(u, ::WaveEquations1D) = u
end # muladd
