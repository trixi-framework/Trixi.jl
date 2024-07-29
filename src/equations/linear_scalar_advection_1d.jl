# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearScalarAdvectionEquation1D

The linear scalar advection equation
```math
\partial_t u + a \partial_1 u  = 0
```
in one space dimension with constant velocity `a`.
"""
struct LinearScalarAdvectionEquation1D{RealT <: Real} <:
       AbstractLinearScalarAdvectionEquation{1, 1}
    advection_velocity::SVector{1, RealT}
end

function LinearScalarAdvectionEquation1D(a::Real)
    LinearScalarAdvectionEquation1D(SVector(a))
end

varnames(::typeof(cons2cons), ::LinearScalarAdvectionEquation1D) = ("scalar",)
varnames(::typeof(cons2prim), ::LinearScalarAdvectionEquation1D) = ("scalar",)

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::LinearScalarAdvectionEquation1D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::LinearScalarAdvectionEquation1D)
    # Store translated coordinate for easy use of exact solution
    RealT = eltype(x)
    x_trans = x - equation.advection_velocity * t

    return SVector(RealT(2))
end

"""
    initial_condition_convergence_test(x, t, equations::LinearScalarAdvectionEquation1D)

A smooth initial condition used for convergence tests
(in combination with [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref)
in non-periodic domains).
"""
function initial_condition_convergence_test(x, t,
                                            equation::LinearScalarAdvectionEquation1D)
    # Store translated coordinate for easy use of exact solution
    RealT = eltype(x)
    x_trans = x - equation.advection_velocity * t

    c = 1
    A = 0.5f0
    L = 2
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f
    scalar = c + A * sin(omega * sum(x_trans))
    return SVector(scalar)
end

"""
    initial_condition_gauss(x, t, equations::LinearScalarAdvectionEquation1D)

A Gaussian pulse used together with
[`BoundaryConditionDirichlet(initial_condition_gauss)`](@ref).
"""
function initial_condition_gauss(x, t, equation::LinearScalarAdvectionEquation1D)
    # Store translated coordinate for easy use of exact solution
    x_trans = x - equation.advection_velocity * t

    scalar = exp(-(x_trans[1]^2))
    return SVector(scalar)
end

"""
    initial_condition_sin(x, t, equations::LinearScalarAdvectionEquation1D)

A sine wave in the conserved variable.
"""
function initial_condition_sin(x, t, equation::LinearScalarAdvectionEquation1D)
    # Store translated coordinate for easy use of exact solution
    x_trans = x - equation.advection_velocity * t

    scalar = sinpi(2 * x_trans[1])
    return SVector(scalar)
end

"""
    initial_condition_linear_x(x, t, equations::LinearScalarAdvectionEquation1D)

A linear function of `x[1]` used together with
[`boundary_condition_linear_x`](@ref).
"""
function initial_condition_linear_x(x, t, equation::LinearScalarAdvectionEquation1D)
    # Store translated coordinate for easy use of exact solution
    x_trans = x - equation.advection_velocity * t

    return SVector(x_trans[1])
end

"""
    boundary_condition_linear_x(u_inner, orientation, direction, x, t,
                                surface_flux_function,
                                equation::LinearScalarAdvectionEquation1D)

Boundary conditions for
[`initial_condition_linear_x`](@ref).
"""
function boundary_condition_linear_x(u_inner, orientation, direction, x, t,
                                     surface_flux_function,
                                     equation::LinearScalarAdvectionEquation1D)
    u_boundary = initial_condition_linear_x(x, t, equation)

    # Calculate boundary flux
    if direction == 2  # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
    end

    return flux
end

# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::LinearScalarAdvectionEquation1D)

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer,
                      equation::LinearScalarAdvectionEquation1D)
    a = equation.advection_velocity[orientation]
    return a * u
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Int,
                                     equation::LinearScalarAdvectionEquation1D)
    λ_max = abs(equation.advection_velocity[orientation])
end

# Essentially first order upwind, see e.g.
# https://math.stackexchange.com/a/4355076/805029
function flux_godunov(u_ll, u_rr, orientation::Int,
                      equation::LinearScalarAdvectionEquation1D)
    u_L = u_ll[1]
    u_R = u_rr[1]

    v_normal = equation.advection_velocity[orientation]
    if v_normal >= 0
        return SVector(v_normal * u_L)
    else
        return SVector(v_normal * u_R)
    end
end

# See https://metaphor.ethz.ch/x/2019/hs/401-4671-00L/literature/mishra_hyperbolic_pdes.pdf ,
# section 4.2.5 and especially equation (4.33).
function flux_engquist_osher(u_ll, u_rr, orientation::Int,
                             equation::LinearScalarAdvectionEquation1D)
    u_L = u_ll[1]
    u_R = u_rr[1]

    return SVector(0.5f0 * (flux(u_L, orientation, equation) +
                    flux(u_R, orientation, equation) -
                    abs(equation.advection_velocity[orientation]) * (u_R - u_L)))
end

@inline have_constant_speed(::LinearScalarAdvectionEquation1D) = True()

@inline function max_abs_speeds(equation::LinearScalarAdvectionEquation1D)
    return abs.(equation.advection_velocity)
end

"""
    splitting_lax_friedrichs(u, orientation::Integer,
                             equations::LinearScalarAdvectionEquation1D)
    splitting_lax_friedrichs(u, which::Union{Val{:minus}, Val{:plus}}
                             orientation::Integer,
                             equations::LinearScalarAdvectionEquation1D)

Naive local Lax-Friedrichs style flux splitting of the form `f⁺ = 0.5 (f + λ u)`
and `f⁻ = 0.5 (f - λ u)` where `λ` is the absolute value of the advection
velocity.

Returns a tuple of the fluxes "minus" (associated with waves going into the
negative axis direction) and "plus" (associated with waves going into the
positive axis direction). If only one of the fluxes is required, use the
function signature with argument `which` set to `Val{:minus}()` or `Val{:plus}()`.

!!! warning "Experimental implementation (upwind SBP)"
    This is an experimental feature and may change in future releases.
"""
@inline function splitting_lax_friedrichs(u, orientation::Integer,
                                          equations::LinearScalarAdvectionEquation1D)
    fm = splitting_lax_friedrichs(u, Val{:minus}(), orientation, equations)
    fp = splitting_lax_friedrichs(u, Val{:plus}(), orientation, equations)
    return fm, fp
end

@inline function splitting_lax_friedrichs(u, ::Val{:plus}, orientation::Integer,
                                          equations::LinearScalarAdvectionEquation1D)
    RealT = eltype(u)
    a = equations.advection_velocity[1]
    return a > 0 ? flux(u, orientation, equations) : SVector(zero(RealT))
end

@inline function splitting_lax_friedrichs(u, ::Val{:minus}, orientation::Integer,
                                          equations::LinearScalarAdvectionEquation1D)
    RealT = eltype(u)
    a = equations.advection_velocity[1]
    return a < 0 ? flux(u, orientation, equations) : SVector(zero(RealT))
end

# Convert conservative variables to primitive
@inline cons2prim(u, equation::LinearScalarAdvectionEquation1D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::LinearScalarAdvectionEquation1D) = u

# Calculate entropy for a conservative state `cons`
@inline entropy(u::Real, ::LinearScalarAdvectionEquation1D) = 0.5f0 * u^2
@inline entropy(u, equation::LinearScalarAdvectionEquation1D) = entropy(u[1], equation)

# Calculate total energy for a conservative state `cons`
@inline energy_total(u::Real, ::LinearScalarAdvectionEquation1D) = 0.5f0 * u^2
@inline function energy_total(u, equation::LinearScalarAdvectionEquation1D)
    energy_total(u[1], equation)
end
end # @muladd
