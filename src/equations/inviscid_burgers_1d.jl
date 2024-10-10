# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    InviscidBurgersEquation1D

The inviscid Burgers' equation
```math
\partial_t u + \frac{1}{2} \partial_1 u^2 = 0
```
in one space dimension.
"""
struct InviscidBurgersEquation1D <: AbstractInviscidBurgersEquation{1, 1} end

varnames(::typeof(cons2cons), ::InviscidBurgersEquation1D) = ("scalar",)
varnames(::typeof(cons2prim), ::InviscidBurgersEquation1D) = ("scalar",)

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::InviscidBurgersEquation1D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::InviscidBurgersEquation1D)
    RealT = eltype(x)
    return SVector(RealT(2))
end

"""
    initial_condition_convergence_test(x, t, equations::InviscidBurgersEquation1D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equation::InviscidBurgersEquation1D)
    RealT = eltype(x)
    c = 2
    A = 1
    L = 1
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f
    scalar = c + A * sin(omega * (x[1] - t))

    return SVector(scalar)
end

"""
    source_terms_convergence_test(u, x, t, equations::InviscidBurgersEquation1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::InviscidBurgersEquation1D)
    # Same settings as in `initial_condition`
    RealT = eltype(x)
    c = 2
    A = 1
    L = 1
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f
    du = omega * A * cos(omega * (x[1] - t)) * (c - 1 + A * sin(omega * (x[1] - t)))

    return SVector(du)
end

# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::InviscidBurgersEquation1D)

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equation::InviscidBurgersEquation1D)
    return SVector(0.5f0 * u[1]^2)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::InviscidBurgersEquation1D)
    u_L = u_ll[1]
    u_R = u_rr[1]

    λ_max = max(abs(u_L), abs(u_R))
end

# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::InviscidBurgersEquation1D)
    u_L = u_ll[1]
    u_R = u_rr[1]

    λ_min = min(u_L, u_R)
    λ_max = max(u_L, u_R)

    return λ_min, λ_max
end

@inline function max_abs_speeds(u, equation::InviscidBurgersEquation1D)
    return (abs(u[1]),)
end

# (Symmetric) Entropy Conserving flux
function flux_ec(u_ll, u_rr, orientation, equation::InviscidBurgersEquation1D)
    u_L = u_ll[1]
    u_R = u_rr[1]

    return SVector((u_L^2 + u_L * u_R + u_R^2) / 6)
end

# See https://metaphor.ethz.ch/x/2019/hs/401-4671-00L/literature/mishra_hyperbolic_pdes.pdf ,
# section 4.1.5 and especially equation (4.16).
function flux_godunov(u_ll, u_rr, orientation, equation::InviscidBurgersEquation1D)
    u_L = u_ll[1]
    u_R = u_rr[1]

    return SVector(0.5f0 * max(max(u_L, 0)^2, min(u_R, 0)^2))
end

# See https://metaphor.ethz.ch/x/2019/hs/401-4671-00L/literature/mishra_hyperbolic_pdes.pdf ,
# section 4.2.5 and especially equation (4.34).
function flux_engquist_osher(u_ll, u_rr, orientation,
                             equation::InviscidBurgersEquation1D)
    u_L = u_ll[1]
    u_R = u_rr[1]

    return SVector(0.5f0 * (max(u_L, 0)^2 + min(u_R, 0)^2))
end

"""
    splitting_lax_friedrichs(u, orientation::Integer,
                             equations::InviscidBurgersEquation1D)
    splitting_lax_friedrichs(u, which::Union{Val{:minus}, Val{:plus}}
                             orientation::Integer,
                             equations::InviscidBurgersEquation1D)

Naive local Lax-Friedrichs style flux splitting of the form `f⁺ = 0.5 (f + λ u)`
and `f⁻ = 0.5 (f - λ u)` where `λ = abs(u)`.

Returns a tuple of the fluxes "minus" (associated with waves going into the
negative axis direction) and "plus" (associated with waves going into the
positive axis direction). If only one of the fluxes is required, use the
function signature with argument `which` set to `Val{:minus}()` or `Val{:plus}()`.

!!! warning "Experimental implementation (upwind SBP)"
    This is an experimental feature and may change in future releases.
"""
@inline function splitting_lax_friedrichs(u, orientation::Integer,
                                          equations::InviscidBurgersEquation1D)
    fm = splitting_lax_friedrichs(u, Val{:minus}(), orientation, equations)
    fp = splitting_lax_friedrichs(u, Val{:plus}(), orientation, equations)
    return fm, fp
end

@inline function splitting_lax_friedrichs(u, ::Val{:plus}, orientation::Integer,
                                          equations::InviscidBurgersEquation1D)
    f = 0.5f0 * u[1]^2
    lambda = abs(u[1])
    return SVector(0.5f0 * (f + lambda * u[1]))
end

@inline function splitting_lax_friedrichs(u, ::Val{:minus}, orientation::Integer,
                                          equations::InviscidBurgersEquation1D)
    f = 0.5f0 * u[1]^2
    lambda = abs(u[1])
    return SVector(0.5f0 * (f - lambda * u[1]))
end

# Convert conservative variables to primitive
@inline cons2prim(u, equation::InviscidBurgersEquation1D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::InviscidBurgersEquation1D) = u
@inline entropy2cons(u, equation::InviscidBurgersEquation1D) = u

# Calculate entropy for a conservative state `cons`
@inline entropy(u::Real, ::InviscidBurgersEquation1D) = 0.5f0 * u^2
@inline entropy(u, equation::InviscidBurgersEquation1D) = entropy(u[1], equation)

# Calculate total energy for a conservative state `cons`
@inline energy_total(u::Real, ::InviscidBurgersEquation1D) = 0.5f0 * u^2
@inline function energy_total(u, equation::InviscidBurgersEquation1D)
    energy_total(u[1], equation)
end
end # @muladd
