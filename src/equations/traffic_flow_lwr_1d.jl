# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    TrafficFlowLWREquations1D

The classic Lighthill-Witham Richards (LWR) model for 1D traffic flow.
The car density is denoted by $u \in [0, 1]$ and 
the maximum possible speed (e.g. due to speed limits) is $v_{\text{max}}$.
```math
\partial_t u + v_{\text{max}} \partial_1 [u (1 - u)] = 0
```
For more details see e.g. Section 11.1 of 
- Randall LeVeque (2002)
Finite Volume Methods for Hyperbolic Problems
[DOI: 10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253)
"""
struct TrafficFlowLWREquations1D{RealT <: Real} <: AbstractTrafficFlowLWREquations{1, 1}
    v_max::RealT

    function TrafficFlowLWREquations1D(v_max = 1.0)
        new{typeof(v_max)}(v_max)
    end
end

varnames(::typeof(cons2cons), ::TrafficFlowLWREquations1D) = ("car-density",)
varnames(::typeof(cons2prim), ::TrafficFlowLWREquations1D) = ("car-density",)

"""
    initial_condition_convergence_test(x, t, equations::TrafficFlowLWREquations1D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::TrafficFlowLWREquations1D)
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
    source_terms_convergence_test(u, x, t, equations::TrafficFlowLWREquations1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::TrafficFlowLWREquations1D)
    # Same settings as in `initial_condition`
    RealT = eltype(x)
    c = 2
    A = 1
    L = 1
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f
    du = omega * cos(omega * (x[1] - t)) *
         (-1 - equations.v_max * (2 * sin(omega * (x[1] - t)) + 3))

    return SVector(du)
end

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::TrafficFlowLWREquations1D)
    return SVector(equations.v_max * u[1] * (1 - u[1]))
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::TrafficFlowLWREquations1D)
    λ_max = max(abs(equations.v_max * (1 - 2 * u_ll[1])),
                abs(equations.v_max * (1 - 2 * u_rr[1])))
end

# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::TrafficFlowLWREquations1D)
    jac_L = equations.v_max * (1 - 2 * u_ll[1])
    jac_R = equations.v_max * (1 - 2 * u_rr[1])

    λ_min = min(jac_L, jac_R)
    λ_max = max(jac_L, jac_R)

    return λ_min, λ_max
end

@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::TrafficFlowLWREquations1D)
    min_max_speed_naive(u_ll, u_rr, orientation, equations)
end

@inline function max_abs_speeds(u, equations::TrafficFlowLWREquations1D)
    return (abs(equations.v_max * (1 - 2 * u[1])),)
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::TrafficFlowLWREquations1D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equations::TrafficFlowLWREquations1D) = u

# Calculate entropy for a conservative state `cons`
@inline entropy(u::Real, ::TrafficFlowLWREquations1D) = 0.5f0 * u^2
@inline entropy(u, equations::TrafficFlowLWREquations1D) = entropy(u[1], equations)

# Calculate total energy for a conservative state `cons`
@inline energy_total(u::Real, ::TrafficFlowLWREquations1D) = 0.5f0 * u^2
@inline energy_total(u, equations::TrafficFlowLWREquations1D) = energy_total(u[1],
                                                                             equations)
end # @muladd
