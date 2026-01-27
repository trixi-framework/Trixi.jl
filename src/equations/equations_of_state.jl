# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    AbstractEquationOfState

The interface for an `AbstractEquationOfState` requires specifying
the following four functions: 
- `pressure(V, T, eos)`
- `energy_internal(V, T, eos)`, the specific internal energy
- `entropy_specific(V, T, eos)`, the specific entropy
- `speed_of_sound(V, T, eos)`

where `eos = equations.equation_of_state`.
`entropy_specific` is required to calculate the mathematical entropy and entropy variables,
and `speed_of_sound` is required to calculate wavespeed estimates for e.g., [`FluxLaxFriedrichs`](@ref).
    
Additional functions can also be specialized to particular equations of state to improve 
efficiency. 
"""
abstract type AbstractEquationOfState end

include("equation_of_state_ideal_gas.jl")
include("equation_of_state_vdw.jl")

#######################################################
#
# Some general fallback routines are provided below
# 
#######################################################

function gibbs_free_energy(V, T, eos)
    s = entropy_specific(V, T, eos)
    p = pressure(V, T, eos)
    e = energy_internal(V, T, eos)
    h = e + p * V
    return h - T * s
end

# compute c_v = de/dT
@inline function heat_capacity_constant_volume(V, T, eos::AbstractEquationOfState)
    return ForwardDiff.derivative(T -> energy_internal(V, T, eos), T)
end

function calc_pressure_derivatives(V, T, eos)
    dpdV_T = ForwardDiff.derivative(V -> pressure(V, T, eos), V)
    dpdT_V = ForwardDiff.derivative(T -> pressure(V, T, eos), T)
    return dpdT_V, dpdV_T
end

# relative tolerance for the Newton solver for temperature
eos_newton_tol(eos::AbstractEquationOfState) = 10 * eps()

"""
    temperature(V, e, eos::AbstractEquationOfState; initial_T = 1.0, tol = eos_newton_tol(eos),
                maxiter = 100)

Calculates the temperature as a function of specific volume `V` and internal energy `e`
by using Newton's method to determine `T` such that `energy_internal(V, T, eos) = e`.
Note that the tolerance may need to be adjusted based on the specific equation of state. 
"""
function temperature(V, e, eos::AbstractEquationOfState; initial_T = 1.0,
                     tol = eos_newton_tol(eos), maxiter = 100)
    T = initial_T
    de = energy_internal(V, T, eos) - e
    iter = 1
    while abs(de) > tol * abs(e) && iter < maxiter
        de = energy_internal(V, T, eos) - e

        # for thermodynamically admissible states, c_v = de_dT_V > 0, which should 
        # guarantee convergence of this iteration.
        de_dT_V = heat_capacity_constant_volume(V, T, eos)

        T = T - de / de_dT_V
        iter += 1
    end
    if iter == maxiter
        println("Warning: nonlinear solve in `temperature(V, T, eos)` did not converge within $maxiter iterations. " *
                "Final states: iter = $iter, V, e = $V, $e with de = $de")
    end
    return T
end

# helper function used in [`flux_terashima_etal`](@ref) and [`flux_terashima_etal_central`](@ref)
@inline function drho_e_drho_at_const_p(V, T, eos::AbstractEquationOfState)
    rho = inv(V)
    e = energy_internal(V, T, eos)

    dpdT_V, dpdV_T = calc_pressure_derivatives(V, T, eos)
    dpdrho_T = dpdV_T * (-V / rho) # V = inv(rho), so dVdrho = -1/rho^2 = -V^2. 
    de_dV_T = T * dpdT_V - pressure(V, T, eos)
    drho_e_drho_T = e + rho * de_dV_T * (-V / rho) # d(rho_e)/drho_|T = e + rho * de_dV|T * dVdrho

    c_v = heat_capacity_constant_volume(V, T, eos)
    return ((-rho * c_v) / (dpdT_V) * dpdrho_T + drho_e_drho_T)
end
end # @muladd
