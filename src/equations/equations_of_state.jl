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
- `energy_internal(V, T, eos)`
- `specific_entropy(V, T, eos)`
- `speed_of_sound(V, T, eos)`

where `eos = equations.equation_of_state`.
`specific_entropy` is required to calculate the mathematical entropy and entropy variables,
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
    s = specific_entropy(V, T, eos)
    p = pressure(V, T, eos)
    e = energy_internal(V, T, eos)
    h = e + p * V
    return h - T * s
end

# compute c_v = de/dT
@inline function heat_capacity_constant_volume(V, T, eos::AbstractEquationOfState)
    return ForwardDiff.derivative(T -> energy_internal(V, T, eos), T)
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
end # @muladd
