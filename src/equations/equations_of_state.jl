@doc raw"""
    The interface for an `AbstractEquationOfState` requires specifying
the following four functions: 
- `pressure(V, T, eos)`
- `internal_energy(V, T, eos)`
- `specific_entropy(V, T, eos)`
- `speed_of_sound(V, T, eos)`
where `eos = equations.equation_of_state`. `specific_entropy` is required to calculate the 
mathematical entropy and entropy variables, and `speed_of_sound` is required to calculate 
wavespeed estimates for e.g., local Lax-Friedrichs fluxes. 
    
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
    e = internal_energy(V, T, eos)
    h = e + p * V
    return h - T * s
end

# calculates dpdV_T, dpdT_V
@inline function calc_pressure_derivatives(V, T, eos::AbstractEquationOfState)
    dpdV_T = ForwardDiff.derivative(V -> pressure(V, T, eos), V)
    dpdT_V = ForwardDiff.derivative(T -> pressure(V, T, eos), T)
    return dpdV_T, dpdT_V
end

@inline function heat_capacity_constant_volume(V, T, eos::AbstractEquationOfState)
    return ForwardDiff.derivative(T -> internal_energy(V, T, eos), T)
end

# calculate the temperature as a function of specific volume `V` and internal energy `e`
# by using Newton's method to determine `T` such that `internal_energy(V, T, eos) = e`.
function temperature(V, e, eos::AbstractEquationOfState; initial_T = 1.0)
    tol = 10 * eps()
    T = initial_T
    de = internal_energy(V, T, eos) - e
    iter = 1
    while abs(de) / abs(e) > tol && iter < 100
        de = internal_energy(V, T, eos) - e

        # c_v = de_dT_V > 0, which should guarantee convergence of this iteration
        de_dT_V = heat_capacity_constant_volume(V, T, eos)

        T = T - de / de_dT_V
        iter += 1
    end
    if iter == 100
        println("Warning: nonlinear solve in `temperature(V, T, eos)` did not converge within 100 iterations. " *
                "Final states: iter = $iter, V, e = $V, $e with de = $de")
    end
    return T
end
