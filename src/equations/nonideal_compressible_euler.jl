# The following functions are dimension-independent

function get_name(equations::AbstractNonIdealCompressibleEulerEquations)
    return (equations |> typeof |> nameof |> string) * "{" *
           (equations.equation_of_state |> typeof |> nameof |> string) * "}"
end

@doc raw"""
    entropy_math(u, equations::AbstractNonIdealCompressibleEulerEquations)

Calculate mathematical entropy for a conservative state `cons` as
```math
S = -\rho s
```
where `s` is the specific entropy determined by the equation of state.
"""
@inline function entropy_math(u, equations::AbstractNonIdealCompressibleEulerEquations)
    eos = equations.equation_of_state
    u_thermo = cons2thermo(u, equations)
    V = first(u_thermo)
    T = last(u_thermo)
    rho = u[1]
    S = -rho * entropy_specific(V, T, eos)
    return S
end

"""
    entropy(cons, equations::AbstractNonIdealCompressibleEulerEquations)

Default entropy is the mathematical entropy
[`entropy_math(u, equations::AbstractNonIdealCompressibleEulerEquations)`](@ref).
"""
@inline function entropy(cons, equations::AbstractNonIdealCompressibleEulerEquations)
    return entropy_math(cons, equations)
end

@inline function density(u, equations::AbstractNonIdealCompressibleEulerEquations)
    rho = u[1]
    return rho
end

@inline function pressure(u, equations::AbstractNonIdealCompressibleEulerEquations)
    eos = equations.equation_of_state
    u_thermo = cons2thermo(u, equations)
    V = first(u_thermo)
    T = last(u_thermo)
    p = pressure(V, T, eos)
    return p
end

@inline function density_pressure(u,
                                  equations::AbstractNonIdealCompressibleEulerEquations)
    rho = density(u, equations)
    p = pressure(u, equations)
    return rho * p
end

@inline function energy_internal_specific(u,
                                          equations::AbstractNonIdealCompressibleEulerEquations)
    eos = equations.equation_of_state
    u_thermo = cons2thermo(u, equations)
    V = first(u_thermo)
    T = last(u_thermo)
    e_internal = energy_internal_specific(V, T, eos)
    return e_internal
end
