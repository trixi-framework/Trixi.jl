# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearElasticityEquations1D(;rho::Real, lambda::Real, mu::Real)

Linear elasticity equations in one space dimension. The equations are given by
```math
\partial_t
\begin{pmatrix}
    v_1 \\ \sigma_{11}
\end{pmatrix}
+
\partial_x
\begin{pmatrix}
    -\frac{1}{\rho} \sigma_{11} \\ -\frac{\rho}{c_1^2} v_1
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0
\end{pmatrix}.
```
The variables are the deformation velocity `v_1` and the stress `\sigma_{11}`.

The parameters are the constant density of the material `\rho`
and the Lamé parameters `\lambda` and `\mu`.
From these, one can compute the dilatational wave speed as
```math
c_1^2= \frac{\lambda + 2 * \mu}{\rho}
```
In one dimension the linear elasticity equations reduce to a wave equation.

For reference, see
- Aleksey Sikstel (2020)
  Analysis and numerical methods for coupled hyperbolic conservation laws
  [DOI: 10.18154/RWTH-2020-07821](https://doi.org/10.18154/RWTH-2020-07821)
"""
struct LinearElasticityEquations1D{RealT <: Real} <:
       AbstractLinearElasticityEquations{1, 2}
    rho::RealT # Constant density of the material
    c1::RealT # Dilatational wave speed
    E::RealT # Young's modulus
end

function LinearElasticityEquations1D(; rho::Real, mu::Real, lambda::Real)
    if !(rho > 0)
        throw(ArgumentError("Density rho must be positive."))
    end
    if !(mu > 0)
        throw(ArgumentError("Shear modulus mu (second Lamé parameter) must be positive."))
    end

    c1_squared = (lambda + 2 * mu) / rho

    # Young's modulus
    # See for reference equation (11-11) in
    # https://cns.gatech.edu/~predrag/courses/PHYS-4421-04/lautrup/book/elastic.pdf
    E = mu * (3 * lambda + 2 * mu) / (lambda + mu)
    return LinearElasticityEquations1D(rho, sqrt(c1_squared), E)
end

function varnames(::typeof(cons2cons), ::LinearElasticityEquations1D)
    return ("v1", "sigma11")
end
function varnames(::typeof(cons2prim), ::LinearElasticityEquations1D)
    return ("v1", "sigma11")
end

"""
    initial_condition_convergence_test(x, t, equations::LinearElasticityEquations1D)

A smooth initial condition used for convergence tests.
This requires that the material parameters `rho` is a positive integer
and `c1` is equal to one.
"""
function initial_condition_convergence_test(x, t,
                                            equations::LinearElasticityEquations1D)
    @unpack rho = equations

    v1 = sinpi(2 * t) * cospi(2 * x[1] / rho)
    sigma11 = -cospi(2 * t) * sinpi(2 * x[1] * rho)

    return SVector(v1, sigma11)
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::LinearElasticityEquations1D)
    @unpack rho, c1 = equations
    v1, sigma11 = u
    f1 = -sigma11 / rho
    f2 = -rho * c1^2 * v1

    return SVector(f1, f2)
end

"""
    have_constant_speed(::LinearElasticityEquations1D)

Indicates whether the characteristic speeds are constant, i.e., independent of the solution.
Queried in the timestep computation [`StepsizeCallback`](@ref) and [`linear_structure`](@ref).

# Returns
- `True()`
"""
@inline have_constant_speed(::LinearElasticityEquations1D) = True()

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearElasticityEquations1D)
    return equations.c1
end

# Required for `StepsizeCallback`
@inline function max_abs_speeds(equations::LinearElasticityEquations1D)
    return equations.c1
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::LinearElasticityEquations1D)
    @unpack c1 = equations

    λ_min = -c1
    λ_max = c1

    return λ_min, λ_max
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearElasticityEquations1D) = u
@inline cons2entropy(u, ::LinearElasticityEquations1D) = u

# Useful for e.g. limiting indicator variable selection
@inline function velocity(u, equations::LinearElasticityEquations1D)
    return u[1]
end

@inline function energy_kinetic(u, equations::LinearElasticityEquations1D)
    return 0.5f0 * equations.rho * u[1]^2
end
@inline function energy_internal(u, equations::LinearElasticityEquations1D)
    return 0.5f0 * u[2]^2 / equations.E
end
@inline function energy_total(u, equations::LinearElasticityEquations1D)
    return energy_kinetic(u, equations) + energy_internal(u, equations)
end

@inline entropy(u, equations::LinearElasticityEquations1D) = energy_total(u, equations)
end # muladd
