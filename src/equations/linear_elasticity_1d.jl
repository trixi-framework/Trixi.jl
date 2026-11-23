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
    -\frac{1}{\rho} \sigma_{11} \\ -\rho c_1^2 v_1
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
c_1^2 = \frac{\lambda + 2 \mu}{\rho}.
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
end

function LinearElasticityEquations1D(; rho::Real, mu::Real, lambda::Real)
    if !(rho > 0)
        throw(ArgumentError("Density rho must be positive."))
    end
    if !(mu > 0)
        throw(ArgumentError("Shear modulus mu (second Lamé parameter) must be positive."))
    end

    c1_squared = (lambda + 2 * mu) / rho
    return LinearElasticityEquations1D(rho, sqrt(c1_squared))
end

# Together with our specialization of `Adapt.adapt_structure`,
# this allows to move semidiscretizations and their components including
# the equations to GPUs and adapt the floating point type, e.g.,
# to `Float32` to improve performance on GPUs.
function Base.similar(equations::LinearElasticityEquations1D,
                      ::Type{NewRealT}) where {NewRealT}
    return LinearElasticityEquations1D(convert(NewRealT, equations.rho),
                                       convert(NewRealT, equations.c1))
end

function varnames(::typeof(cons2cons), ::LinearElasticityEquations1D)
    return ("v1", "sigma11")
end
function varnames(::typeof(cons2prim), ::LinearElasticityEquations1D)
    return ("v1", "sigma11")
end

@doc raw"""
    initial_condition_convergence_test(x, t, equations::LinearElasticityEquations1D)

A smooth solution used for convergence tests, given by
```math
v_1 = \sin(2 \pi c_1 t) \cos(2 \pi x), \quad
\sigma_{11} = \rho c_1 \cos(2 \pi c_1 t) \sin(2 \pi x).
```
It is periodic in space with period one.
"""
function initial_condition_convergence_test(x, t,
                                            equations::LinearElasticityEquations1D)
    @unpack rho, c1 = equations

    v1 = sinpi(2 * c1 * t) * cospi(2 * x[1])
    sigma11 = rho * c1 * cospi(2 * c1 * t) * sinpi(2 * x[1])

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

# Gradient of the `entropy` with respect to the conservative variables
@inline function cons2entropy(u, equations::LinearElasticityEquations1D)
    @unpack rho, c1 = equations
    v1, sigma11 = u
    return SVector(rho * v1, sigma11 / (rho * c1^2))
end

# Useful for e.g. limiting indicator variable selection
@inline function velocity(u, equations::LinearElasticityEquations1D)
    return u[1]
end

@doc raw"""
    energy_kinetic(u, equations::LinearElasticityEquations1D)

Calculate kinetic energy for a conservative state `u` as
```math
E_{kin} = \frac{1}{2} \rho v_1^2
```
"""
@inline function energy_kinetic(u, equations::LinearElasticityEquations1D)
    return 0.5f0 * equations.rho * u[1]^2
end

@doc raw"""
    energy_internal(u, equations::LinearElasticityEquations1D)

Calculate internal energy for a conservative state `u` as
```math
E_{int} = \frac{1}{2} \frac{\sigma_{11}^2}{\rho c_1^2}
```
with the P-wave modulus ``\rho c_1^2 = \lambda + 2 \mu``.
"""
@inline function energy_internal(u, equations::LinearElasticityEquations1D)
    @unpack rho, c1 = equations
    return 0.5f0 * u[2]^2 / (rho * c1^2)
end

"""
    energy_total(u, equations::LinearElasticityEquations1D)

Calculate total energy for a conservative state `u` as the sum of
[`energy_kinetic`](@ref) and [`energy_internal`](@ref).
"""
@inline function energy_total(u, equations::LinearElasticityEquations1D)
    return energy_kinetic(u, equations) + energy_internal(u, equations)
end

"""
    entropy(u, equations::LinearElasticityEquations1D)

Calculate entropy for a conservative state `u`,
here same as [`energy_total(u, equations::AbstractLinearElasticityEquations)`](@ref).
"""
@inline entropy(u, equations::LinearElasticityEquations1D) = energy_total(u, equations)
end # muladd
