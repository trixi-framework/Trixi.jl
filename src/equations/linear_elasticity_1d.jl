# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearElasticityEquations1D(rho, lambda, mu)

Linear elasticity equations in one space dimension. The equations are given by
```math
\partial_t
\begin{pmatrix}
    v \\ \sigma
\end{pmatrix}
+
\partial_x
\begin{pmatrix}
    -\frac{1}{\rho} \sigma \\ -\frac{\rho}{c_1^2} v
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0
\end{pmatrix}
```
The variables are the deformation velocity `v` and the stress `\sigma`.

The parameters are the constant density of the material `\rho`
and the Lamé parameters `\lambda` and `\mu`.
"""
struct LinearElasticityEquations1D{RealT <: Real} <:
       AbstractLinearElasticityEquations{1, 2}
    rho::RealT # Constant density of the material
    c_1_squared::RealT # Dilational wave speed
end

function LinearElasticityEquations1D(rho::Real, mu::Real, lambda::Real)
    @assert rho > 0 "Density rho must be positive."
    @assert mu > 0 "Shear modulus mu (second Lamé parameter) must be positive."

    c_1_squared = (lambda + 2 * mu) / rho
    return LinearElasticityEquations1D(rho, c_1_squared)
end

# Constructor with keywords
function LinearElasticityEquations1D(; rho::Real, mu::Real, lambda::Real)
    c_1_squared = (lambda + 2 * mu) / rho
    return LinearElasticityEquations1D(rho, c_1_squared)
end

# Constructor with keywords with dilatational wave speed supplied
function LinearElasticityEquations1D(; rho::Real, c_1_squared::Real)
    return LinearElasticityEquations1D(rho, c_1_squared)
end

function varnames(::typeof(cons2cons), ::LinearElasticityEquations1D)
    ("v", "sigma")
end
function varnames(::typeof(cons2prim), ::LinearElasticityEquations1D)
    ("v", "sigma")
end

"""
    initial_condition_convergence_test(x, t, equations::LinearElasticityEquations1D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::LinearElasticityEquations1D)
    rho_prime = -cospi(2 * t) * sinpi(2 * x[1])
    v1_prime = sinpi(2 * t) * cospi(2 * x[1])
    p_prime = rho_prime

    return SVector(rho_prime, v1_prime, p_prime)
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::LinearElasticityEquations1D)
    @unpack rho, c_1_squared = equations
    v, sigma = u
    f1 = -sigma / rho
    f2 = -rho * c_1_squared * v

    return SVector(f1, f2)
end

@inline have_constant_speed(::LinearElasticityEquations1D) = True()

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearElasticityEquations1D)
    @unpack c_1_squared = equations
    return sqrt(c_1_squared)
end

# Calculate estimate for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearElasticityEquations1D)
    min_max_speed_davis(u_ll, u_rr, orientation, equations)
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::LinearElasticityEquations1D)
    @unpack c_1_squared = equations

    λ_min = -sqrt(c_1_squared)
    λ_max = sqrt(c_1_squared)

    return λ_min, λ_max
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearElasticityEquations1D) = u
@inline cons2entropy(u, ::LinearElasticityEquations1D) = u
end # muladd
