# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    AcousticPerturbationEquations2D(v_mean_global, c_mean_global, rho_mean_global)

Acoustic perturbation equations (APE) in two space dimensions. The equations are given by
```math
\begin{aligned}
  \frac{\partial\mathbf{v'}}{\partial t} + \nabla (\bar{\mathbf{v}}\cdot\mathbf{v'})
    + \nabla\left( \frac{\bar{c}^2 \tilde{p}'}{\bar{\rho}} \right) &= 0 \\
  \frac{\partial \tilde{p}'}{\partial t} +
    \nabla\cdot (\bar{\rho} \mathbf{v'} + \bar{\mathbf{v}} \tilde{p}') &= 0.
\end{aligned}
```
The bar ``\bar{(\cdot)}`` indicates time-averaged quantities. The unknowns of the APE are the
perturbed velocities ``\mathbf{v'} = (v_1', v_2')^T`` and the scaled perturbed pressure
``\tilde{p}' = \frac{p'}{\bar{c}^2}``, where ``p'`` denotes the perturbed pressure and the
perturbed variables are defined by ``\phi' = \phi - \bar{\phi}``.

In addition to the unknowns, Trixi.jl currently stores the mean values in the state vector,
i.e. the state vector used internally is given by
```math
\mathbf{u} =
  \begin{pmatrix}
    v_1' \\ v_2' \\ \tilde{p}' \\ \bar{v}_1 \\ \bar{v}_2 \\ \bar{c} \\ \bar{\rho}
  \end{pmatrix}.
```
This affects the implementation and use of these equations in various ways:
* The flux values corresponding to the mean values must be zero.
* The mean values have to be considered when defining initial conditions, boundary conditions or
  source terms.
* [`AnalysisCallback`](@ref) analyzes these variables too.
* Trixi.jl's visualization tools will visualize the mean values by default.

The constructor accepts a 2-tuple `v_mean_global` and scalars `c_mean_global` and `rho_mean_global`
which can be used to make the definition of initial conditions for problems with constant mean flow
more flexible. These values are ignored if the mean values are defined internally in an initial
condition.

The equations are based on the APE-4 system introduced in the following paper:
- Roland Ewert and Wolfgang Schröder (2003)
  Acoustic perturbation equations based on flow decomposition via source filtering
  [DOI: 10.1016/S0021-9991(03)00168-2](https://doi.org/10.1016/S0021-9991(03)00168-2)
"""
struct AcousticPerturbationEquations2D{RealT <: Real} <:
       AbstractAcousticPerturbationEquations{2, 7}
    v_mean_global::SVector{2, RealT}
    c_mean_global::RealT
    rho_mean_global::RealT
end

function AcousticPerturbationEquations2D(v_mean_global::NTuple{2, <:Real},
                                         c_mean_global::Real,
                                         rho_mean_global::Real)
    return AcousticPerturbationEquations2D(SVector(v_mean_global), c_mean_global,
                                           rho_mean_global)
end

function AcousticPerturbationEquations2D(; v_mean_global::NTuple{2, <:Real},
                                         c_mean_global::Real,
                                         rho_mean_global::Real)
    return AcousticPerturbationEquations2D(SVector(v_mean_global), c_mean_global,
                                           rho_mean_global)
end

function varnames(::typeof(cons2cons), ::AcousticPerturbationEquations2D)
    ("v1_prime", "v2_prime", "p_prime_scaled",
     "v1_mean", "v2_mean", "c_mean", "rho_mean")
end
function varnames(::typeof(cons2prim), ::AcousticPerturbationEquations2D)
    ("v1_prime", "v2_prime", "p_prime",
     "v1_mean", "v2_mean", "c_mean", "rho_mean")
end

# Convenience functions for retrieving state variables and mean variables
function cons2state(u, equations::AcousticPerturbationEquations2D)
    return SVector(u[1], u[2], u[3])
end

function cons2mean(u, equations::AcousticPerturbationEquations2D)
    return SVector(u[4], u[5], u[6], u[7])
end

function varnames(::typeof(cons2state), ::AcousticPerturbationEquations2D)
    ("v1_prime", "v2_prime", "p_prime_scaled")
end
function varnames(::typeof(cons2mean), ::AcousticPerturbationEquations2D)
    ("v1_mean", "v2_mean", "c_mean", "rho_mean")
end

"""
    global_mean_vars(equations::AcousticPerturbationEquations2D)

Returns the global mean variables stored in `equations`. This makes it easier
to define flexible initial conditions for problems with constant mean flow.
"""
function global_mean_vars(equations::AcousticPerturbationEquations2D)
    return equations.v_mean_global[1], equations.v_mean_global[2],
           equations.c_mean_global,
           equations.rho_mean_global
end

"""
    initial_condition_constant(x, t, equations::AcousticPerturbationEquations2D)

A constant initial condition where the state variables are zero and the mean flow is constant.
Uses the global mean values from `equations`.
"""
function initial_condition_constant(x, t, equations::AcousticPerturbationEquations2D)
    v1_prime = 0
    v2_prime = 0
    p_prime_scaled = 0

    return SVector(v1_prime, v2_prime, p_prime_scaled, global_mean_vars(equations)...)
end

"""
    initial_condition_convergence_test(x, t, equations::AcousticPerturbationEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref). Uses the global mean values from `equations`.
"""
function initial_condition_convergence_test(x, t,
                                            equations::AcousticPerturbationEquations2D)
    RealT = eltype(x)
    a = 1
    c = 2
    L = 2
    f = 2.0f0 / L
    A = convert(RealT, 0.2)
    omega = 2 * convert(RealT, pi) * f
    init = c + A * sin(omega * (x[1] + x[2] - a * t))

    v1_prime = init
    v2_prime = init
    p_prime = init^2

    prim = SVector(v1_prime, v2_prime, p_prime, global_mean_vars(equations)...)

    return prim2cons(prim, equations)
end

"""
    source_terms_convergence_test(u, x, t, equations::AcousticPerturbationEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref).
"""
function source_terms_convergence_test(u, x, t,
                                       equations::AcousticPerturbationEquations2D)
    v1_mean, v2_mean, c_mean, rho_mean = cons2mean(u, equations)

    RealT = eltype(u)
    a = 1
    c = 2
    L = 2
    f = 2.0f0 / L
    A = convert(RealT, 0.2)
    omega = 2 * convert(RealT, pi) * f

    si, co = sincos(omega * (x[1] + x[2] - a * t))
    tmp = v1_mean + v2_mean - a

    du1 = du2 = A * omega * co * (2 * c / rho_mean + tmp + 2 / rho_mean * A * si)
    du3 = A * omega * co * (2 * c_mean^2 * rho_mean + 2 * c * tmp + 2 * A * tmp * si) /
          c_mean^2

    du4 = du5 = du6 = du7 = 0

    return SVector(du1, du2, du3, du4, du5, du6, du7)
end

"""
    initial_condition_gauss(x, t, equations::AcousticPerturbationEquations2D)

A Gaussian pulse in a constant mean flow. Uses the global mean values from `equations`.
"""
function initial_condition_gauss(x, t, equations::AcousticPerturbationEquations2D)
    v1_prime = 0
    v2_prime = 0
    p_prime = exp(-4 * (x[1]^2 + x[2]^2))

    prim = SVector(v1_prime, v2_prime, p_prime, global_mean_vars(equations)...)

    return prim2cons(prim, equations)
end

"""
    boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                            equations::AcousticPerturbationEquations2D)

Boundary conditions for a solid wall.
"""
function boundary_condition_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function,
                                 equations::AcousticPerturbationEquations2D)
    # Boundary state is equal to the inner state except for the perturbed velocity. For boundaries
    # in the -x/+x direction, we multiply the perturbed velocity in the x direction by -1.
    # Similarly, for boundaries in the -y/+y direction, we multiply the perturbed velocity in the
    # y direction by -1
    if direction in (1, 2) # x direction
        u_boundary = SVector(-u_inner[1], u_inner[2], u_inner[3],
                             cons2mean(u_inner, equations)...)
    else # y direction
        u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3],
                             cons2mean(u_inner, equations)...)
    end

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::AcousticPerturbationEquations2D)

Use an orthogonal projection of the perturbed velocities to zero out the normal velocity
while retaining the possibility of a tangential velocity in the boundary state.
Further details are available in the paper:
- Marcus Bauer, Jürgen Dierke and Roland Ewert (2011)
  Application of a discontinuous Galerkin method to discretize acoustic perturbation equations
  [DOI: 10.2514/1.J050333](https://doi.org/10.2514/1.J050333)
"""
function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, x, t,
                                      surface_flux_function,
                                      equations::AcousticPerturbationEquations2D)
    # normalize the outward pointing direction
    normal = normal_direction / norm(normal_direction)

    # compute the normal perturbed velocity
    u_normal = normal[1] * u_inner[1] + normal[2] * u_inner[2]

    # create the "external" boundary solution state
    u_boundary = SVector(u_inner[1] - 2 * u_normal * normal[1],
                         u_inner[2] - 2 * u_normal * normal[2],
                         u_inner[3], cons2mean(u_inner, equations)...)

    # calculate the boundary flux
    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

    return flux
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::AcousticPerturbationEquations2D)
    v1_prime, v2_prime, p_prime_scaled = cons2state(u, equations)
    v1_mean, v2_mean, c_mean, rho_mean = cons2mean(u, equations)

    # Calculate flux for conservative state variables
    RealT = eltype(u)
    if orientation == 1
        f1 = v1_mean * v1_prime + v2_mean * v2_prime +
             c_mean^2 * p_prime_scaled / rho_mean
        f2 = zero(RealT)
        f3 = rho_mean * v1_prime + v1_mean * p_prime_scaled
    else
        f1 = zero(RealT)
        f2 = v1_mean * v1_prime + v2_mean * v2_prime +
             c_mean^2 * p_prime_scaled / rho_mean
        f3 = rho_mean * v2_prime + v2_mean * p_prime_scaled
    end

    # The rest of the state variables are actually variable coefficients, hence the flux should be
    # zero. See https://github.com/trixi-framework/Trixi.jl/issues/358#issuecomment-784828762
    # for details.
    f4 = f5 = f6 = f7 = 0

    return SVector(f1, f2, f3, f4, f5, f6, f7)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::AcousticPerturbationEquations2D)
    # Calculate v = v_prime + v_mean
    v_prime_ll = u_ll[orientation]
    v_prime_rr = u_rr[orientation]
    v_mean_ll = u_ll[orientation + 3]
    v_mean_rr = u_rr[orientation + 3]

    v_ll = v_prime_ll + v_mean_ll
    v_rr = v_prime_rr + v_mean_rr

    c_mean_ll = u_ll[6]
    c_mean_rr = u_rr[6]

    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_mean_ll, c_mean_rr)
end

# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector,
                      equations::AcousticPerturbationEquations2D)
    v1_prime, v2_prime, p_prime_scaled = cons2state(u, equations)
    v1_mean, v2_mean, c_mean, rho_mean = cons2mean(u, equations)

    f1 = normal_direction[1] * (v1_mean * v1_prime + v2_mean * v2_prime +
          c_mean^2 * p_prime_scaled / rho_mean)
    f2 = normal_direction[2] * (v1_mean * v1_prime + v2_mean * v2_prime +
          c_mean^2 * p_prime_scaled / rho_mean)
    f3 = (normal_direction[1] * (rho_mean * v1_prime + v1_mean * p_prime_scaled)
          +
          normal_direction[2] * (rho_mean * v2_prime + v2_mean * p_prime_scaled))

    # The rest of the state variables are actually variable coefficients, hence the flux should be
    # zero. See https://github.com/trixi-framework/Trixi.jl/issues/358#issuecomment-784828762
    # for details.
    f4 = f5 = f6 = f7 = 0

    return SVector(f1, f2, f3, f4, f5, f6, f7)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::AcousticPerturbationEquations2D)
    # Calculate v = v_prime + v_mean
    v_prime_ll = normal_direction[1] * u_ll[1] + normal_direction[2] * u_ll[2]
    v_prime_rr = normal_direction[1] * u_rr[1] + normal_direction[2] * u_rr[2]
    v_mean_ll = normal_direction[1] * u_ll[4] + normal_direction[2] * u_ll[5]
    v_mean_rr = normal_direction[1] * u_rr[4] + normal_direction[2] * u_rr[5]

    v_ll = v_prime_ll + v_mean_ll
    v_rr = v_prime_rr + v_mean_rr

    c_mean_ll = u_ll[6]
    c_mean_rr = u_rr[6]

    # The v_normals are already scaled by the norm
    λ_max = max(abs(v_ll), abs(v_rr)) +
            max(c_mean_ll, c_mean_rr) * norm(normal_direction)
end

# Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the mean values
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr,
                                                              orientation_or_normal_direction,
                                                              equations::AcousticPerturbationEquations2D)
    λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction,
                                  equations)
    diss = -0.5f0 * λ * (u_rr - u_ll)
    z = 0

    return SVector(diss[1], diss[2], diss[3], z, z, z, z)
end

@inline have_constant_speed(::AcousticPerturbationEquations2D) = False()

@inline function max_abs_speeds(u, equations::AcousticPerturbationEquations2D)
    v1_mean = u[4]
    v2_mean = u[5]
    c_mean = u[6]

    return abs(v1_mean) + c_mean, abs(v2_mean) + c_mean
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::AcousticPerturbationEquations2D)
    p_prime_scaled = u[3]
    c_mean = u[6]
    p_prime = p_prime_scaled * c_mean^2

    return SVector(u[1], u[2], p_prime, u[4], u[5], u[6], u[7])
end

# Convert primitive variables to conservative
@inline function prim2cons(u, equations::AcousticPerturbationEquations2D)
    p_prime = u[3]
    c_mean = u[6]
    p_prime_scaled = p_prime / c_mean^2

    return SVector(u[1], u[2], p_prime_scaled, u[4], u[5], u[6], u[7])
end

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equations::AcousticPerturbationEquations2D) = u
end # @muladd
