# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LatticeBoltzmannEquations3D(; Ma, Re, collision_op=collision_bgk,
                               c=1, L=1, rho0=1, u0=nothing, nu=nothing)

The Lattice-Boltzmann equations
```math
\partial_t u_\alpha + v_{\alpha,1} \partial_1 u_\alpha + v_{\alpha,2} \partial_2 u_\alpha + v_{\alpha,3} \partial_3 u_\alpha = 0
```
in three space dimensions for the D3Q27 scheme.

The characteristic Mach number and Reynolds numbers are specified as `Ma` and `Re`. By the
default, the collision operator `collision_op` is set to the BGK model. `c`, `L`, and `rho0`
specify the mean thermal molecular velocity, the characteristic length, and the reference density,
respectively. They can usually be left to the default values. If desired, instead of the Mach
number, one can set the macroscopic reference velocity `u0` directly (`Ma` needs to be set to
`nothing` in this case). Likewise, instead of the Reynolds number one can specify the kinematic
viscosity `nu` directly (in this case, `Re` needs to be set to `nothing`).


The twenty-seven discrete velocity directions of the D3Q27 scheme are sorted as follows [4]:
* plane at `z = -1`:
  ```
    24    17     21       y
       ┌───┼───┐          │
       │       │          │
    10 ┼   6   ┼ 15        ──── x
       │       │         ╱
       └───┼───┘        ╱
    20    12     26    z
  ```
* plane at `z = 0`:
  ```
    14     3     7        y
       ┌───┼───┐          │
       │       │          │
     2 ┼  27   ┼ 1         ──── x
       │       │         ╱
       └───┼───┘        ╱
     8     4     13    z
  ```
* plane at `z = +1`:
  ```
    25    11     19       y
       ┌───┼───┐          │
       │       │          │
    16 ┼   5   ┼ 9         ──── x
       │       │         ╱
       └───┼───┘        ╱
    22    18     23    z
  ```
Note that usually the velocities are numbered from `0` to `26`, where `0` corresponds to the zero
velocity. Due to Julia using 1-based indexing, here we use indices from `1` to `27`, where `1`
through `26` correspond to the velocity directions in [4] and `27` is the zero velocity.

The corresponding opposite directions are:
*  1 ←→  2
*  2 ←→  1
*  3 ←→  4
*  4 ←→  3
*  5 ←→  6
*  6 ←→  5
*  7 ←→  8
*  8 ←→  7
*  9 ←→ 10
* 10 ←→  9
* 11 ←→ 12
* 12 ←→ 11
* 13 ←→ 14
* 14 ←→ 13
* 15 ←→ 16
* 16 ←→ 15
* 17 ←→ 18
* 18 ←→ 17
* 19 ←→ 20
* 20 ←→ 19
* 21 ←→ 22
* 22 ←→ 21
* 23 ←→ 24
* 24 ←→ 23
* 25 ←→ 26
* 26 ←→ 25
* 27 ←→ 27

The main sources for the base implementation were
1. Misun Min, Taehun Lee, **A spectral-element discontinuous Galerkin lattice Boltzmann method for
   nearly incompressible flows**, J Comput Phys 230(1), 2011
   [doi:10.1016/j.jcp.2010.09.024](https://doi.org/10.1016/j.jcp.2010.09.024)
2. Karsten Golly, **Anwendung der Lattice-Boltzmann Discontinuous Galerkin Spectral Element Method
   (LB-DGSEM) auf laminare und turbulente nahezu inkompressible Strömungen im dreidimensionalen
   Raum**, Master Thesis, University of Cologne, 2018.
3. Dieter Hänel, **Molekulare Gasdynamik**, Springer-Verlag Berlin Heidelberg, 2004
   [doi:10.1007/3-540-35047-0](https://doi.org/10.1007/3-540-35047-0)
4. Dieter Krüger et al., **The Lattice Boltzmann Method**, Springer International Publishing, 2017
   [doi:10.1007/978-3-319-44649-3](https://doi.org/10.1007/978-3-319-44649-3)
"""
struct LatticeBoltzmannEquations3D{RealT <: Real, CollisionOp} <:
       AbstractLatticeBoltzmannEquations{3, 27}
    c::RealT    # mean thermal molecular velocity
    c_s::RealT  # isothermal speed of sound
    rho0::RealT # macroscopic reference density

    Ma::RealT   # characteristic Mach number
    u0::RealT   # macroscopic reference velocity

    Re::RealT   # characteristic Reynolds number
    L::RealT    # reference length
    nu::RealT   # kinematic viscosity

    weights::SVector{27, RealT}  # weighting factors for the equilibrium distribution
    v_alpha1::SVector{27, RealT} # discrete molecular velocity components in x-direction
    v_alpha2::SVector{27, RealT} # discrete molecular velocity components in y-direction
    v_alpha3::SVector{27, RealT} # discrete molecular velocity components in z-direction

    collision_op::CollisionOp   # collision operator for the collision kernel
end

function LatticeBoltzmannEquations3D(; Ma, Re, collision_op = collision_bgk,
                                     c = 1, L = 1, rho0 = 1, u0 = nothing, nu = nothing)
    # Sanity check that exactly one of Ma, u0 is not `nothing`
    if isnothing(Ma) && isnothing(u0)
        error("Mach number `Ma` and reference speed `u0` may not both be `nothing`")
    elseif !isnothing(Ma) && !isnothing(u0)
        error("Mach number `Ma` and reference speed `u0` may not both be set")
    end

    # Sanity check that exactly one of Re, nu is not `nothing`
    if isnothing(Re) && isnothing(nu)
        error("Reynolds number `Re` and visocsity `nu` may not both be `nothing`")
    elseif !isnothing(Re) && !isnothing(nu)
        error("Reynolds number `Re` and visocsity `nu` may not both be set")
    end

    # Calculate isothermal speed of sound
    # The relation between the isothermal speed of sound `c_s` and the mean thermal molecular velocity
    # `c` depends on the used phase space discretization, and is valid for D3Q27 (and others). For
    # details, see, e.g., [3] in the docstring above.
    # c_s = c / sqrt(3)

    # Calculate missing quantities
    if isnothing(Ma)
        RealT = eltype(u0)
        c_s = c / sqrt(convert(RealT, 3))
        Ma = u0 / c_s
    elseif isnothing(u0)
        RealT = eltype(Ma)
        c_s = c / sqrt(convert(RealT, 3))
        u0 = Ma * c_s
    end
    if isnothing(Re)
        Re = u0 * L / nu
    elseif isnothing(nu)
        nu = u0 * L / Re
    end

    # Promote to common data type
    Ma, Re, c, L, rho0, u0, nu = promote(Ma, Re, c, L, rho0, u0, nu)

    # Source for weights and speeds: [4] in docstring above
    weights = SVector{27, RealT}(2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27, 1 / 54,
                                 1 / 54,
                                 1 / 54,
                                 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54,
                                 1 / 54,
                                 1 / 54,
                                 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216,
                                 1 / 216,
                                 1 / 216, 8 / 27)
    v_alpha1 = SVector{27, RealT}(c, -c, 0, 0, 0, 0, c, -c, c,
                                  -c, 0, 0, c, -c, c, -c, 0, 0,
                                  c, -c, c, -c, c, -c, -c, c, 0)
    v_alpha2 = SVector{27, RealT}(0, 0, c, -c, 0, 0, c, -c, 0,
                                  0, c, -c, -c, c, 0, 0, c, -c,
                                  c, -c, c, -c, -c, c, c, -c, 0)
    v_alpha3 = SVector{27, RealT}(0, 0, 0, 0, c, -c, 0, 0, c,
                                  -c, c, -c, 0, 0, -c, c, -c, c,
                                  c, -c, -c, c, c, -c, c, -c, 0)

    LatticeBoltzmannEquations3D(c, c_s, rho0, Ma, u0, Re, L, nu,
                                weights, v_alpha1, v_alpha2, v_alpha3,
                                collision_op)
end

function varnames(::typeof(cons2cons), equations::LatticeBoltzmannEquations3D)
    ntuple(v -> "pdf" * string(v), Val(nvariables(equations)))
end
function varnames(::typeof(cons2prim), equations::LatticeBoltzmannEquations3D)
    varnames(cons2cons, equations)
end

# Convert conservative variables to macroscopic
@inline function cons2macroscopic(u, equations::LatticeBoltzmannEquations3D)
    rho = density(u, equations)
    v1, v2, v3 = velocity(u, equations)
    p = pressure(u, equations)
    return SVector(rho, v1, v2, v3, p)
end
function varnames(::typeof(cons2macroscopic), ::LatticeBoltzmannEquations3D)
    ("rho", "v1", "v2", "v3", "p")
end

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::LatticeBoltzmannEquations3D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::LatticeBoltzmannEquations3D)
    @unpack u0 = equations

    RealT = eltype(x)
    rho = convert(RealT, pi)
    v1 = u0
    v2 = u0
    v3 = u0

    return equilibrium_distribution(rho, v1, v2, v3, equations)
end

# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::LatticeBoltzmannEquations3D)

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::LatticeBoltzmannEquations3D)
    if orientation == 1 # x-direction
        v_alpha = equations.v_alpha1
    elseif orientation == 2 # y-direction
        v_alpha = equations.v_alpha2
    else # z-direction
        v_alpha = equations.v_alpha3
    end
    return v_alpha .* u
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
# @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::LatticeBoltzmannEquations3D)
#   λ_max =
# end

@inline function flux_godunov(u_ll, u_rr, orientation::Integer,
                              equations::LatticeBoltzmannEquations3D)
    if orientation == 1 # x-direction
        v_alpha = equations.v_alpha1
    elseif orientation == 2 # y-direction
        v_alpha = equations.v_alpha2
    else # z-direction
        v_alpha = equations.v_alpha3
    end
    return 0.5f0 * (v_alpha .* (u_ll + u_rr) - abs.(v_alpha) .* (u_rr - u_ll))
end

"""
    density(p::Real, equations::LatticeBoltzmannEquations3D)
    density(u, equations::LatticeBoltzmannEquations3D)

Calculate the macroscopic density from the pressure `p` or the particle distribution functions `u`.
"""
@inline density(p::Real, equations::LatticeBoltzmannEquations3D) = p / equations.c_s^2
@inline density(u, equations::LatticeBoltzmannEquations3D) = sum(u)

"""
    velocity(u, orientation, equations::LatticeBoltzmannEquations3D)

Calculate the macroscopic velocity for the given `orientation` (1 -> x, 2 -> y, 3 -> z) from the
particle distribution functions `u`.
"""
@inline function velocity(u, orientation::Integer,
                          equations::LatticeBoltzmannEquations3D)
    if orientation == 1 # x-direction
        v_alpha = equations.v_alpha1
    elseif orientation == 2 # y-direction
        v_alpha = equations.v_alpha2
    else # z-direction
        v_alpha = equations.v_alpha3
    end

    return dot(v_alpha, u) / density(u, equations)
end

"""
    velocity(u, equations::LatticeBoltzmannEquations3D)

Calculate the macroscopic velocity vector from the particle distribution functions `u`.
"""
@inline function velocity(u, equations::LatticeBoltzmannEquations3D)
    @unpack v_alpha1, v_alpha2, v_alpha3 = equations
    rho = density(u, equations)

    return SVector(dot(v_alpha1, u) / rho,
                   dot(v_alpha2, u) / rho,
                   dot(v_alpha3, u) / rho)
end

"""
    pressure(rho::Real, equations::LatticeBoltzmannEquations3D)
    pressure(u, equations::LatticeBoltzmannEquations3D)

Calculate the macroscopic pressure from the density `rho` or the  particle distribution functions
`u`.
"""
@inline function pressure(rho::Real, equations::LatticeBoltzmannEquations3D)
    rho * equations.c_s^2
end
@inline function pressure(u, equations::LatticeBoltzmannEquations3D)
    pressure(density(u, equations), equations)
end

"""
    equilibrium_distribution(alpha, rho, v1, v2, v3, equations::LatticeBoltzmannEquations3D)

Calculate the local equilibrium distribution for the distribution function with index `alpha` and
given the macroscopic state defined by `rho`, `v1`, `v2`, `v3`.
"""
@inline function equilibrium_distribution(alpha, rho, v1, v2, v3,
                                          equations::LatticeBoltzmannEquations3D)
    @unpack weights, c_s, v_alpha1, v_alpha2, v_alpha3 = equations

    va_v = v_alpha1[alpha] * v1 + v_alpha2[alpha] * v2 + v_alpha3[alpha] * v3
    cs_squared = c_s^2
    v_squared = v1^2 + v2^2 + v3^2

    return weights[alpha] * rho *
           (1 + va_v / cs_squared
            + va_v^2 / (2 * cs_squared^2)
            -
            v_squared / (2 * cs_squared))
end

@inline function equilibrium_distribution(rho, v1, v2, v3,
                                          equations::LatticeBoltzmannEquations3D)
    return SVector(equilibrium_distribution(1, rho, v1, v2, v3, equations),
                   equilibrium_distribution(2, rho, v1, v2, v3, equations),
                   equilibrium_distribution(3, rho, v1, v2, v3, equations),
                   equilibrium_distribution(4, rho, v1, v2, v3, equations),
                   equilibrium_distribution(5, rho, v1, v2, v3, equations),
                   equilibrium_distribution(6, rho, v1, v2, v3, equations),
                   equilibrium_distribution(7, rho, v1, v2, v3, equations),
                   equilibrium_distribution(8, rho, v1, v2, v3, equations),
                   equilibrium_distribution(9, rho, v1, v2, v3, equations),
                   equilibrium_distribution(10, rho, v1, v2, v3, equations),
                   equilibrium_distribution(11, rho, v1, v2, v3, equations),
                   equilibrium_distribution(12, rho, v1, v2, v3, equations),
                   equilibrium_distribution(13, rho, v1, v2, v3, equations),
                   equilibrium_distribution(14, rho, v1, v2, v3, equations),
                   equilibrium_distribution(15, rho, v1, v2, v3, equations),
                   equilibrium_distribution(16, rho, v1, v2, v3, equations),
                   equilibrium_distribution(17, rho, v1, v2, v3, equations),
                   equilibrium_distribution(18, rho, v1, v2, v3, equations),
                   equilibrium_distribution(19, rho, v1, v2, v3, equations),
                   equilibrium_distribution(20, rho, v1, v2, v3, equations),
                   equilibrium_distribution(21, rho, v1, v2, v3, equations),
                   equilibrium_distribution(22, rho, v1, v2, v3, equations),
                   equilibrium_distribution(23, rho, v1, v2, v3, equations),
                   equilibrium_distribution(24, rho, v1, v2, v3, equations),
                   equilibrium_distribution(25, rho, v1, v2, v3, equations),
                   equilibrium_distribution(26, rho, v1, v2, v3, equations),
                   equilibrium_distribution(27, rho, v1, v2, v3, equations))
end

function equilibrium_distribution(u, equations::LatticeBoltzmannEquations3D)
    rho = density(u, equations)
    v1, v2, v3 = velocity(u, equations)

    return equilibrium_distribution(rho, v1, v2, v3, equations)
end

"""
    collision_bgk(u, dt, equations::LatticeBoltzmannEquations3D)

Collision operator for the Bhatnagar, Gross, and Krook (BGK) model.
"""
@inline function collision_bgk(u, dt, equations::LatticeBoltzmannEquations3D)
    @unpack c_s, nu = equations
    tau = nu / (c_s^2 * dt)
    return -(u - equilibrium_distribution(u, equations)) / (tau + 0.5f0)
end

@inline have_constant_speed(::LatticeBoltzmannEquations3D) = True()

@inline function max_abs_speeds(equations::LatticeBoltzmannEquations3D)
    @unpack c = equations

    return c, c, c
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LatticeBoltzmannEquations3D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equations::LatticeBoltzmannEquations3D) = u

# Calculate kinetic energy for a conservative state `u`
@inline function energy_kinetic(u, equations::LatticeBoltzmannEquations3D)
    rho = density(u, equations)
    v1, v2, v3 = velocity(u, equations)

    return 0.5f0 * (v1^2 + v2^2 + v3^2) / rho / equations.rho0
end

# Calculate nondimensionalized kinetic energy for a conservative state `u`
@inline function energy_kinetic_nondimensional(u,
                                               equations::LatticeBoltzmannEquations3D)
    return energy_kinetic(u, equations) / equations.u0^2
end
end # @muladd
