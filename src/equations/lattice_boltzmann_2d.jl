# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LatticeBoltzmannEquations2D(; Ma, Re, collision_op=collision_bgk,
                               c=1, L=1, rho0=1, u0=nothing, nu=nothing)

The Lattice-Boltzmann equations
```math
\partial_t u_\alpha + v_{\alpha,1} \partial_1 u_\alpha + v_{\alpha,2} \partial_2 u_\alpha = 0
```
in two space dimensions for the D2Q9 scheme.

The characteristic Mach number and Reynolds numbers are specified as `Ma` and `Re`. By the
default, the collision operator `collision_op` is set to the BGK model. `c`, `L`, and `rho0`
specify the mean thermal molecular velocity, the characteristic length, and the reference density,
respectively. They can usually be left to the default values. If desired, instead of the Mach
number, one can set the macroscopic reference velocity `u0` directly (`Ma` needs to be set to
`nothing` in this case). Likewise, instead of the Reynolds number one can specify the kinematic
viscosity `nu` directly (in this case, `Re` needs to be set to `nothing`).


The nine discrete velocity directions of the D2Q9 scheme are sorted as follows [4]:
```
  6     2     5       y
    ┌───┼───┐         │
    │       │         │
  3 ┼   9   ┼ 1        ──── x
    │       │        ╱
    └───┼───┘       ╱
  7     4     8    z
```
Note that usually the velocities are numbered from `0` to `8`, where `0` corresponds to the zero
velocity. Due to Julia using 1-based indexing, here we use indices from `1` to `9`, where `1`
through `8` correspond to the velocity directions in [4] and `9` is the zero velocity.

The corresponding opposite directions are:
* 1 ←→ 3
* 2 ←→ 4
* 3 ←→ 1
* 4 ←→ 2
* 5 ←→ 7
* 6 ←→ 8
* 7 ←→ 5
* 8 ←→ 6
* 9 ←→ 9

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
struct LatticeBoltzmannEquations2D{RealT <: Real, CollisionOp} <:
       AbstractLatticeBoltzmannEquations{2, 9}
    c::RealT    # mean thermal molecular velocity
    c_s::RealT  # isothermal speed of sound
    rho0::RealT # macroscopic reference density

    Ma::RealT   # characteristic Mach number
    u0::RealT   # macroscopic reference velocity

    Re::RealT   # characteristic Reynolds number
    L::RealT    # reference length
    nu::RealT   # kinematic viscosity

    weights::SVector{9, RealT}  # weighting factors for the equilibrium distribution
    v_alpha1::SVector{9, RealT} # discrete molecular velocity components in x-direction
    v_alpha2::SVector{9, RealT} # discrete molecular velocity components in y-direction

    collision_op::CollisionOp   # collision operator for the collision kernel
end

function LatticeBoltzmannEquations2D(; Ma, Re, collision_op = collision_bgk,
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
    # `c` depends on the used phase space discretization, and is valid for D2Q9 (and others). For
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

    # Source for weights and speeds: [4] in the docstring above
    weights = SVector{9, RealT}(1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36,
                                1 / 36, 4 / 9)
    v_alpha1 = SVector{9, RealT}(c, 0, -c, 0, c, -c, -c, c, 0)
    v_alpha2 = SVector{9, RealT}(0, c, 0, -c, c, c, -c, -c, 0)

    LatticeBoltzmannEquations2D(c, c_s, rho0, Ma, u0, Re, L, nu,
                                weights, v_alpha1, v_alpha2,
                                collision_op)
end

function varnames(::typeof(cons2cons), equations::LatticeBoltzmannEquations2D)
    ntuple(v -> "pdf" * string(v), nvariables(equations))
end
function varnames(::typeof(cons2prim), equations::LatticeBoltzmannEquations2D)
    varnames(cons2cons, equations)
end

# Convert conservative variables to macroscopic
@inline function cons2macroscopic(u, equations::LatticeBoltzmannEquations2D)
    rho = density(u, equations)
    v1, v2 = velocity(u, equations)
    p = pressure(u, equations)
    return SVector(rho, v1, v2, p)
end
function varnames(::typeof(cons2macroscopic), ::LatticeBoltzmannEquations2D)
    ("rho", "v1", "v2", "p")
end

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::LatticeBoltzmannEquations2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::LatticeBoltzmannEquations2D)
    @unpack u0 = equations

    RealT = eltype(x)
    rho = convert(RealT, pi)
    v1 = u0
    v2 = u0

    return equilibrium_distribution(rho, v1, v2, equations)
end

"""
    boundary_condition_noslip_wall(u_inner, orientation, direction, x, t,
                                   surface_flux_function,
                                   equations::LatticeBoltzmannEquations2D)

No-slip wall boundary condition using the bounce-back approach.
"""
@inline function boundary_condition_noslip_wall(u_inner, orientation, direction, x, t,
                                                surface_flux_function,
                                                equations::LatticeBoltzmannEquations2D)
    # For LBM no-slip wall boundary conditions, we set the boundary state to
    # - the inner state for outgoing particle distribution functions
    # - the *opposite* inner state for all other particle distribution functions
    # See the list of (opposite) directions in the docstring of `LatticeBoltzmannEquations2D`.
    if direction == 1 # boundary in -x direction
        pdf1 = u_inner[3]
        pdf2 = u_inner[4]
        pdf3 = u_inner[3] # outgoing
        pdf4 = u_inner[2]
        pdf5 = u_inner[7]
        pdf6 = u_inner[6] # outgoing
        pdf7 = u_inner[7] # outgoing
        pdf8 = u_inner[6]
        pdf9 = u_inner[9]
    elseif direction == 2 # boundary in +x direction
        pdf1 = u_inner[1] # outgoing
        pdf2 = u_inner[4]
        pdf3 = u_inner[1]
        pdf4 = u_inner[2]
        pdf5 = u_inner[5] # outgoing
        pdf6 = u_inner[8]
        pdf7 = u_inner[5]
        pdf8 = u_inner[8] # outgoing
        pdf9 = u_inner[9]
    elseif direction == 3 # boundary in -y direction
        pdf1 = u_inner[3]
        pdf2 = u_inner[4]
        pdf3 = u_inner[1]
        pdf4 = u_inner[4] # outgoing
        pdf5 = u_inner[7]
        pdf6 = u_inner[8]
        pdf7 = u_inner[7] # outgoing
        pdf8 = u_inner[8] # outgoing
        pdf9 = u_inner[9]
    else # boundary in +y direction
        pdf1 = u_inner[3]
        pdf2 = u_inner[2] # outgoing
        pdf3 = u_inner[1]
        pdf4 = u_inner[2]
        pdf5 = u_inner[5] # outgoing
        pdf6 = u_inner[6] # outgoing
        pdf7 = u_inner[5]
        pdf8 = u_inner[6]
        pdf9 = u_inner[9]
    end
    u_boundary = SVector(pdf1, pdf2, pdf3, pdf4, pdf5, pdf6, pdf7, pdf8, pdf9)

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::LatticeBoltzmannEquations2D)

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::LatticeBoltzmannEquations2D)
    if orientation == 1
        v_alpha = equations.v_alpha1
    else
        v_alpha = equations.v_alpha2
    end
    return v_alpha .* u
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
# @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::LatticeBoltzmannEquations2D)
#   λ_max =
# end

@inline function flux_godunov(u_ll, u_rr, orientation::Integer,
                              equations::LatticeBoltzmannEquations2D)
    if orientation == 1
        v_alpha = equations.v_alpha1
    else
        v_alpha = equations.v_alpha2
    end
    return 0.5f0 * (v_alpha .* (u_ll + u_rr) - abs.(v_alpha) .* (u_rr - u_ll))
end

"""
    density(p::Real, equations::LatticeBoltzmannEquations2D)
    density(u, equations::LatticeBoltzmannEquations2D)

Calculate the macroscopic density from the pressure `p` or the particle distribution functions `u`.
"""
@inline density(p::Real, equations::LatticeBoltzmannEquations2D) = p / equations.c_s^2
@inline density(u, equations::LatticeBoltzmannEquations2D) = sum(u)

"""
    velocity(u, orientation, equations::LatticeBoltzmannEquations2D)

Calculate the macroscopic velocity for the given `orientation` (1 -> x, 2 -> y) from the
particle distribution functions `u`.
"""
@inline function velocity(u, orientation::Integer,
                          equations::LatticeBoltzmannEquations2D)
    if orientation == 1
        v_alpha = equations.v_alpha1
    else
        v_alpha = equations.v_alpha2
    end

    return dot(v_alpha, u) / density(u, equations)
end

"""
    velocity(u, equations::LatticeBoltzmannEquations2D)

Calculate the macroscopic velocity vector from the particle distribution functions `u`.
"""
@inline function velocity(u, equations::LatticeBoltzmannEquations2D)
    @unpack v_alpha1, v_alpha2 = equations
    rho = density(u, equations)

    return SVector(dot(v_alpha1, u) / rho,
                   dot(v_alpha2, u) / rho)
end

"""
    pressure(rho::Real, equations::LatticeBoltzmannEquations2D)
    pressure(u, equations::LatticeBoltzmannEquations2D)

Calculate the macroscopic pressure from the density `rho` or the  particle distribution functions
`u`.
"""
@inline function pressure(rho::Real, equations::LatticeBoltzmannEquations2D)
    rho * equations.c_s^2
end
@inline function pressure(u, equations::LatticeBoltzmannEquations2D)
    pressure(density(u, equations), equations)
end

"""
    equilibrium_distribution(alpha, rho, v1, v2, equations::LatticeBoltzmannEquations2D)

Calculate the local equilibrium distribution for the distribution function with index `alpha` and
given the macroscopic state defined by `rho`, `v1`, `v2`.
"""
@inline function equilibrium_distribution(alpha, rho, v1, v2,
                                          equations::LatticeBoltzmannEquations2D)
    @unpack weights, c_s, v_alpha1, v_alpha2 = equations

    va_v = v_alpha1[alpha] * v1 + v_alpha2[alpha] * v2
    cs_squared = c_s^2
    v_squared = v1^2 + v2^2

    return weights[alpha] * rho *
           (1 + va_v / cs_squared
            + va_v^2 / (2 * cs_squared^2)
            -
            v_squared / (2 * cs_squared))
end

@inline function equilibrium_distribution(rho, v1, v2,
                                          equations::LatticeBoltzmannEquations2D)
    return SVector(equilibrium_distribution(1, rho, v1, v2, equations),
                   equilibrium_distribution(2, rho, v1, v2, equations),
                   equilibrium_distribution(3, rho, v1, v2, equations),
                   equilibrium_distribution(4, rho, v1, v2, equations),
                   equilibrium_distribution(5, rho, v1, v2, equations),
                   equilibrium_distribution(6, rho, v1, v2, equations),
                   equilibrium_distribution(7, rho, v1, v2, equations),
                   equilibrium_distribution(8, rho, v1, v2, equations),
                   equilibrium_distribution(9, rho, v1, v2, equations))
end

function equilibrium_distribution(u, equations::LatticeBoltzmannEquations2D)
    rho = density(u, equations)
    v1, v2 = velocity(u, equations)

    return equilibrium_distribution(rho, v1, v2, equations)
end

"""
    collision_bgk(u, dt, equations::LatticeBoltzmannEquations2D)

Collision operator for the Bhatnagar, Gross, and Krook (BGK) model.
"""
@inline function collision_bgk(u, dt, equations::LatticeBoltzmannEquations2D)
    @unpack c_s, nu = equations
    tau = nu / (c_s^2 * dt)
    return -(u - equilibrium_distribution(u, equations)) / (tau + 0.5f0)
end

@inline have_constant_speed(::LatticeBoltzmannEquations2D) = True()

@inline function max_abs_speeds(equations::LatticeBoltzmannEquations2D)
    @unpack c = equations

    return c, c
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LatticeBoltzmannEquations2D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equations::LatticeBoltzmannEquations2D) = u
end # @muladd
