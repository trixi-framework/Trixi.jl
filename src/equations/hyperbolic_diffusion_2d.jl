# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    HyperbolicDiffusionEquations2D

The linear hyperbolic diffusion equations in two space dimensions.
A description of this system can be found in Sec. 2.5 of the book "I Do Like CFD, Too: Vol 1".
The book is freely available at [http://www.cfdbooks.com/](http://www.cfdbooks.com/) and further analysis can be found in
the paper by Nishikawa [DOI: 10.1016/j.jcp.2007.07.029](https://doi.org/10.1016/j.jcp.2007.07.029)
"""
struct HyperbolicDiffusionEquations2D{RealT <: Real} <:
       AbstractHyperbolicDiffusionEquations{2, 3}
    Lr::RealT     # reference length scale
    inv_Tr::RealT # inverse of the reference time scale
    nu::RealT     # diffusion constant
end

function HyperbolicDiffusionEquations2D(; nu = 1.0, Lr = inv(2pi))
    Tr = Lr^2 / nu
    HyperbolicDiffusionEquations2D(promote(Lr, inv(Tr), nu)...)
end

varnames(::typeof(cons2cons), ::HyperbolicDiffusionEquations2D) = ("phi", "q1", "q2")
varnames(::typeof(cons2prim), ::HyperbolicDiffusionEquations2D) = ("phi", "q1", "q2")
function default_analysis_errors(::HyperbolicDiffusionEquations2D)
    (:l2_error, :linf_error, :residual)
end

@inline function residual_steady_state(du, ::HyperbolicDiffusionEquations2D)
    abs(du[1])
end

# Set initial conditions at physical location `x` for pseudo-time `t`
@inline function initial_condition_poisson_nonperiodic(x, t,
                                                       equations::HyperbolicDiffusionEquations2D)
    # elliptic equation: -ν Δϕ = f in Ω, u = g on ∂Ω
    RealT = eltype(x)
    if iszero(t)
        phi = one(RealT)
        q1 = one(RealT)
        q2 = one(RealT)
    else
        # TODO: sincospi
        sinpi_x1, cospi_x1 = sincos(convert(RealT, pi) * x[1])
        sinpi_2x2, cospi_2x2 = sincos(convert(RealT, pi) * 2 * x[2])
        phi = 2 * cospi_x1 * sinpi_2x2 + 2 # ϕ
        q1 = -2 * convert(RealT, pi) * sinpi_x1 * sinpi_2x2     # ϕ_x
        q2 = 4 * convert(RealT, pi) * cospi_x1 * cospi_2x2     # ϕ_y
    end
    return SVector(phi, q1, q2)
end

@inline function source_terms_poisson_nonperiodic(u, x, t,
                                                  equations::HyperbolicDiffusionEquations2D)
    # elliptic equation: -ν Δϕ = f in Ω, u = g on ∂Ω
    # analytical solution: ϕ = 2cos(πx)sin(2πy) + 2 and f = 10π^2cos(πx)sin(2πy)
    RealT = eltype(u)
    @unpack inv_Tr = equations

    x1, x2 = x
    du1 = 10 * convert(RealT, pi)^2 * cospi(x1) * sinpi(2 * x2)
    du2 = -inv_Tr * u[2]
    du3 = -inv_Tr * u[3]

    return SVector(du1, du2, du3)
end

@inline function boundary_condition_poisson_nonperiodic(u_inner, orientation, direction,
                                                        x, t,
                                                        surface_flux_function,
                                                        equations::HyperbolicDiffusionEquations2D)
    # elliptic equation: -ν Δϕ = f in Ω, u = g on ∂Ω
    u_boundary = initial_condition_poisson_nonperiodic(x, one(t), equations)

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

"""
    source_terms_harmonic(u, x, t, equations::HyperbolicDiffusionEquations2D)

Source term that only includes the forcing from the hyperbolic diffusion system.
"""
@inline function source_terms_harmonic(u, x, t,
                                       equations::HyperbolicDiffusionEquations2D)
    # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
    @unpack inv_Tr = equations
    phi, q1, q2 = u

    du2 = -inv_Tr * q1
    du3 = -inv_Tr * q2

    return SVector(0, du2, du3)
end

"""
    initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::HyperbolicDiffusionEquations2D)

Setup used for convergence tests of the Euler equations with self-gravity used in
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
in combination with [`source_terms_harmonic`](@ref).
"""
function initial_condition_eoc_test_coupled_euler_gravity(x, t,
                                                          equations::HyperbolicDiffusionEquations2D)

    # Determine phi_x, phi_y
    RealT = eltype(x)
    G = 1 # gravitational constant
    C = -2 * G / convert(RealT, pi)
    A = convert(RealT, 0.1) # perturbation coefficient must match Euler setup
    rho1 = A * sinpi(x[1] + x[2] - t)
    # initialize with ansatz of gravity potential
    phi = C * rho1
    q1 = C * A * convert(RealT, pi) * cospi(x[1] + x[2] - t) # = gravity acceleration in x-direction
    q2 = q1                                     # = gravity acceleration in y-direction

    return SVector(phi, q1, q2)
end

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer,
                      equations::HyperbolicDiffusionEquations2D)
    phi, q1, q2 = u
    @unpack inv_Tr = equations

    RealT = eltype(u)
    if orientation == 1
        f1 = -equations.nu * q1
        f2 = -phi * inv_Tr
        f3 = zero(RealT)
    else
        f1 = -equations.nu * q2
        f2 = zero(RealT)
        f3 = -phi * inv_Tr
    end

    return SVector(f1, f2, f3)
end

# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector,
                      equations::HyperbolicDiffusionEquations2D)
    phi, q1, q2 = u
    @unpack inv_Tr = equations

    f1 = -equations.nu * (normal_direction[1] * q1 + normal_direction[2] * q2)
    f2 = -phi * inv_Tr * normal_direction[1]
    f3 = -phi * inv_Tr * normal_direction[2]

    return SVector(f1, f2, f3)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::HyperbolicDiffusionEquations2D)
    sqrt(equations.nu * equations.inv_Tr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::HyperbolicDiffusionEquations2D)
    sqrt(equations.nu * equations.inv_Tr) * norm(normal_direction)
end

@inline function flux_godunov(u_ll, u_rr, orientation::Integer,
                              equations::HyperbolicDiffusionEquations2D)
    # Obtain left and right fluxes
    phi_ll, q1_ll, q2_ll = u_ll
    phi_rr, q1_rr, q2_rr = u_rr
    f_ll = flux(u_ll, orientation, equations)
    f_rr = flux(u_rr, orientation, equations)

    # this is an optimized version of the application of the upwind dissipation matrix:
    #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
    λ_max = sqrt(equations.nu * equations.inv_Tr)
    f1 = 0.5f0 * (f_ll[1] + f_rr[1]) - 0.5f0 * λ_max * (phi_rr - phi_ll)
    if orientation == 1 # x-direction
        f2 = 0.5f0 * (f_ll[2] + f_rr[2]) - 0.5f0 * λ_max * (q1_rr - q1_ll)
        f3 = 0.5f0 * (f_ll[3] + f_rr[3])
    else # y-direction
        f2 = 0.5f0 * (f_ll[2] + f_rr[2])
        f3 = 0.5f0 * (f_ll[3] + f_rr[3]) - 0.5f0 * λ_max * (q2_rr - q2_ll)
    end

    return SVector(f1, f2, f3)
end

@inline function flux_godunov(u_ll, u_rr, normal_direction::AbstractVector,
                              equations::HyperbolicDiffusionEquations2D)
    # Obtain left and right fluxes
    phi_ll, q1_ll, q2_ll = u_ll
    phi_rr, q1_rr, q2_rr = u_rr
    f_ll = flux(u_ll, normal_direction, equations)
    f_rr = flux(u_rr, normal_direction, equations)

    # this is an optimized version of the application of the upwind dissipation matrix:
    #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
    λ_max = sqrt(equations.nu * equations.inv_Tr)
    f1 = 0.5f0 * (f_ll[1] + f_rr[1]) -
         0.5f0 * λ_max * (phi_rr - phi_ll) *
         sqrt(normal_direction[1]^2 + normal_direction[2]^2)
    f2 = 0.5f0 * (f_ll[2] + f_rr[2]) -
         0.5f0 * λ_max * (q1_rr - q1_ll) * normal_direction[1]
    f3 = 0.5f0 * (f_ll[3] + f_rr[3]) -
         0.5f0 * λ_max * (q2_rr - q2_ll) * normal_direction[2]

    return SVector(f1, f2, f3)
end

@inline have_constant_speed(::HyperbolicDiffusionEquations2D) = True()

@inline function max_abs_speeds(eq::HyperbolicDiffusionEquations2D)
    λ = sqrt(eq.nu * eq.inv_Tr)
    return λ, λ
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::HyperbolicDiffusionEquations2D) = u

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
@inline function cons2entropy(u, equations::HyperbolicDiffusionEquations2D)
    phi, q1, q2 = u
    w1 = phi
    w2 = equations.Lr^2 * q1
    w3 = equations.Lr^2 * q2

    return SVector(w1, w2, w3)
end

# Calculate entropy for a conservative state `u` (here: same as total energy)
@inline function entropy(u, equations::HyperbolicDiffusionEquations2D)
    energy_total(u, equations)
end

# Calculate total energy for a conservative state `u`
@inline function energy_total(u, equations::HyperbolicDiffusionEquations2D)
    # energy function as found in equations (2.5.12) in the book "I Do Like CFD, Vol. 1"
    phi, q1, q2 = u
    return 0.5f0 * (phi^2 + equations.Lr^2 * (q1^2 + q2^2))
end
end # @muladd
