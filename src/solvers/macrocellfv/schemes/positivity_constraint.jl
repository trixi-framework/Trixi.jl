# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Positivity constraint of the `MacroCellFV` solver: the admissible set of the
# equations and the positivity limiter of the reconstructed face states.

"""
    ReconstructionPositivity(reconstruction; epsilon = 1.0e-12, omega = 0.25)

Positivity limiter for the face states of a `reconstruction`, e.g.,
[`ReconstructionWENO5Z`](@ref) or [`ReconstructionMUSCL`](@ref), in the spirit of
- Xiangxiong Zhang and Chi-Wang Shu (2010)
  On positivity-preserving high order discontinuous Galerkin schemes for compressible
  Euler equations on rectangular meshes
  [DOI: 10.1016/j.jcp.2010.08.016](https://doi.org/10.1016/j.jcp.2010.08.016)
Each face state `w` of a subcell (in the reconstruction variables) is replaced by
`w̄ + θ (w - w̄)` with the subcell value `w̄` and the largest `θ ∈ [0, 1]` (found by
bisection) for which the quantities that must stay positive (for the compressible
Euler equations the density and the pressure, for the multicomponent Euler equations
the pressure and the partial densities) of the face state, converted to conservative variables, are at least
`epsilon` times those of the subcell value. Then, both face states of the subcell are
scaled together towards `w̄` such that the subcell value is a convex combination
`ū = omega (u_minus + u_plus) + (1 - 2 omega) u*` of the face states and an admissible
interior state `u*`, which bounds the outflow through the faces by the content of the
subcell for time steps with `λ α <= omega` per direction (Zhang and Shu, 2010). Since
the admissibility is checked on the conservative states, the limiter works with any
reconstruction variables; with `cons2recon = cons2cons` (conservative variables),
WENO keeps its order of accuracy for smooth solutions, while the reconstruction of
primitive variables from the subcell averages is only second order accurate (use a
moment inversion to point values, e.g., [`MomentInversionWENO`](@ref), for higher order
with primitive reconstruction variables).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionPositivity{Reconstruction, RealT}
    reconstruction::Reconstruction
    epsilon::RealT
    omega::RealT
end

function ReconstructionPositivity(reconstruction; epsilon = 1.0e-12, omega = 0.25)
    0 < omega <= 0.5 || throw(ArgumentError("`omega` must be in (0, 1/2]"))
    epsilon, omega = promote(epsilon, omega)
    return ReconstructionPositivity(reconstruction, epsilon, omega)
end

function Base.show(io::IO, reconstruction::ReconstructionPositivity)
    print(io, "ReconstructionPositivity(", reconstruction.reconstruction, ")")
end

@inline nhalo(reconstruction::ReconstructionPositivity) = nhalo(reconstruction.reconstruction)

# Quantities that must stay positive (see `ReconstructionPositivity`), as functions of
# the conservative variables (none for general equations)
@inline positivity_quantities(u, equations) = SVector{0, eltype(u)}()

@inline function positivity_quantities(u, equations::AbstractCompressibleEulerEquations)
    return SVector(density(u, equations), pressure(u, equations))
end

@inline function positivity_quantities(u,
                                       equations::AbstractCompressibleEulerMulticomponentEquations)
    # (the partial densities are the last variables)
    offset = nvariables(equations) - ncomponents(equations)
    return vcat(SVector(pressure(u, equations)),
                SVector{ncomponents(equations)}(u[i + offset]
                                                for i in eachcomponent(equations)))
end

# `true` if all quantities of `positivity_quantities` of the conservative state `u` are at
# least `bounds` (`false` for NaN values)
@inline function positivity_admissible(u, bounds, equations)
    q = positivity_quantities(u, equations)
    admissible = true
    for i in eachindex(bounds)
        admissible &= q[i] >= bounds[i]
    end
    return admissible
end

# Largest `θ ∈ [0, 1]` such that the quantities `quantities(θ)` (see
# `positivity_quantities`) are at least `bounds`, for admissible `quantities(0)`; 1
# without further evaluations if `quantities(1)` is admissible. First guess: the
# linear interpolation of the quantities between `θ = 0` and `θ = 1`, which is
# exact for quantities that are linear in `θ` (e.g., the partial densities) and
# admissible for concave ones (e.g., the pressure along a line in conservative
# variables). Otherwise (or for NaN values), 12 bisection steps below the guess; the
# result is always admissible.
@inline function positivity_theta(quantities::Quantities, bounds) where {Quantities}
    RealT = eltype(bounds)
    q_1 = quantities(one(RealT))
    all(q_1 .>= bounds) && return one(RealT)
    q_0 = quantities(zero(RealT))
    theta_hi = one(RealT)
    for i in eachindex(bounds)
        if q_1[i] < bounds[i]
            # `q_0[i] >= bounds[i] > q_1[i]` (the comparison skips NaN values)
            theta_i = (q_0[i] - bounds[i]) / (q_0[i] - q_1[i])
            theta_hi = theta_i < theta_hi ? theta_i : theta_hi
        end
    end
    # (the guess might not be admissible for quantities that are not concave in `θ`)
    all(quantities(theta_hi) .>= bounds) && return theta_hi
    theta_lo = zero(RealT)
    for _ in 1:12
        theta = (theta_lo + theta_hi) / 2
        if all(quantities(theta) .>= bounds)
            theta_lo = theta
        else
            theta_hi = theta
        end
    end
    return theta_lo
end

@inline function reconstruct_cell(reconstruction::ReconstructionPositivity, stencil,
                                  equations, recon2cons)
    w_minus, w_plus = reconstruct_cell(reconstruction.reconstruction, stencil)
    # (the subcell average, also for reconstructions from point values)
    w_center = reconstruction_center(reconstruction.reconstruction, stencil)
    RealT = eltype(w_center)
    epsilon = convert(RealT, reconstruction.epsilon)
    omega = convert(RealT, reconstruction.omega)
    u_center = recon2cons(w_center, equations)
    bounds = epsilon * positivity_quantities(u_center, equations)
    # Inadmissible subcell values cannot be helped here
    positivity_admissible(u_center, bounds, equations) || return w_center, w_center

    # Each face state separately
    theta_minus = positivity_theta_face(w_minus, w_center, bounds, equations,
                                        recon2cons)
    theta_plus = positivity_theta_face(w_plus, w_center, bounds, equations, recon2cons)
    w_minus_limited = w_center + theta_minus * (w_minus - w_center)
    w_plus_limited = w_center + theta_plus * (w_plus - w_center)

    # Both face states together: the subcell value must be a convex combination
    # `ū = ω (u_minus + u_plus) + (1 - 2ω) u*` with an admissible interior state `u*`
    # (Zhang and Shu, 2010), which bounds the outflow through both faces by the content
    # of the subcell
    theta = positivity_theta(PositivityInterior(w_minus_limited, w_plus_limited,
                                                w_center, u_center, omega, equations,
                                                recon2cons),
                             bounds)
    # (unlimited states are returned unchanged, without round-off errors)
    theta == 1 && theta_minus == 1 && theta_plus == 1 && return w_minus, w_plus
    return (w_center + theta * (w_minus_limited - w_center),
            w_center + theta * (w_plus_limited - w_center))
end

# Protected quantities of the face state `w_center + θ (w - w_center)` (function object
# instead of a closure, so that no captured variables are boxed)
struct PositivityFace{W, Equations, Recon2Cons}
    w::W
    w_center::W
    equations::Equations
    recon2cons::Recon2Cons
end

@inline function (f::PositivityFace)(theta)
    @unpack w, w_center, equations, recon2cons = f
    return positivity_quantities(recon2cons(w_center + theta * (w - w_center),
                                            equations),
                                 equations)
end

@inline function positivity_theta_face(w, w_center, bounds, equations, recon2cons)
    return positivity_theta(PositivityFace(w, w_center, equations, recon2cons),
                            bounds)
end

# Protected quantities of the interior state of the convex decomposition of the subcell
# value for the face states scaled by `θ` towards `w_center`
struct PositivityInterior{W, U, RealT, Equations, Recon2Cons}
    w_minus::W
    w_plus::W
    w_center::W
    u_center::U
    omega::RealT
    equations::Equations
    recon2cons::Recon2Cons
end

@inline function (f::PositivityInterior)(theta)
    @unpack w_minus, w_plus, w_center, u_center, omega, equations, recon2cons = f
    u_minus = recon2cons(w_center + theta * (w_minus - w_center), equations)
    u_plus = recon2cons(w_center + theta * (w_plus - w_center), equations)
    u_star = (u_center - omega * (u_minus + u_plus)) / (1 - 2 * omega)
    return positivity_quantities(u_star, equations)
end
end # @muladd
