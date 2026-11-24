# Projection of a state onto the numerical admissible set of the ideal GLM-MHD equations,
# used by `PositivityPreservingLimiterLiuZhang`.
#
# In the ideal GLM-MHD equations of Trixi.jl, the total energy contains the energy
# `psi^2 / 2` of the divergence cleaning field `psi`, so that the internal energy is
#   rho_e = rho_e_total - |m|^2 / (2 rho) - |B|^2 / 2 - psi^2 / 2.
# Thus, the numerical admissible set { rho >= rho_floor, rho_e >= rho_e_floor } is the
# admissible set of the ideal MHD equations with the magnetic field `B` replaced by the
# vector `(B, psi)`. It is convex since `rho_e` is a concave function of the conservative
# variables (for rho > 0).
#
# The projection follows the slicing algorithm of
# - Chen Liu, Chi-Wang Shu, Xiangxiong Zhang (2026)
#   Efficient admissible set projection in optimization-based invariant-domain-preserving
#   limiters for ideal MHD
#   [arXiv: 2605.10929](https://arxiv.org/abs/2605.10929)
# For a fixed magnetic energy `beta = |z|^2` with `z = (B, psi)`, the projection decouples into
# (i)  the projection of `z` onto the sphere of radius `sqrt(beta)`, which rescales `z`
#      (Lemma 3.4 of the reference), and
# (ii) the projection of `(rho, m, rho_e_total)` onto the "Euler-like" slice
#        { rho >= rho_floor, rho_e_total - |m|^2 / (2 rho) >= rho_e_floor + beta / 2 }.
# The squared distance `d2(beta)` of the resulting state to the input state is strictly
# convex in `beta`, and its minimizer lies in the interval `[beta_low, |z|^2]` given in
# Lemma 3.7 of the reference.
#
# The implementation differs from the reference in two points.
# - The slice subproblem (ii) has three momentum components also in one and two space
#   dimensions. Since both the slice and the squared distance are invariant under rotations
#   of the momentum, the minimizer is parallel to the input momentum, and the subproblem
#   reduces to the 1D compressible Euler projection of `(rho, |m|, rho_e_total)` in
#   `admissible_projection_euler_1d.jl`.
# - Instead of minimizing `d2(beta)` with a derivative-free method, which determines the
#   minimizer only up to roughly the square root of the machine precision, the root of
#   the derivative `d2'(beta)` is computed to machine precision with `find_zero_brent`.
#   By the envelope theorem, the derivative of the squared distance `f(beta)` of the slice
#   projection is half of the Lagrange multiplier of the internal energy constraint of the
#   slice, which equals the change of the total energy in the slice projection,
#     f'(beta) = rho_e_total*(beta) - rho_e_total.
#   The derivative of the magnetic part is `1 - |z| / sqrt(beta)`. Hence,
#     d2'(beta) = (rho_e_total*(beta) - rho_e_total) + 1 - |z| / sqrt(beta)
#   is available at no additional cost from the slice projection and is strictly
#   increasing in `beta`.
#
# In one space dimension, the normal magnetic field component `B1` is constant in time
# since its flux vanishes. Thus, `B1` is kept fixed by the projection, i.e., the state is
# projected onto the (still convex) intersection of the admissible set with { B1 = const. },
# and the magnetic energy `B1^2 / 2` is added to the internal energy floor of the slice.

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Return the magnetic-type variables that may be rescaled by the projection and the
# magnetic energy of the components that are kept fixed.
@inline function magnetic_variables_free_and_fixed_energy(u, ::IdealGlmMhdEquations1D)
    # B1 is constant in 1D and kept fixed
    return SVector(u[7], u[8]), 0.5f0 * u[6]^2
end

@inline function magnetic_variables_free_and_fixed_energy(u,
                                                          ::Union{IdealGlmMhdEquations2D,
                                                                  IdealGlmMhdEquations3D})
    return SVector(u[6], u[7], u[8], u[9]), zero(eltype(u))
end

@inline function assemble_projected_state(rho, momentum, rho_e_total, magnetic_free, u,
                                          ::IdealGlmMhdEquations1D)
    return vcat(SVector(rho), momentum, SVector(rho_e_total, u[6]), magnetic_free)
end

@inline function assemble_projected_state(rho, momentum, rho_e_total, magnetic_free, u,
                                          ::Union{IdealGlmMhdEquations2D,
                                                  IdealGlmMhdEquations3D})
    return vcat(SVector(rho), momentum, SVector(rho_e_total), magnetic_free)
end

# Projection of `(rho, momentum, rho_e_total)` with three momentum components onto the slice
#   { rho >= rho_floor, rho_e_total - |momentum|^2 / (2 rho) >= rho_e_floor_slice }
# via the 1D compressible Euler projection of `(rho, |momentum|, rho_e_total)`; see the
# comment at the top of this file.
@inline function project_to_euler_slice(rho, momentum, rho_e_total,
                                        rho_floor, rho_e_floor_slice,
                                        equations_euler::CompressibleEulerEquations1D)
    momentum_norm = norm(momentum)
    u_euler = SVector(rho, momentum_norm, rho_e_total)
    u_euler_projected = project_to_admissible_set(u_euler,
                                                  (rho_floor, rho_e_floor_slice),
                                                  (density, energy_internal),
                                                  equations_euler)
    rho_projected, momentum_norm_projected, rho_e_total_projected = u_euler_projected

    # The projected scalar momentum is nonnegative since the input `momentum_norm` is
    # nonnegative (by symmetry, flipping the sign of a negative candidate would decrease the
    # distance to the input state).
    if iszero(momentum_norm)
        momentum_projected = zero(momentum)
    else
        momentum_projected = (momentum_norm_projected / momentum_norm) * momentum
    end

    return rho_projected, momentum_projected, rho_e_total_projected
end

"""
    project_to_admissible_set(cell_average, lower_bounds, variables,
                              equations::AbstractIdealGlmMhdEquations)

Project the state `cell_average` onto the numerical admissible set
`{ rho >= rho_floor, rho_e >= rho_e_floor }` of the ideal GLM-MHD equations, where
`lower_bounds = (rho_floor, rho_e_floor)` and `rho_e` is the internal energy (see
[`energy_internal`](@ref)). The projection minimizes the Euclidean distance in the
conservative variables and uses the slicing algorithm of
- Chen Liu, Chi-Wang Shu, Xiangxiong Zhang (2026)
  Efficient admissible set projection in optimization-based invariant-domain-preserving
  limiters for ideal MHD
  [arXiv: 2605.10929](https://arxiv.org/abs/2605.10929)
with the divergence cleaning field `psi` treated as an additional component of the
magnetic field. In one space dimension, the constant normal magnetic field component `B1`
is kept fixed. See the comments in the source file for details.
"""
function project_to_admissible_set(cell_average, lower_bounds, variables,
                                   equations::AbstractIdealGlmMhdEquations)
    rho_floor, rho_e_floor = lower_bounds
    u = cell_average

    if state_is_admissible(u, lower_bounds, variables, equations)
        return u
    end

    rho = u[1]
    momentum = SVector(u[2], u[3], u[4])
    rho_e_total = u[5]
    z, magnetic_energy_fixed = magnetic_variables_free_and_fixed_energy(u, equations)
    z_norm = norm(z)

    # The slice subproblems are solved with the 1D compressible Euler projection.
    equations_euler = CompressibleEulerEquations1D(equations.gamma)
    rho_e_floor_slice(beta) = rho_e_floor + magnetic_energy_fixed + 0.5f0 * beta

    if iszero(z_norm)
        rho_projected, momentum_projected, rho_e_total_projected = project_to_euler_slice(rho,
                                                                                          momentum,
                                                                                          rho_e_total,
                                                                                          rho_floor,
                                                                                          rho_e_floor_slice(zero(z_norm)),
                                                                                          equations_euler)
        return assemble_projected_state(rho_projected, momentum_projected,
                                        rho_e_total_projected, z, u, equations)
    end

    # Derivative of the squared distance `d2(beta)` of the projection onto the slice with
    # magnetic energy `beta` to the input state; see the comment at the top of this file.
    function d2_prime(beta)
        _, _, rho_e_total_beta = project_to_euler_slice(rho, momentum, rho_e_total,
                                                        rho_floor,
                                                        rho_e_floor_slice(beta),
                                                        equations_euler)
        return (rho_e_total_beta - rho_e_total) + 1 - z_norm / sqrt(beta)
    end

    # Search interval `[beta_low, beta_high]` of Lemma 3.7 of Liu, Shu, Zhang (2026),
    # where `f_0` is the squared distance of the slice projection for `beta = 0`.
    beta_high = z_norm^2
    rho_0, momentum_0, rho_e_total_0 = project_to_euler_slice(rho, momentum,
                                                              rho_e_total,
                                                              rho_floor,
                                                              rho_e_floor_slice(zero(z_norm)),
                                                              equations_euler)
    f_0 = (rho_0 - rho)^2 + sum(abs2, momentum_0 - momentum) +
          (rho_e_total_0 - rho_e_total)^2
    beta_low = (z_norm / (1 + sqrt(f_0) + 0.5f0 * beta_high))^2

    # `d2_prime` is strictly increasing. If it does not change sign in the search interval,
    # the minimizer is the corresponding endpoint; `d2_prime(beta_high) >= 0` always holds,
    # with equality iff the internal energy constraint is inactive for unchanged `z`.
    if d2_prime(beta_low) >= 0
        beta = beta_low
    elseif d2_prime(beta_high) <= 0
        beta = beta_high
    else
        beta = find_zero_brent(d2_prime, beta_low, beta_high)
    end

    rho_projected, momentum_projected, rho_e_total_projected = project_to_euler_slice(rho,
                                                                                      momentum,
                                                                                      rho_e_total,
                                                                                      rho_floor,
                                                                                      rho_e_floor_slice(beta),
                                                                                      equations_euler)
    z_projected = (sqrt(beta) / z_norm) * z

    return assemble_projected_state(rho_projected, momentum_projected,
                                    rho_e_total_projected, z_projected, u, equations)
end
end # @muladd
