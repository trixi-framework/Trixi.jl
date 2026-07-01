# The main algorithm in this file is `project_to_admissible_set`, which is based on the 
# algorithm in Appendix B.2 of Liu, Milesis, Shu, Zhang (2026). `project_to_admissible_set`
# takes an arbitrary state and returns the closest state in the admissible set (e.g., which 
# satisfies rho >= rho_floor and rho_e >= rho_e_floor). If a state is admissible, the 
# projection just returns the state itself. 
#
# The projection is done by minimizing the ℓ² distance to the input state subject to admissibility 
# constraints. The minimizer is unique and can be determined by forming the Lagrangian and using 
# the KKT conditions. The algorithm itself enumerates all possible candidate states that satisfy 
# the KKT conditions and returns the one that has the smallest ℓ² distance to the input state. 
# 
# The candidate states correspond to mu = 0 or mu > 0, and lambda = 0 or lambda > 0, where 
# mu and lambda are Lagrange multipliers corresponding to the two admissibility constraints. 
# This results in 4 cases, each generating 1 or more candidate states:
#   • μ = 0, λ = 0 — no candidate; return u if admissible
#   • μ = 0, λ > 0  — up to 1 candidate (ρ pinned to ρ_floor)
#   • μ > 0, λ > 0  — up to 1 (momentum ≈ 0) or ≤ 3 (cubic in momentum) candidates
#   • μ > 0, λ = 0  — up to 1 (momentum ≈ 0) or ≤ 4 (ρ/momentum root pairs) candidates
# Branches are separate `if` blocks and may all contribute candidates. The current best candidate 
# is updated by `update_best_candidate!`, and after all cases are considered, a candidate 
# is returned if one was found.
#
# If no candidate was found, we throw an error. In the examples tested, this has typically 
# corresponded to either NaN input values or extreme catastrophic cancellation (e.g., if 
# internal energy is extremely small but kinetic and total energy are very large), and has 
# been due to issues independent of the projection algorithm.  

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# for compressible Euler, we introduce a small tolerance close to machine 
# precision to relax constraints. This is intended to account for roundoff 
# in intermediate calculations within project_to_admissible_set. 
@inline function euler_arithmetic_tol(rho_floor, rho_e_floor)
    T = promote_type(typeof(rho_floor), typeof(rho_e_floor))
    return 10 * eps(T)
end

# Return (best_dist_squared, best_u, has_candidate) updated when u_candidate 
# is closer to u than the current best; otherwise return the current best.
@inline function update_best_candidate!(best_dist_squared, best_u,
                                        has_candidate,
                                        u_candidate, u,
                                        equations::Union{CompressibleEulerEquations1D,
                                                         CompressibleEulerEquations2D})
    dist_squared = sum(abs2, u_candidate - u)

    # if the new candidate is closer than the current best candidate (or if there is no 
    # current best candidate), return the new candidate state. 
    if !has_candidate || dist_squared < best_dist_squared
        return dist_squared, u_candidate, true
    end
    return best_dist_squared, best_u, has_candidate
end

# Used in the μ > 0, λ > 0 branch of Appendix B.2 of Liu, Milesis, Shu, Zhang (2026).
# Here, rho_v1 comes from the solution of a cubic equation. 
@inline function cubic_momentum_root_satisfies_kkt(rho_v1, rho_v1_orig, rho_orig,
                                                   rho_floor, rho_e_floor)
    momentum_sign_complementarity = (rho_v1 > zero(rho_v1) && rho_v1_orig > rho_v1) ||
                                    (rho_v1 < zero(rho_v1) && rho_v1_orig < rho_v1)
    # Internal energy admissibility at ρ = ρ_floor (density pinned to floor in this branch).
    satisfies_energy_internal_constraint_at_rho_floor = 2 * rho_floor * rho_orig +
                                                        rho_v1 *
                                                        (rho_v1_orig - rho_v1) <
                                                        2 * rho_floor * rho_e_floor
    return momentum_sign_complementarity &&
           satisfies_energy_internal_constraint_at_rho_floor
end

# Real roots of m^3 + p*m + q = 0. Returns (n_roots, roots::SVector{3,T}).
# Note that roots[n_roots+1:end] are not accessed and are simply set to zero.
function calc_depressed_cubic_roots(p, q)
    delta = 4 * p^3 + 27 * q^2
    T = typeof(delta)
    n_roots = 0
    if delta > zero(T)
        Y1 = 1.5f0 * (9 * q + sqrt(3 * delta))
        Y2 = 1.5f0 * (9 * q - sqrt(3 * delta))
        n_roots = 1
        root_1 = -(cbrt(Y1) + cbrt(Y2)) / 3
        root_2 = zero(T) # not used
        root_3 = zero(T) # not used
    elseif iszero(delta) && (!iszero(p) || !iszero(q))
        n_roots = 2
        root_1 = 3 * q / p
        root_2 = -1.5f0 * q / p
        root_3 = zero(T) # not used
    elseif delta < zero(T)
        n_roots = 3
        sqrt_3 = sqrt(3)
        sqrt_mp_3 = sqrt(-p / 3)
        theta = acos(-1.5f0 * sqrt_mp_3 * q / p)
        s, c = sincos(theta / 3)

        root_1 = -2 * sqrt_mp_3 * c
        root_2 = sqrt_mp_3 * (c + sqrt(3) * s)
        root_3 = sqrt_mp_3 * (c - sqrt(3) * s)
    end
    return n_roots, (root_1, root_2, root_3)
end

"""
    project_to_admissible_set(cell_average, lower_bounds, variables,
                              equations::CompressibleEulerEquations1D)

Implements Appendix B.2 of
- Liu, Milesis, Shu, Zhang (2026)
  Efficient optimization-based invariant-domain-preserving limiters in solving gas dynamics equations
  [arXiv: 2510.21080](https://arxiv.org/abs/2510.21080)

Given an out-of-bounds solution state, this returns the closest point in the admissible set.
This is possible by noting that there are only a finite number of possible candidate states
that satisfy the KKT conditions. This implementation enumerates all candidates and returns 
the one that is closest to the input state. 

This code was translated in part using AI tools from private code shared by Prof. Chen Liu. 
"""
function project_to_admissible_set(cell_average, lower_bounds, variables,
                                   equations::CompressibleEulerEquations1D)
    rho_floor, rho_e_floor = lower_bounds
    u = cell_average
    rho, rho_v1, rho_e_total = u
    arithmetic_tol = euler_arithmetic_tol(rho_floor, rho_e_floor)
    RealT = typeof(arithmetic_tol)
    @assert arithmetic_tol<minimum(lower_bounds) "arithmetic_tol must be smaller than the tolerance of the numerical admissible set"

    if state_is_admissible(u, lower_bounds, variables, equations)
        return u
    end

    best_dist_squared = typemax(RealT)
    best_u = zero(typeof(u))
    has_candidate = false

    density_below_floor = rho < rho_floor
    momentum_is_near_zero = abs(rho_v1) < arithmetic_tol

    # Case: mu = 0 and lambda > 0: internal-energy constraint is active.
    if density_below_floor &&
       (2 * rho_floor * rho_e_floor + rho_v1 * rho_v1) <= 2 * rho_floor * rho_e_total
        u_candidate = SVector(rho_floor, rho_v1, rho_e_total)
        best_dist_squared, best_u, has_candidate = update_best_candidate!(best_dist_squared,
                                                                          best_u,
                                                                          has_candidate,
                                                                          u_candidate,
                                                                          u,
                                                                          equations)
    end

    # Case: mu > 0 and lambda > 0
    if momentum_is_near_zero
        if density_below_floor && rho_e_total < rho_e_floor
            u_candidate = SVector(rho_floor, zero(RealT), rho_e_floor)
            best_dist_squared, best_u, has_candidate = update_best_candidate!(best_dist_squared,
                                                                              best_u,
                                                                              has_candidate,
                                                                              u_candidate,
                                                                              u,
                                                                              equations)
        end
    else
        p = 2 * rho_floor * (2 * rho_e_floor - rho_e_total)
        q = -2 * rho_floor * rho_floor * rho_v1
        n_roots, roots = calc_depressed_cubic_roots(p, q)
        for i in 1:n_roots
            rho_v1_candidate = roots[i]
            if cubic_momentum_root_satisfies_kkt(rho_v1_candidate, rho_v1, rho,
                                                 rho_floor, rho_e_floor)
                rho_e_total_candidate = rho_e_floor +
                                        rho_v1_candidate * rho_v1_candidate /
                                        (2 * rho_floor)
                u_candidate = SVector(rho_floor, rho_v1_candidate,
                                      rho_e_total_candidate)
                best_dist_squared, best_u, has_candidate = update_best_candidate!(best_dist_squared,
                                                                                  best_u,
                                                                                  has_candidate,
                                                                                  u_candidate,
                                                                                  u,
                                                                                  equations)
            end
        end
    end

    # Case: mu > 0 and lambda = 0
    if momentum_is_near_zero
        if !density_below_floor && rho_e_total < rho_e_floor
            u_candidate = SVector(rho, zero(RealT), rho_e_floor)
            best_dist_squared, best_u, has_candidate = update_best_candidate!(best_dist_squared,
                                                                              best_u,
                                                                              has_candidate,
                                                                              u_candidate,
                                                                              u,
                                                                              equations)
        end
    else
        discriminant_rho = rho * rho -
                           (2 * rho * rho_v1 * rho_v1 * (rho_e_total - rho_e_floor) -
                            rho_v1^4) /
                           (2 * rho_v1 * rho_v1 + (rho_e_floor + rho - rho_e_total)^2)
        has_real_rho_candidates = discriminant_rho >= zero(discriminant_rho)
        if has_real_rho_candidates
            sqrt_discriminant_rho = sqrt(discriminant_rho)
            for rho_candidate in (0.5f0 * (rho - sqrt_discriminant_rho),
                                  0.5f0 * (rho + sqrt_discriminant_rho))
                discriminant_momentum = -8 * rho_candidate * rho_candidate +
                                        8 * rho * rho_candidate + rho_v1 * rho_v1
                # Roundoff can make discriminant_momentum slightly negative at the real-root
                # boundary; treat as zero so the >= 0 check passes and sqrt is valid.
                if discriminant_momentum < zero(discriminant_momentum) &&
                   discriminant_momentum > -arithmetic_tol
                    discriminant_momentum = zero(discriminant_momentum)
                end
                candidate_density_satisfies_floor = rho_candidate >=
                                                    rho_floor - arithmetic_tol
                has_real_momentum_candidates = discriminant_momentum >=
                                               zero(discriminant_momentum)
                if candidate_density_satisfies_floor && has_real_momentum_candidates
                    sqrt_discriminant_momentum = sqrt(discriminant_momentum)
                    for rho_v1_candidate in (0.5f0 *
                                             (rho_v1 - sqrt_discriminant_momentum),
                                             0.5f0 *
                                             (rho_v1 + sqrt_discriminant_momentum))
                        # μ > 0 sign check (λ = 0 branch): candidate internal energy must
                        # exceed the original. ρ_candidate = ½(ρ ± √Δ_ρ) can suffer
                        # catastrophic cancellation when ρ and √Δ_ρ are opposite in sign
                        # and similar in magnitude; the resulting error in ρ_candidate can
                        # flip this comparison. Remedies: relax (1 - arithmetic_tol), e.g. to
                        # sqrt(eps(RealT)), or detect cancellation and evaluate ρ_candidate
                        # via a stable formula.
                        candidate_energy_internal_times_rho = rho_e_floor *
                                                              rho_candidate +
                                                              0.5f0 * rho_v1_candidate *
                                                              rho_v1_candidate
                        original_energy_internal_times_rho = rho_e_total *
                                                             rho_candidate *
                                                             (1 - arithmetic_tol)

                        if candidate_energy_internal_times_rho >
                           original_energy_internal_times_rho
                            rho_e_total_candidate = rho_e_floor +
                                                    0.5f0 * rho_v1_candidate *
                                                    rho_v1_candidate / rho_candidate
                            u_candidate = SVector(rho_candidate, rho_v1_candidate,
                                                  rho_e_total_candidate)
                            best_dist_squared, best_u, has_candidate = update_best_candidate!(best_dist_squared,
                                                                                              best_u,
                                                                                              has_candidate,
                                                                                              u_candidate,
                                                                                              u,
                                                                                              equations)
                        end
                    end
                end
            end
        end
    end

    if !has_candidate
        error("Failed to find projection onto Euler admissible set for state ", u,
              " with rho = ", rho, " and rho_e = ",
              rho_e_total - 0.5f0 * rho_v1 * rho_v1 / rho,
              " and rho_floor = ", rho_floor, " and rho_e_floor = ", rho_e_floor, ".")
    end

    return best_u
end
end # @muladd
