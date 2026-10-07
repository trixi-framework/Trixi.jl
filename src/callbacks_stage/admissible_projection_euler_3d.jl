# See the comment at the top of admissible_projection_euler_1d.jl for a
# high-level description of the algorithm.

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function project_euler_cubic_branch!(best_dist_squared, best_u, has_candidate, u,
                                     rho_floor, rho_e_floor,
                                     primary_idx,
                                     equations::CompressibleEulerEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e_total = u
    if primary_idx == 1
        rho_v_primary, rho_v_secondary, rho_v_tertiary = rho_v1, rho_v2, rho_v3
    elseif primary_idx == 2
        rho_v_primary, rho_v_secondary, rho_v_tertiary = rho_v2, rho_v1, rho_v3
    else
        rho_v_primary, rho_v_secondary, rho_v_tertiary = rho_v3, rho_v1, rho_v2
    end
    a = 1 + (rho_v_secondary / rho_v_primary)^2 + (rho_v_tertiary / rho_v_primary)^2
    # eps = rho_floor and beta = 2*(rho_e_floor - rho_floor).
    p = 2 * rho_floor * (rho_floor + rho_e_floor - rho_e_total) / a
    q = -2 * rho_floor * rho_floor * rho_v_primary / a
    n_roots, roots = calc_depressed_cubic_roots(p, q)
    for i in 1:n_roots
        rho_v_primary_candidate = roots[i]
        if cubic_momentum_root_satisfies_kkt(rho_v_primary_candidate, rho_v_primary,
                                             rho, a,
                                             rho_floor)
            rho_e_total_candidate = rho_e_floor +
                                    a * rho_v_primary_candidate *
                                    rho_v_primary_candidate /
                                    (2 * rho_floor)
            rho_v_secondary_candidate = (rho_v_secondary / rho_v_primary) *
                                        rho_v_primary_candidate
            rho_v_tertiary_candidate = (rho_v_tertiary / rho_v_primary) *
                                       rho_v_primary_candidate
            u_candidate = if primary_idx == 1
                SVector(rho_floor, rho_v_primary_candidate,
                        rho_v_secondary_candidate, rho_v_tertiary_candidate,
                        rho_e_total_candidate)
            elseif primary_idx == 2
                SVector(rho_floor, rho_v_secondary_candidate,
                        rho_v_primary_candidate, rho_v_tertiary_candidate,
                        rho_e_total_candidate)
            else
                SVector(rho_floor, rho_v_secondary_candidate,
                        rho_v_tertiary_candidate, rho_v_primary_candidate,
                        rho_e_total_candidate)
            end
            best_dist_squared, best_u, has_candidate = update_best_candidate!(best_dist_squared,
                                                                              best_u,
                                                                              has_candidate,
                                                                              u_candidate,
                                                                              u,
                                                                              equations)
        end
    end
    return best_dist_squared, best_u, has_candidate
end

function project_euler_lambda_zero_branch!(best_dist_squared, best_u, has_candidate, u,
                                           rho_floor, rho_e_floor, arithmetic_tol,
                                           primary_idx,
                                           equations::CompressibleEulerEquations3D)
    # mu > 0 energy checks below (lambda = 0 branch): rho_candidate = 1/2(rho +/- sqrt(D_rho))
    # can suffer catastrophic cancellation when rho and sqrt(D_rho) are opposite in sign
    # and similar in magnitude; error in rho_candidate can then flip the
    # (1 - arithmetic_tol) comparison.
    rho, rho_v1, rho_v2, rho_v3, rho_e_total = u
    if primary_idx == 1
        rho_v_primary, rho_v_secondary, rho_v_tertiary = rho_v1, rho_v2, rho_v3
    elseif primary_idx == 2
        rho_v_primary, rho_v_secondary, rho_v_tertiary = rho_v2, rho_v1, rho_v3
    else
        rho_v_primary, rho_v_secondary, rho_v_tertiary = rho_v3, rho_v1, rho_v2
    end
    a = 1 + (rho_v_secondary / rho_v_primary)^2 + (rho_v_tertiary / rho_v_primary)^2
    m_sq = rho_v1 * rho_v1 + rho_v2 * rho_v2 + rho_v3 * rho_v3
    discriminant_rho = rho * rho -
                       (2 * rho * m_sq *
                        (rho_e_total - rho_e_floor) -
                        m_sq * m_sq) /
                       (2 * m_sq +
                        (rho_e_floor + rho - rho_e_total)^2)
    if discriminant_rho >= zero(discriminant_rho)
        sqrt_discriminant_rho = sqrt(discriminant_rho)
        for rho_candidate in (0.5f0 * a_minus_sqrt_b_rationalized(rho,
                                                          discriminant_rho,
                                                          sqrt_discriminant_rho),
                              0.5f0 * (rho + sqrt_discriminant_rho))
            # For rho_candidate = ½(ρ ± √Δ_ρ), the momentum discriminant reduces to
            # -8aρ_c² + 8aρρ_c + (aρv)² = 2a(ρ² - Δ_ρ) + (aρv)² (which is independent of
            # rho_candidate).
            discriminant_rho_v_primary = 2 * a * (rho^2 - discriminant_rho) +
                                         (a * rho_v_primary)^2
            # Roundoff can make discriminant_rho_v_primary slightly negative at the real-root
            # boundary; treat as zero so the >= 0 check passes.
            if discriminant_rho_v_primary < zero(discriminant_rho_v_primary) &&
               discriminant_rho_v_primary > -arithmetic_tol
                discriminant_rho_v_primary = zero(discriminant_rho_v_primary)
            end
            if rho_candidate >= rho_floor - arithmetic_tol &&
               discriminant_rho_v_primary >= zero(discriminant_rho_v_primary)
                sqrt_discriminant_rho_v_primary = sqrt(discriminant_rho_v_primary) / a
                for rho_v_primary_candidate in (0.5f0 *
                                                (rho_v_primary -
                                                 sqrt_discriminant_rho_v_primary),
                                                0.5f0 *
                                                (rho_v_primary +
                                                 sqrt_discriminant_rho_v_primary))
                    # μ > 0 sign check (λ = 0 branch); see admissible_projection_euler_1d.jl.
                    candidate_energy_internal_times_rho = rho_e_floor * rho_candidate +
                                                          0.5f0 * a *
                                                          rho_v_primary_candidate *
                                                          rho_v_primary_candidate
                    original_energy_internal_times_rho = rho_e_total * rho_candidate *
                                                         (1 - arithmetic_tol)
                    if candidate_energy_internal_times_rho >
                       original_energy_internal_times_rho
                        rho_e_total_candidate = rho_e_floor +
                                                0.5f0 * a * rho_v_primary_candidate *
                                                rho_v_primary_candidate / rho_candidate
                        rho_v_secondary_candidate = (rho_v_secondary / rho_v_primary) *
                                                    rho_v_primary_candidate
                        rho_v_tertiary_candidate = (rho_v_tertiary / rho_v_primary) *
                                                   rho_v_primary_candidate
                        u_candidate = if primary_idx == 1
                            SVector(rho_candidate, rho_v_primary_candidate,
                                    rho_v_secondary_candidate,
                                    rho_v_tertiary_candidate,
                                    rho_e_total_candidate)
                        elseif primary_idx == 2
                            SVector(rho_candidate, rho_v_secondary_candidate,
                                    rho_v_primary_candidate,
                                    rho_v_tertiary_candidate,
                                    rho_e_total_candidate)
                        else
                            SVector(rho_candidate, rho_v_secondary_candidate,
                                    rho_v_tertiary_candidate,
                                    rho_v_primary_candidate,
                                    rho_e_total_candidate)
                        end
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
    return best_dist_squared, best_u, has_candidate
end

"""
    project_to_admissible_set(cell_average, lower_bounds, variables,
                              equations::CompressibleEulerEquations3D)

Implements Appendix B.2 of
- Liu, Milesis, Shu, Zhang (2026)
  Efficient optimization-based invariant-domain-preserving limiters in solving gas dynamics equations
  [arXiv: 2510.21080](https://arxiv.org/abs/2510.21080)

Given an out-of-bounds solution state, this returns the closest point in the admissible set.
This is possible by noting that there are only a finite number of possible candidate states
that satisfy the KKT conditions. This implementation enumerates all candidates and returns
the one that is closest to the input state.

This code was translated was extrapolated from the 2D code by Prof. Chen Liu using AI tools.
"""
function project_to_admissible_set(cell_average, lower_bounds, variables,
                                   equations::CompressibleEulerEquations3D)
    rho_floor, rho_e_floor = lower_bounds
    u = cell_average
    rho, rho_v1, rho_v2, rho_v3, rho_e_total = u
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
    momentum_is_near_zero = abs(rho_v1) < arithmetic_tol &&
                            abs(rho_v2) < arithmetic_tol &&
                            abs(rho_v3) < arithmetic_tol

    # Case: mu = 0 and lambda > 0
    if density_below_floor &&
       (2 * rho_floor * rho_e_floor +
        rho_v1 * rho_v1 + rho_v2 * rho_v2 + rho_v3 * rho_v3) <=
       2 * rho_floor * rho_e_total
        u_candidate = SVector(rho_floor, rho_v1, rho_v2, rho_v3, rho_e_total)
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
            u_candidate = SVector(rho_floor, zero(RealT), zero(RealT), zero(RealT),
                                  rho_e_floor)
            best_dist_squared, best_u, has_candidate = update_best_candidate!(best_dist_squared,
                                                                              best_u,
                                                                              has_candidate,
                                                                              u_candidate,
                                                                              u,
                                                                              equations)
        end
    else
        abs_v1, abs_v2, abs_v3 = abs(rho_v1), abs(rho_v2), abs(rho_v3)
        primary_idx = if abs_v1 >= abs_v2 && abs_v1 >= abs_v3
            1
        elseif abs_v2 >= abs_v3
            2
        else
            3
        end
        best_dist_squared, best_u, has_candidate = project_euler_cubic_branch!(best_dist_squared,
                                                                               best_u,
                                                                               has_candidate,
                                                                               u,
                                                                               rho_floor,
                                                                               rho_e_floor,
                                                                               primary_idx,
                                                                               equations)
    end

    # Case: mu > 0 and lambda = 0
    if momentum_is_near_zero
        if !density_below_floor && rho_e_total < rho_e_floor
            u_candidate = SVector(rho, zero(RealT), zero(RealT), zero(RealT),
                                  rho_e_floor)
            best_dist_squared, best_u, has_candidate = update_best_candidate!(best_dist_squared,
                                                                              best_u,
                                                                              has_candidate,
                                                                              u_candidate,
                                                                              u,
                                                                              equations)
        end
    else
        abs_v1, abs_v2, abs_v3 = abs(rho_v1), abs(rho_v2), abs(rho_v3)
        primary_idx = if abs_v1 >= abs_v2 && abs_v1 >= abs_v3
            1
        elseif abs_v2 >= abs_v3
            2
        else
            3
        end
        best_dist_squared, best_u, has_candidate = project_euler_lambda_zero_branch!(best_dist_squared,
                                                                                     best_u,
                                                                                     has_candidate,
                                                                                     u,
                                                                                     rho_floor,
                                                                                     rho_e_floor,
                                                                                     arithmetic_tol,
                                                                                     primary_idx,
                                                                                     equations)
    end

    if !has_candidate
        error("Failed to find projection onto Euler admissible set for state ", u,
              " with rho = ", rho, " and rho_e = ",
              rho_e_total -
              0.5f0 * (rho_v1 * rho_v1 + rho_v2 * rho_v2 + rho_v3 * rho_v3) / rho,
              " and rho_floor = ", rho_floor, " and rho_e_floor = ", rho_e_floor, ".")
    end

    return best_u
end
end # @muladd
