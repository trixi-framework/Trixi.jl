# Used in the mu > 0, lambda > 0 branch of Appendix B.2 of Liu, Milesis, Shu, Zhang (2026).
@inline function cubic_momentum_root_satisfies_kkt(rho_v1, rho_v1_orig, rho_orig, a,
                                                   rho_floor, rho_e_floor)
    momentum_sign_complementarity = (rho_v1 > zero(rho_v1) && rho_v1_orig > rho_v1) ||
                                    (rho_v1 < zero(rho_v1) && rho_v1_orig < rho_v1)
    satisfies_energy_internal_constraint_at_rho_floor = 2 * rho_floor * rho_orig +
                                                        a * rho_v1 *
                                                        (rho_v1_orig - rho_v1) <
                                                        2 * rho_floor * rho_e_floor
    return momentum_sign_complementarity &&
           satisfies_energy_internal_constraint_at_rho_floor
end

function project_euler_cubic_branch!(best_dist_squared, best_u, has_candidate, u,
                                     rho_floor, rho_e_floor,
                                     use_v1_as_primary,
                                     equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e_total = u
    rho_v_primary, rho_v_secondary = use_v1_as_primary ? (rho_v1, rho_v2) : (rho_v2, rho_v1)
    a = 1 + (rho_v_secondary / rho_v_primary)^2
    p = rho_floor * (4 * rho_e_floor - 2 * rho_e_total) / a
    q = -2 * rho_floor * rho_e_floor * rho_v_primary / a
    n_roots, roots = calc_depressed_cubic_roots(p, q)
    for i in 1:n_roots
        rho_v_primary_candidate = roots[i]
        if cubic_momentum_root_satisfies_kkt(rho_v_primary_candidate, rho_v_primary, rho, a,
                                             rho_floor, rho_e_floor)
            rho_e_total_candidate = rho_e_floor +
                                    a * rho_v_primary_candidate * rho_v_primary_candidate /
                                    (2 * rho_floor)
            rho_v_secondary_candidate = (rho_v_secondary / rho_v_primary) *
                                        rho_v_primary_candidate
            u_candidate = if use_v1_as_primary
                SVector(rho_floor, rho_v_primary_candidate,
                        rho_v_secondary_candidate, rho_e_total_candidate)
            else
                SVector(rho_floor, rho_v_secondary_candidate,
                        rho_v_primary_candidate, rho_e_total_candidate)
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
                                           use_v1_as_primary,
                                           equations::CompressibleEulerEquations2D)
    # mu > 0 energy checks below (lambda = 0 branch): rho_candidate = 1/2(rho +/- sqrt(D_rho))
    # can suffer catastrophic cancellation when rho and sqrt(D_rho) are opposite in sign
    # and similar in magnitude; error in rho_candidate can then flip the
    # (1 - arithmetic_tol) comparison.
    rho, rho_v1, rho_v2, rho_e_total = u
    rho_v_primary, rho_v_secondary = use_v1_as_primary ? (rho_v1, rho_v2) : (rho_v2, rho_v1)
    a = 1 + (rho_v_secondary / rho_v_primary)^2
    discriminant_rho = rho * rho -
                       (2 * rho * rho_v_primary * rho_v_primary *
                        (rho_e_total - rho_e_floor) -
                        a * rho_v_primary^4) /
                       (2 * rho_v_primary * rho_v_primary +
                        (rho_e_floor + rho - rho_e_total)^2 / a)
    if discriminant_rho >= zero(discriminant_rho)
        sqrt_discriminant_rho = sqrt(discriminant_rho)
        for rho_candidate in (0.5 * (rho - sqrt_discriminant_rho),
                              0.5 * (rho + sqrt_discriminant_rho))
            discriminant_rho_v_primary = -8 * a * rho_candidate * rho_candidate +
                                         8 * a * rho * rho_candidate +
                                         (a * rho_v_primary)^2
            # Roundoff can make discriminant_rho_v_primary slightly negative at the real-root
            # boundary; treat as zero so the >= 0 check passes and sqrt is valid.
            if discriminant_rho_v_primary < zero(discriminant_rho_v_primary) &&
               discriminant_rho_v_primary > -arithmetic_tol
                discriminant_rho_v_primary = zero(discriminant_rho_v_primary)
            end
            if rho_candidate >= rho_floor - arithmetic_tol &&
               discriminant_rho_v_primary >= zero(discriminant_rho_v_primary)
                sqrt_discriminant_rho_v_primary = sqrt(discriminant_rho_v_primary) / a
                for rho_v_primary_candidate in (0.5 *
                                                (rho_v_primary -
                                                 sqrt_discriminant_rho_v_primary),
                                                0.5 *
                                                (rho_v_primary +
                                                 sqrt_discriminant_rho_v_primary))
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
                        u_candidate = if use_v1_as_primary
                            SVector(rho_candidate, rho_v_primary_candidate,
                                    rho_v_secondary_candidate,
                                    rho_e_total_candidate)
                        else
                            SVector(rho_candidate, rho_v_secondary_candidate,
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
                              equations::CompressibleEulerEquations2D)

Implements Appendix B.2 of
- Liu, Milesis, Shu, Zhang (2026)
  Efficient optimization-based invariant-domain-preserving limiters in solving gas dynamics equations
  [arXiv: 2510.21080](https://arxiv.org/abs/2510.21080)

Given an out-of-bounds solution state, this returns the closest point in the admissible set.
This is possible by noting that there are only a finite number of possible candidate states
that satisfy the KKT conditions. This implementation enumerates all candidates and returns
the one that is closest to the input state.

This code was translated from code written by Prof. Chen Liu using AI tools.
"""
function project_to_admissible_set(cell_average, lower_bounds, variables,
                                   equations::CompressibleEulerEquations2D)
    rho_floor, rho_e_floor = lower_bounds
    u = cell_average
    rho, rho_v1, rho_v2, rho_e_total = u
    RealT = typeof(rho)
    arithmetic_tol = euler_arithmetic_tol(rho_floor, rho_e_floor)
    @assert arithmetic_tol<minimum(lower_bounds) "arithmetic_tol must be smaller than the tolerance of the numerical admissible set"

    if state_is_admissible(u, lower_bounds, variables, equations)
        return u
    end

    best_dist_squared = typemax(RealT)
    best_u = zero(typeof(u))
    has_candidate = false

    density_below_floor = rho < rho_floor
    momentum_is_near_zero = abs(rho_v1) < arithmetic_tol && abs(rho_v2) < arithmetic_tol

    # Case: mu = 0 and lambda > 0
    if density_below_floor &&
       (2 * rho_floor * rho_e_floor + rho_v1 * rho_v1 + rho_v2 * rho_v2) <=
       2 * rho_floor * rho_e_total
        u_candidate = SVector(rho_floor, rho_v1, rho_v2, rho_e_total)
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
            u_candidate = SVector(rho_floor, zero(RealT), zero(RealT), rho_e_floor)
            best_dist_squared, best_u, has_candidate = update_best_candidate!(best_dist_squared,
                                                                              best_u,
                                                                              has_candidate,
                                                                              u_candidate,
                                                                              u,
                                                                              equations)
        end
    else
        use_v1_as_primary = abs(rho_v1) >= abs(rho_v2)
        best_dist_squared, best_u, has_candidate = project_euler_cubic_branch!(best_dist_squared,
                                                                               best_u,
                                                                               has_candidate,
                                                                               u,
                                                                               rho_floor,
                                                                               rho_e_floor,
                                                                               use_v1_as_primary,
                                                                               equations)
    end

    # Case: mu > 0 and lambda = 0
    if momentum_is_near_zero
        if !density_below_floor && rho_e_total < rho_e_floor
            u_candidate = SVector(rho, zero(RealT), zero(RealT), rho_e_floor)
            best_dist_squared, best_u, has_candidate = update_best_candidate!(best_dist_squared,
                                                                              best_u,
                                                                              has_candidate,
                                                                              u_candidate,
                                                                              u,
                                                                              equations)
        end
    else
        use_v1_as_primary = abs(rho_v1) >= abs(rho_v2)
        best_dist_squared, best_u, has_candidate = project_euler_lambda_zero_branch!(best_dist_squared,
                                                                                     best_u,
                                                                                     has_candidate,
                                                                                     u,
                                                                                     rho_floor,
                                                                                     rho_e_floor,
                                                                                     arithmetic_tol,
                                                                                     use_v1_as_primary,
                                                                                     equations)
    end

    if !has_candidate
        error("Failed to find projection onto Euler admissible set for state ", u,
              " with rho_floor = ", rho_floor, " and rho_e_floor = ", rho_e_floor, ".")
    end

    return best_u
end
