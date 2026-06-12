@inline function project_to_admissible_set(cell_average, lower_bounds, variables,
                                           equations::CompressibleEulerEquations2D)
    rho_floor, rho_e_floor = lower_bounds
    return project_euler_2d_to_admissible_set(cell_average, rho_floor, rho_e_floor,
                                              equations)
end

@inline function state_is_admissible(u, thresholds, arithmetic_tol,
                                     equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e_total = u
    rho_floor, rho_e_floor = thresholds
    return rho >= rho_floor * (1 - arithmetic_tol) &&
           rho_v1 * rho_v1 + rho_v2 * rho_v2 + 2 * rho_e_floor * rho <=
           2 * rho * rho_e_total * (1 + arithmetic_tol)
end

@inline function projection_distance_squared_2d(u_candidate, u)
    return sum(abs2, u_candidate - u)
end

# Return (best_dist_squared, best_u, has_candidate) updated when u_candidate is closer to u
# than the current best; otherwise return the inputs unchanged.
@inline function update_best_candidate_2d!(best_dist_squared, best_u, has_candidate,
                                           u_candidate, u)
    dist_squared = projection_distance_squared_2d(u_candidate, u)
    if !has_candidate || dist_squared < best_dist_squared
        return dist_squared, u_candidate, true
    end
    return best_dist_squared, best_u, has_candidate
end

# Appendix B.2 (μ > 0, λ > 0 branch): filters depressed-cubic momentum roots that fail the
# KKT sign condition or the active energy constraint at ρ = ρ_floor.
@inline function cubic_momentum_constraint_satisfied_2d(rho_v1, rho_v1_orig, rho_orig, a,
                                                        rho_floor, rho_e_floor)
    return ((rho_v1 > zero(rho_v1) && rho_v1_orig > rho_v1) ||
            (rho_v1 < zero(rho_v1) && rho_v1_orig < rho_v1)) &&
           2 * rho_floor * rho_orig + a * rho_v1 * (rho_v1_orig - rho_v1) <
           2 * rho_floor * rho_e_floor
end

function project_euler_2d_cubic_branch!(best_dist_squared, best_u, has_candidate, u,
                                        rho_floor, rho_e_floor, arithmetic_tol,
                                        use_v1_as_primary)
    rho, rho_v1, rho_v2, rho_e_total = u
    if use_v1_as_primary
        a = 1 + (rho_v2 / rho_v1)^2
        p = rho_floor * (4 * rho_e_floor - 2 * rho_e_total) / a
        q = -2 * rho_floor * rho_e_floor * rho_v1 / a
        n_roots, roots = calc_depressed_cubic_roots(p, q)
        for i in 1:n_roots
            rho_v1_c = roots[i]
            if cubic_momentum_constraint_satisfied_2d(rho_v1_c, rho_v1, rho, a, rho_floor,
                                                      rho_e_floor)
                rho_e_total_c = rho_e_floor + a * rho_v1_c * rho_v1_c / (2 * rho_floor)
                rho_v2_c = (rho_v2 / rho_v1) * rho_v1_c
                u_candidate = SVector(rho_floor, rho_v1_c, rho_v2_c, rho_e_total_c)
                best_dist_squared, best_u, has_candidate = update_best_candidate_2d!(best_dist_squared,
                                                                                      best_u,
                                                                                      has_candidate,
                                                                                      u_candidate,
                                                                                      u)
            end
        end
    else
        a = 1 + (rho_v1 / rho_v2)^2
        p = rho_floor * (4 * rho_e_floor - 2 * rho_e_total) / a
        q = -2 * rho_floor * rho_e_floor * rho_v2 / a
        n_roots, roots = calc_depressed_cubic_roots(p, q)
        for i in 1:n_roots
            rho_v2_c = roots[i]
            if cubic_momentum_constraint_satisfied_2d(rho_v2_c, rho_v2, rho, a, rho_floor,
                                                      rho_e_floor)
                rho_e_total_c = rho_e_floor + a * rho_v2_c * rho_v2_c / (2 * rho_floor)
                rho_v1_c = (rho_v1 / rho_v2) * rho_v2_c
                u_candidate = SVector(rho_floor, rho_v1_c, rho_v2_c, rho_e_total_c)
                best_dist_squared, best_u, has_candidate = update_best_candidate_2d!(best_dist_squared,
                                                                                      best_u,
                                                                                      has_candidate,
                                                                                      u_candidate,
                                                                                      u)
            end
        end
    end
    return best_dist_squared, best_u, has_candidate
end

function project_euler_2d_lambda_zero_branch!(best_dist_squared, best_u, has_candidate, u,
                                              rho_floor, rho_e_floor, arithmetic_tol,
                                              use_v1_as_primary)
    # μ > 0 energy checks below (λ = 0 branch): ρ_c = ½(ρ ± √Δ_ρ) can suffer catastrophic
    # cancellation when ρ and √Δ_ρ are opposite in sign and similar in magnitude; error in ρ_c
    # can then flip the (1 - arithmetic_tol) comparison. Remedies: relax that factor, e.g. to
    # sqrt(eps(RealT)), or detect cancellation and evaluate ρ_c via a stable formula.
    rho, rho_v1, rho_v2, rho_e_total = u
    if use_v1_as_primary
        a = 1 + (rho_v2 / rho_v1)^2
        delta_rho = rho * rho -
                    (2 * rho * rho_v1 * rho_v1 * (rho_e_total - rho_e_floor) - a * rho_v1^4) /
                    (2 * rho_v1 * rho_v1 + (rho_e_floor + rho - rho_e_total)^2 / a)
        if delta_rho >= zero(delta_rho)
            for rho_c in (0.5 * (rho - sqrt(delta_rho)), 0.5 * (rho + sqrt(delta_rho)))
                delta_rho_v1 = -8 * a * rho_c * rho_c + 8 * a * rho * rho_c + (a * rho_v1)^2
                # Roundoff can make delta_rho_v1 slightly negative at the real-root boundary;
                # treat as zero so the >= 0 check passes and sqrt(delta_rho_v1) is valid.
                if delta_rho_v1 < zero(delta_rho_v1) && delta_rho_v1 > -arithmetic_tol
                    delta_rho_v1 = zero(delta_rho_v1)
                end
                if rho_c >= rho_floor - arithmetic_tol &&
                   delta_rho_v1 >= zero(delta_rho_v1)
                    sqrt_delta_rho_v1 = sqrt(delta_rho_v1) / a
                    for rho_v1_c in (0.5 * (rho_v1 - sqrt_delta_rho_v1),
                                     0.5 * (rho_v1 + sqrt_delta_rho_v1))
                        if (rho_e_floor * rho_c + 0.5f0 * a * rho_v1_c * rho_v1_c >
                            rho_e_total * rho_c * (1 - arithmetic_tol))
                            rho_e_total_c = rho_e_floor +
                                            0.5f0 * a * rho_v1_c * rho_v1_c / rho_c
                            rho_v2_c = (rho_v2 / rho_v1) * rho_v1_c
                            u_candidate = SVector(rho_c, rho_v1_c, rho_v2_c, rho_e_total_c)
                            best_dist_squared, best_u, has_candidate = update_best_candidate_2d!(best_dist_squared,
                                                                                                  best_u,
                                                                                                  has_candidate,
                                                                                                  u_candidate,
                                                                                                  u)
                        end
                    end
                end
            end
        end
    else
        a = 1 + (rho_v1 / rho_v2)^2
        delta_rho = rho * rho -
                    (2 * rho * rho_v2 * rho_v2 * (rho_e_total - rho_e_floor) - a * rho_v2^4) /
                    (2 * rho_v2 * rho_v2 + (rho_e_floor + rho - rho_e_total)^2 / a)
        if delta_rho >= zero(delta_rho)
            for rho_c in (0.5 * (rho - sqrt(delta_rho)), 0.5 * (rho + sqrt(delta_rho)))
                delta_rho_v2 = -8 * a * rho_c * rho_c + 8 * a * rho * rho_c + (a * rho_v2)^2
                # Roundoff can make delta_rho_v2 slightly negative at the real-root boundary;
                # treat as zero so the >= 0 check passes and sqrt(delta_rho_v2) is valid.
                if delta_rho_v2 < zero(delta_rho_v2) && delta_rho_v2 > -arithmetic_tol
                    delta_rho_v2 = zero(delta_rho_v2)
                end
                if rho_c >= rho_floor - arithmetic_tol &&
                   delta_rho_v2 >= zero(delta_rho_v2)
                    sqrt_delta_rho_v2 = sqrt(delta_rho_v2) / a
                    for rho_v2_c in (0.5 * (rho_v2 - sqrt_delta_rho_v2),
                                     0.5 * (rho_v2 + sqrt_delta_rho_v2))
                        if (rho_e_floor * rho_c + 0.5f0 * a * rho_v2_c * rho_v2_c >
                            rho_e_total * rho_c * (1 - arithmetic_tol))
                            rho_e_total_c = rho_e_floor +
                                            0.5f0 * a * rho_v2_c * rho_v2_c / rho_c
                            rho_v1_c = (rho_v1 / rho_v2) * rho_v2_c
                            u_candidate = SVector(rho_c, rho_v1_c, rho_v2_c, rho_e_total_c)
                            best_dist_squared, best_u, has_candidate = update_best_candidate_2d!(best_dist_squared,
                                                                                                  best_u,
                                                                                                  has_candidate,
                                                                                                  u_candidate,
                                                                                                  u)
                        end
                    end
                end
            end
        end
    end
    return best_dist_squared, best_u, has_candidate
end

function project_euler_2d_to_admissible_set(u, rho_floor, rho_e_floor,
                                            equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e_total = u
    RealT = typeof(rho)
    thresholds = (rho_floor, rho_e_floor)
    arithmetic_tol = euler_arithmetic_tol(rho_floor, rho_e_floor)
    @assert arithmetic_tol<minimum(thresholds) "arithmetic_tol must be smaller than the tolerance of the numerical admissible set"

    if state_is_admissible(u, thresholds, arithmetic_tol, equations)
        return u
    end

    best_dist_squared = typemax(RealT)
    best_u = zero(typeof(u))
    has_candidate = false

    # Case: mu = 0 and lambda > 0
    if rho < rho_floor &&
       2 * rho_floor * rho_e_floor + rho_v1 * rho_v1 + rho_v2 * rho_v2 <=
       2 * rho_floor * rho_e_total
        u_candidate = SVector(rho_floor, rho_v1, rho_v2, rho_e_total)
        best_dist_squared, best_u, has_candidate = update_best_candidate_2d!(best_dist_squared,
                                                                             best_u,
                                                                             has_candidate,
                                                                             u_candidate,
                                                                             u)
    end

    # Case: mu > 0 and lambda > 0
    if abs(rho_v1) < arithmetic_tol && abs(rho_v2) < arithmetic_tol
        if rho < rho_floor && rho_e_total < rho_e_floor
            u_candidate = SVector(rho_floor, zero(RealT), zero(RealT), rho_e_floor)
            best_dist_squared, best_u, has_candidate = update_best_candidate_2d!(best_dist_squared,
                                                                                   best_u,
                                                                                   has_candidate,
                                                                                   u_candidate,
                                                                                   u)
        end
    else
        use_v1_as_primary = abs(rho_v1) >= abs(rho_v2)
        best_dist_squared, best_u, has_candidate = project_euler_2d_cubic_branch!(best_dist_squared,
                                                                                  best_u,
                                                                                  has_candidate,
                                                                                  u,
                                                                                  rho_floor,
                                                                                  rho_e_floor,
                                                                                  arithmetic_tol,
                                                                                  use_v1_as_primary)
    end

    # Case: mu > 0 and lambda = 0
    if abs(rho_v1) < arithmetic_tol && abs(rho_v2) < arithmetic_tol
        if rho >= rho_floor && rho_e_total < rho_e_floor
            u_candidate = SVector(rho, zero(RealT), zero(RealT), rho_e_floor)
            best_dist_squared, best_u, has_candidate = update_best_candidate_2d!(best_dist_squared,
                                                                                   best_u,
                                                                                   has_candidate,
                                                                                   u_candidate,
                                                                                   u)
        end
    else
        use_v1_as_primary = abs(rho_v1) >= abs(rho_v2)
        best_dist_squared, best_u, has_candidate = project_euler_2d_lambda_zero_branch!(best_dist_squared,
                                                                                       best_u,
                                                                                       has_candidate,
                                                                                       u,
                                                                                       rho_floor,
                                                                                       rho_e_floor,
                                                                                       arithmetic_tol,
                                                                                       use_v1_as_primary)
    end

    if !has_candidate
        error("Failed to find projection onto Euler admissible set for state ", u,
              " with rho_floor = ", rho_floor, " and rho_e_floor = ", rho_e_floor, ".")
    end

    return best_u
end
