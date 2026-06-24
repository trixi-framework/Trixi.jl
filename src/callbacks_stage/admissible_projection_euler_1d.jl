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
    T = typeof(p)
    delta = 4 * p^3 + 27 * q^2
    n_roots = 0
    if delta > zero(delta)
        Y1 = 1.5 * (9 * q + sqrt(3 * delta))
        Y2 = 1.5 * (9 * q - sqrt(3 * delta))
        n_roots = 1
        root_1 = -(sign(Y1) * abs(Y1)^(1 / 3) + sign(Y2) * abs(Y2)^(1 / 3)) / 3
        root_2 = zero(T) # not used
        root_3 = zero(T) # not used
    elseif iszero(delta) && (!iszero(p) || !iszero(q))
        n_roots = 2
        root_1 = 3 * q / p
        root_2 = -1.5 * q / p
        root_3 = zero(T) # not used
    elseif delta < zero(delta)
        theta = acos(-1.5 * sqrt(-3 / p) * q / p)
        n_roots = 3
        root_1 = -2 * sqrt(-p / 3) * cos(theta / 3)
        root_2 = sqrt(-p / 3) * (cos(theta / 3) + sqrt(3) * sin(theta / 3))
        root_3 = sqrt(-p / 3) * (cos(theta / 3) - sqrt(3) * sin(theta / 3))
    end
    return n_roots, SVector(root_1, root_2, root_3)
end

"""
    project_to_admissible_set(cell_average, lower_bound, variables,
                              equations::CompressibleEulerEquations1D)

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
                                   equations::CompressibleEulerEquations1D)
    rho_floor, rho_e_floor = lower_bounds
    u = cell_average
    rho, rho_v1, rho_e_total = u
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
    density_at_or_above_floor = rho >= rho_floor
    momentum_is_near_zero = abs(rho_v1) < arithmetic_tol

    # Case: mu = 0 and lambda > 0
    energy_internal_lower_bound_at_rho_floor = 2 * rho_floor * rho_e_floor +
                                             rho_v1 * rho_v1
    energy_internal_budget_at_rho_floor = 2 * rho_floor * rho_e_total
    energy_internal_admissible_after_density_lift = energy_internal_lower_bound_at_rho_floor <=
                                                    energy_internal_budget_at_rho_floor

    # mu = 0 and lambda > 0: internal-energy constraint is active.
    case_mu_is_zero_and_lambda_is_positive = density_below_floor &&
                                             energy_internal_admissible_after_density_lift
    if case_mu_is_zero_and_lambda_is_positive
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
        total_energy_internal_below_floor_at_zero_velocity = rho_e_total < rho_e_floor
        # mu > 0 and lambda > 0: internal-energy constraint is active.
        case_mu_is_positive_and_lambda_is_positive_zero_momentum = density_below_floor &&
                                                                   total_energy_internal_below_floor_at_zero_velocity
        if case_mu_is_positive_and_lambda_is_positive_zero_momentum
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
        energy_internal_below_floor = rho_e_total < rho_e_floor
        case_mu_is_positive_and_lambda_is_zero_zero_momentum = density_at_or_above_floor &&
                                                               energy_internal_below_floor
        if case_mu_is_positive_and_lambda_is_zero_zero_momentum
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
            for rho_candidate in (0.5 * (rho - sqrt_discriminant_rho),
                                  0.5 * (rho + sqrt_discriminant_rho))
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
                    for rho_v1_candidate in (0.5 *
                                             (rho_v1 - sqrt_discriminant_momentum),
                                             0.5 *
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
                        lambda_zero_energy_internal_sign_condition = candidate_energy_internal_times_rho >
                                                                     original_energy_internal_times_rho
                        if lambda_zero_energy_internal_sign_condition
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
