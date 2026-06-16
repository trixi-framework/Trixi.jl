# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function state_is_admissible(u, thresholds,
                                     equations::CompressibleEulerEquations1D)
    rho, rho_v1, rho_e_total = u
    rho_floor, rho_e_floor = thresholds
    return rho >= rho_floor &&
           rho_v1 * rho_v1 + 2 * rho_e_floor * rho <= 2 * rho * rho_e_total
end

# for compressible Euler, we introduce a small tolerance close to 
# machine precision to relax the strict inequalities enforced by the 
# constraints. 
@inline function euler_arithmetic_tol(rho_floor, rho_e_floor)
    T = promote_type(typeof(rho_floor), typeof(rho_e_floor))
    return 10 * eps(T)
end

@inline function projection_distance_squared_1d(u_candidate, u)
    return sum(abs2, u_candidate - u)
end

# Return (best_dist_squared, best_u, has_candidate) updated when u_candidate 
# is closer to u than the current best; otherwise return the current best.
@inline function update_best_candidate_1d!(best_dist_squared, best_u,
                                           has_candidate,
                                           u_candidate, u)
    dist2 = projection_distance_squared_1d(u_candidate, u)
    if !has_candidate || dist2 < best_dist_squared
        return dist2, u_candidate, true
    end
    return best_dist_squared, best_u, has_candidate
end

# Used in the μ > 0, λ > 0 branch of Appendix B.2 of Liu, Milesis, Shu, Zhang (2026).
@inline function cubic_momentum_constraint_satisfied(rho_v1, rho_v1_orig, rho_orig,
                                                     rho_floor, rho_e_floor)
    return ((rho_v1 > zero(rho_v1) && rho_v1_orig > rho_v1) ||
            (rho_v1 < zero(rho_v1) && rho_v1_orig < rho_v1)) &&
           2 * rho_floor * rho_orig + rho_v1 * (rho_v1_orig - rho_v1) <
           2 * rho_floor * rho_e_floor
end

# Real roots of m^3 + p*m + q = 0. Returns (n_roots, roots::SVector{3,T}).
# Note that roots[n_roots+1:end] are aren't accessed. 
function calc_depressed_cubic_roots(p, q)
    T = typeof(p)
    delta = 4 * p^3 + 27 * q^2
    n_roots = 0
    if delta > zero(delta)
        Y1 = 1.5 * (9 * q + sqrt(3 * delta))
        Y2 = 1.5 * (9 * q - sqrt(3 * delta))
        n_roots = 1
        root_1 = -(sign(Y1) * abs(Y1)^(1 // 3) + sign(Y2) * abs(Y2)^(1 // 3)) / 3
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
    thresholds = (rho_floor, rho_e_floor)
    arithmetic_tol = euler_arithmetic_tol(rho_floor, rho_e_floor)
    @assert arithmetic_tol<minimum(thresholds) "arithmetic_tol must be smaller than the tolerance of the numerical admissible set"

    if state_is_admissible(u, thresholds, equations)
        return u
    end

    best_dist_squared = typemax(RealT)
    best_u = zero(typeof(u))
    has_candidate = false

    # Case: mu = 0 and lambda > 0
    if rho < rho_floor &&
       2 * rho_floor * rho_e_floor + rho_v1 * rho_v1 <= 2 * rho_floor * rho_e_total
        u_candidate = SVector(rho_floor, rho_v1, rho_e_total)
        best_dist_squared, best_u, has_candidate = update_best_candidate_1d!(best_dist_squared,
                                                                             best_u,
                                                                             has_candidate,
                                                                             u_candidate,
                                                                             u)
    end

    # Case: mu > 0 and lambda > 0
    if abs(rho_v1) < arithmetic_tol
        if rho < rho_floor && rho_e_total < rho_e_floor
            u_candidate = SVector(rho_floor, zero(RealT), rho_e_floor)
            best_dist_squared, best_u, has_candidate = update_best_candidate_1d!(best_dist_squared,
                                                                                 best_u,
                                                                                 has_candidate,
                                                                                 u_candidate,
                                                                                 u)
        end
    else
        p = 2 * rho_floor * (2 * rho_e_floor - rho_e_total)
        q = -2 * rho_floor * rho_floor * rho_v1
        n_roots, roots = calc_depressed_cubic_roots(p, q)
        for i in 1:n_roots
            rho_v1_candidate = roots[i]
            if cubic_momentum_constraint_satisfied(rho_v1_candidate, rho_v1, rho,
                                                   rho_floor, rho_e_floor)
                rho_e_total_c = rho_e_floor +
                                rho_v1_candidate * rho_v1_candidate / (2 * rho_floor)
                u_candidate = SVector(rho_floor, rho_v1_candidate, rho_e_total_c)
                best_dist_squared, best_u, has_candidate = update_best_candidate_1d!(best_dist_squared,
                                                                                     best_u,
                                                                                     has_candidate,
                                                                                     u_candidate,
                                                                                     u)
            end
        end
    end

    # Case: mu > 0 and lambda = 0
    if abs(rho_v1) < arithmetic_tol
        if rho >= rho_floor && rho_e_total < rho_e_floor
            u_candidate = SVector(rho, zero(RealT), rho_e_floor)
            best_dist_squared, best_u, has_candidate = update_best_candidate_1d!(best_dist_squared,
                                                                                 best_u,
                                                                                 has_candidate,
                                                                                 u_candidate,
                                                                                 u)
        end
    else
        delta2 = rho * rho -
                 (2 * rho * rho_v1 * rho_v1 * (rho_e_total - rho_e_floor) - rho_v1^4) /
                 (2 * rho_v1 * rho_v1 + (rho_e_floor + rho - rho_e_total)^2)
        if delta2 >= zero(delta2)
            sqrt_delta2 = sqrt(delta2)
            for rho_c in (0.5 * (rho - sqrt_delta2), 0.5 * (rho + sqrt_delta2))
                delta3 = -8 * rho_c * rho_c + 8 * rho * rho_c + rho_v1 * rho_v1
                # Roundoff can make delta3 slightly negative at the real-root boundary;
                # treat as zero so the >= 0 check passes and sqrt(delta3) is valid.
                if delta3 < zero(delta3) && delta3 > -arithmetic_tol
                    delta3 = zero(delta3)
                end
                if rho_c >= rho_floor - arithmetic_tol && delta3 >= zero(delta3)
                    sqrt_delta3 = sqrt(delta3)
                    for rho_v1_candidate in (0.5 * (rho_v1 - sqrt_delta3),
                                             0.5 * (rho_v1 + sqrt_delta3))
                        # μ > 0 sign check (λ = 0 branch): candidate energy must exceed the original.
                        # ρ_c = ½(ρ ± √Δ₂) can suffer catastrophic cancellation when ρ and √Δ₂ are
                        # opposite in sign and similar in magnitude; the resulting error in ρ_c can
                        # flip this comparison. Remedies: relax (1 - arithmetic_tol), e.g. to
                        # sqrt(eps(RealT)), or detect cancellation and evaluate ρ_c via a stable formula.
                        if rho_e_floor * rho_c +
                           0.5f0 * rho_v1_candidate * rho_v1_candidate >
                           rho_e_total * rho_c * (1 - arithmetic_tol)
                            rho_e_total_c = rho_e_floor +
                                            0.5f0 * rho_v1_candidate *
                                            rho_v1_candidate / rho_c
                            u_candidate = SVector(rho_c, rho_v1_candidate,
                                                  rho_e_total_c)
                            best_dist_squared, best_u, has_candidate = update_best_candidate_1d!(best_dist_squared,
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

    if !has_candidate
        error("Failed to find projection onto Euler admissible set for state ", u,
              " with rho = ", rho, " and rho_e = ",
              rho_e_total - 0.5f0 * rho_v1 * rho_v1 / rho,
              " and rho_floor = ", rho_floor, " and rho_e_floor = ", rho_e_floor, ".")
    end

    return best_u
end
end # @muladd
