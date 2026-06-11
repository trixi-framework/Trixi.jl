# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

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
@inline function project_to_admissible_set(cell_average, lower_bound, variables,
                                           equations::CompressibleEulerEquations1D)
    rho_floor, rho_e_floor = variable_projection_floors(lower_bound, variables, equations)
    return project_euler_1d_to_admissible_set(cell_average, rho_floor, rho_e_floor,
                                              equations)
end

# Liu-Zhang limiting requires an internal energy bound; if a pressure bound is specified,
# it is converted to an internal energy bound.
function variable_projection_floors(thresholds, variables,
                                    equations::Union{CompressibleEulerEquations1D,
                                                     CompressibleEulerEquations2D})
    rho_floor = nothing
    rho_e_floor = nothing
    for (threshold, variable) in zip(thresholds, variables)
        if variable === Trixi.density
            rho_floor = threshold
        elseif variable === energy_internal
            rho_e_floor = threshold
        elseif variable === pressure
            rho_e_floor = threshold / (equations.gamma - 1)
        else
            error("PositivityPreservingLimiterLiuZhang for CompressibleEulerEquations1D requires ",
                  "variables = (density, energy_internal) or (density, pressure) ",
                  "(in either order); got unsupported variable.")
        end
    end
    if rho_floor === nothing || rho_e_floor === nothing
        error("PositivityPreservingLimiterLiuZhang for compressible Euler requires exactly ",
              "one density threshold and one internal-energy or pressure threshold.")
    end
    if rho_floor <= 0 || rho_e_floor <= 0
        error("Density and internal-energy floors must be positive; got rho_floor = ",
              rho_floor, " and rho_e_floor = ", rho_e_floor, ".")
    end
    return rho_floor, rho_e_floor
end

@inline function state_is_admissible(u, thresholds, arithmetic_tol,
                                     equations::CompressibleEulerEquations1D)
    rho, rho_v1, rho_e_total = u
    rho_floor, rho_e_floor = thresholds
    return rho >= rho_floor * (1 - arithmetic_tol) &&
           rho_v1 * rho_v1 + 2 * rho_e_floor * rho <=
           2 * rho * rho_e_total * (1 + arithmetic_tol)
end

# TODO: move into PositivityPreservingLimiterLiuZhang struct when 1D/2D share a common setup path
@inline function euler_arithmetic_tol(rho_floor, rho_e_floor,
                                      ::Type{RealT}) where {RealT}
    return 10 * eps(RealT)
end

@inline function projection_distance_squared_1d(rho, rho_v1, rho_e_total, x, y, z)
    return (rho - x)^2 + (rho_v1 - y)^2 + (rho_e_total - z)^2
end

@inline function consider_projection_candidate_1d!(best_dist2, best_rho, best_rho_v1,
                                                   best_rho_e_total, has_candidate,
                                                   rho, rho_v1, rho_e_total, x, y, z)
    dist2 = projection_distance_squared_1d(rho, rho_v1, rho_e_total, x, y, z)
    if !has_candidate || dist2 < best_dist2
        return dist2, rho, rho_v1, rho_e_total, true
    end
    return best_dist2, best_rho, best_rho_v1, best_rho_e_total, has_candidate
end

@inline function cubic_momentum_constraint_satisfied(rho_v1, rho_v1_orig, rho_orig,
                                                     rho_floor, rho_e_floor)
    return ((rho_v1 > zero(rho_v1) && rho_v1_orig > rho_v1) ||
            (rho_v1 < zero(rho_v1) && rho_v1_orig < rho_v1)) &&
           2 * rho_floor * rho_orig + rho_v1 * (rho_v1_orig - rho_v1) <
           2 * rho_floor * rho_e_floor
end

@inline function clamp_small_negative_discriminant(delta, arithmetic_tol)
    if delta < zero(delta) && delta > -arithmetic_tol
        return zero(delta)
    end
    return delta
end

# Real roots of m^3 + p*m + q = 0. Returns (n_roots, roots::SVector{3,T}).
# `roots` is zero-initialized; only indices 1:n_roots are overwritten. Unused slots stay
# zero so the SVector has no undefined entries (callers must still loop only 1:n_roots;
# 0 is a valid root and must not be inferred from the trailing zeros alone).
function calc_depressed_cubic_roots(p, q)
    T = typeof(p)
    roots = zeros(MVector{3, T})
    delta = 4 * p^3 + 27 * q^2
    n_roots = 0
    if delta > zero(delta)
        Y1 = 1.5 * (9 * q + sqrt(3 * delta))
        Y2 = 1.5 * (9 * q - sqrt(3 * delta))
        n_roots = 1
        roots[1] = -(sign(Y1) * abs(Y1)^(1 // 3) + sign(Y2) * abs(Y2)^(1 // 3)) / 3
    elseif iszero(delta) && (!iszero(p) || !iszero(q))
        n_roots = 2
        roots[1] = 3 * q / p
        roots[2] = -1.5 * q / p
    elseif delta < zero(delta)
        theta = acos(-1.5 * sqrt(-3 / p) * q / p)
        n_roots = 3
        roots[1] = -2 * sqrt(-p / 3) * cos(theta / 3)
        roots[2] = sqrt(-p / 3) * (cos(theta / 3) + sqrt(3) * sin(theta / 3))
        roots[3] = sqrt(-p / 3) * (cos(theta / 3) - sqrt(3) * sin(theta / 3))
    end
    return n_roots, SVector(roots)
end

function project_euler_1d_to_admissible_set(u, rho_floor, rho_e_floor,
                                            equations::CompressibleEulerEquations1D)
    rho, rho_v1, rho_e_total = u
    RealT = typeof(rho)
    thresholds = (rho_floor, rho_e_floor)
    arithmetic_tol = euler_arithmetic_tol(rho_floor, rho_e_floor, RealT)
    @assert arithmetic_tol < minimum(thresholds) "arithmetic_tol must be smaller than the tolerance of the numerical admissible set"

    if state_is_admissible(u, thresholds, arithmetic_tol, equations)
        return u
    end

    x, y, z = rho, rho_v1, rho_e_total
    best_dist2 = typemax(RealT)
    best_rho = zero(RealT)
    best_rho_v1 = zero(RealT)
    best_rho_e_total = zero(RealT)
    has_candidate = false

    # Case: mu = 0 and lambda > 0
    if x < rho_floor && 2 * rho_floor * rho_e_floor + y * y <= 2 * rho_floor * z
        best_dist2, best_rho, best_rho_v1, best_rho_e_total, has_candidate = consider_projection_candidate_1d!(best_dist2,
                                                                                                               best_rho,
                                                                                                               best_rho_v1,
                                                                                                               best_rho_e_total,
                                                                                                               has_candidate,
                                                                                                               rho_floor,
                                                                                                               y,
                                                                                                               z,
                                                                                                               x,
                                                                                                               y,
                                                                                                               z)
    end

    # Case: mu > 0 and lambda > 0
    if abs(y) < arithmetic_tol
        if x < rho_floor && z < rho_e_floor
            best_dist2, best_rho, best_rho_v1, best_rho_e_total, has_candidate = consider_projection_candidate_1d!(best_dist2,
                                                                                                                   best_rho,
                                                                                                                   best_rho_v1,
                                                                                                                   best_rho_e_total,
                                                                                                                   has_candidate,
                                                                                                                   rho_floor,
                                                                                                                   zero(RealT),
                                                                                                                   rho_e_floor,
                                                                                                                   x,
                                                                                                                   y,
                                                                                                                   z)
        end
    else
        p = 2 * rho_floor * (2 * rho_e_floor - z)
        q = -2 * rho_floor * rho_floor * y
        n_roots, roots = calc_depressed_cubic_roots(p, q)
        for i in 1:n_roots
            rho_v1_c = roots[i]
            if cubic_momentum_constraint_satisfied(rho_v1_c, y, x, rho_floor, rho_e_floor)
                rho_e_total_c = rho_e_floor + rho_v1_c * rho_v1_c / (2 * rho_floor)
                best_dist2, best_rho, best_rho_v1, best_rho_e_total, has_candidate = consider_projection_candidate_1d!(best_dist2,
                                                                                                                       best_rho,
                                                                                                                       best_rho_v1,
                                                                                                                       best_rho_e_total,
                                                                                                                       has_candidate,
                                                                                                                       rho_floor,
                                                                                                                       rho_v1_c,
                                                                                                                       rho_e_total_c,
                                                                                                                       x,
                                                                                                                       y,
                                                                                                                       z)
            end
        end
    end

    # Case: mu > 0 and lambda = 0
    if abs(y) < arithmetic_tol
        if x >= rho_floor && z < rho_e_floor
            best_dist2, best_rho, best_rho_v1, best_rho_e_total, has_candidate = consider_projection_candidate_1d!(best_dist2,
                                                                                                                   best_rho,
                                                                                                                   best_rho_v1,
                                                                                                                   best_rho_e_total,
                                                                                                                   has_candidate,
                                                                                                                   x,
                                                                                                                   zero(RealT),
                                                                                                                   rho_e_floor,
                                                                                                                   x,
                                                                                                                   y,
                                                                                                                   z)
        end
    else
        delta2 = x * x -
                 (2 * x * y * y * (z - rho_e_floor) - y^4) /
                 (2 * y * y + (rho_e_floor + x - z)^2)
        if delta2 >= zero(delta2)
            sqrt_delta2 = sqrt(delta2)
            for rho_c in (0.5 * (x - sqrt_delta2), 0.5 * (x + sqrt_delta2))
                delta3 = -8 * rho_c * rho_c + 8 * x * rho_c + y * y
                delta3 = clamp_small_negative_discriminant(delta3,
                                                           arithmetic_tol)
                if rho_c >= rho_floor - arithmetic_tol && delta3 >= zero(delta3)
                    sqrt_delta3 = sqrt(delta3)
                    for rho_v1_c in (0.5 * (y - sqrt_delta3), 0.5 * (y + sqrt_delta3))
                        # μ > 0 sign check (λ = 0 branch): candidate energy must exceed the original.
                        # ρ_c = ½(x ± √Δ₂) can suffer catastrophic cancellation when x and √Δ₂ are
                        # opposite in sign and similar in magnitude; the resulting error in ρ_c can
                        # flip this comparison. Remedies: relax (1 - arithmetic_tol), e.g. to
                        # sqrt(eps(RealT)), or detect cancellation and evaluate ρ_c via a stable formula.
                        if rho_e_floor * rho_c + 0.5f0 * rho_v1_c * rho_v1_c >
                           z * rho_c * (1 - arithmetic_tol)
                            rho_e_total_c = rho_e_floor +
                                            0.5f0 * rho_v1_c * rho_v1_c / rho_c
                            best_dist2, best_rho, best_rho_v1, best_rho_e_total, has_candidate = consider_projection_candidate_1d!(best_dist2,
                                                                                                                                   best_rho,
                                                                                                                                   best_rho_v1,
                                                                                                                                   best_rho_e_total,
                                                                                                                                   has_candidate,
                                                                                                                                   rho_c,
                                                                                                                                   rho_v1_c,
                                                                                                                                   rho_e_total_c,
                                                                                                                                   x,
                                                                                                                                   y,
                                                                                                                                   z)
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

    return SVector(best_rho, best_rho_v1, best_rho_e_total)
end

end # @muladd