@inline function project_to_admissible_set(cell_average, lower_bound, variables,
                                           equations::CompressibleEulerEquations1D)
    rho_floor, rho_e_floor = variable_projection_floors(lower_bound, variables, equations)
    return project_euler_1d_to_admissible_set(cell_average, rho_floor, rho_e_floor)
end

@inline function project_to_admissible_set(cell_average, lower_bound, variables,
                                           equations::CompressibleEulerEquations2D)
    rho_floor, rho_e_floor = variable_projection_floors(lower_bound, variables, equations)
    return project_euler_2d_to_admissible_set(cell_average, rho_floor, rho_e_floor)
end

function variable_projection_floors(thresholds, variables,
                                    equations::Union{CompressibleEulerEquations1D,
                                                     CompressibleEulerEquations2D})
    rho_floor = nothing
    rho_e_floor = nothing
    for (threshold, variable) in zip(thresholds, variables)
        if limiter_variable_is(variable, Trixi.density)
            rho_floor = threshold
        elseif limiter_variable_is(variable, energy_internal)
            rho_e_floor = threshold
        elseif limiter_variable_is(variable, pressure)
            rho_e_floor = threshold / (equations.gamma - 1)
        else
            error("PositivityPreservingLimiterLiuZhang for compressible Euler requires ",
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

@inline function limiter_variable_is(variable, reference)
    return variable === reference
end

@inline function euler_admissible_1d(rho, m, E, rho_floor, rho_e_floor)
    return rho >= rho_floor && m * m + 2 * rho_e_floor * rho <= 2 * rho * E
end

@inline function euler_admissible_2d(rho, m1, m2, E, rho_floor, rho_e_floor)
    return rho >= rho_floor &&
           m1 * m1 + m2 * m2 + 2 * rho_e_floor * rho <= 2 * rho * E
end

@inline function euler_projection_artm_tol(rho_floor, rho_e_floor,
                                           ::Type{RealT}) where {RealT}
    return min(rho_floor, rho_e_floor) * sqrt(eps(RealT))
end

@inline function projection_distance_squared_1d(rho, m, E, x, y, z)
    return (rho - x)^2 + (m - y)^2 + (E - z)^2
end

@inline function projection_distance_squared_2d(rho, m1, m2, E, x, y1, y2, z)
    return (rho - x)^2 + (m1 - y1)^2 + (m2 - y2)^2 + (E - z)^2
end

@inline function consider_projection_candidate_1d!(best_dist2, best_rho, best_m, best_E,
                                                   has_candidate,
                                                   rho, m, E, x, y, z)
    dist2 = projection_distance_squared_1d(rho, m, E, x, y, z)
    if !has_candidate || dist2 < best_dist2
        return dist2, rho, m, E, true
    end
    return best_dist2, best_rho, best_m, best_E, has_candidate
end

@inline function consider_projection_candidate_2d!(best_dist2, best_rho, best_m1, best_m2,
                                                   best_E, has_candidate,
                                                   rho, m1, m2, E, x, y1, y2, z)
    dist2 = projection_distance_squared_2d(rho, m1, m2, E, x, y1, y2, z)
    if !has_candidate || dist2 < best_dist2
        return dist2, rho, m1, m2, E, true
    end
    return best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate
end

@inline function cubic_momentum_constraint_satisfied(m, y, x, rho_floor, rho_e_floor)
    return ((m > zero(m) && y > m) || (m < zero(m) && y < m)) &&
           2 * rho_floor * x + m * (y - m) < 2 * rho_floor * rho_e_floor
end

@inline function cubic_momentum_constraint_satisfied_2d(m1, y1, x, a, rho_floor,
                                                        rho_e_floor)
    return ((m1 > zero(m1) && y1 > m1) || (m1 < zero(m1) && y1 < m1)) &&
           2 * rho_floor * x + a * m1 * (y1 - m1) < 2 * rho_floor * rho_e_floor
end

@inline function clamp_small_negative_discriminant(delta, artm_tol)
    if delta < zero(delta) && delta > -artm_tol
        return zero(delta)
    end
    return delta
end

# Fill at most three real roots of m^3 + p*m + q = 0 into `roots`; return the number of roots.
function fill_depressed_cubic_roots!(roots, p, q)
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
    return n_roots
end

function project_euler_1d_to_admissible_set(u, rho_floor, rho_e_floor)
    rho, m, E = u
    RealT = typeof(rho)
    artm_tol = euler_projection_artm_tol(rho_floor, rho_e_floor, RealT)

    if euler_admissible_1d(rho, m, E, rho_floor, rho_e_floor)
        return u
    end

    x, y, z = rho, m, E
    best_dist2 = typemax(RealT)
    best_rho = zero(RealT)
    best_m = zero(RealT)
    best_E = zero(RealT)
    has_candidate = false

    # Case: mu = 0 and lambda > 0
    if x < rho_floor && 2 * rho_floor * rho_e_floor + y * y <= 2 * rho_floor * z
        best_dist2, best_rho, best_m, best_E, has_candidate = consider_projection_candidate_1d!(best_dist2,
                                                                                                best_rho,
                                                                                                best_m,
                                                                                                best_E,
                                                                                                has_candidate,
                                                                                                rho_floor,
                                                                                                y,
                                                                                                z,
                                                                                                x,
                                                                                                y,
                                                                                                z)
    end

    # Case: mu > 0 and lambda > 0
    if abs(y) < artm_tol
        if x < rho_floor && z < rho_e_floor
            best_dist2, best_rho, best_m, best_E, has_candidate = consider_projection_candidate_1d!(best_dist2,
                                                                                                    best_rho,
                                                                                                    best_m,
                                                                                                    best_E,
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
        roots = MVector{3, RealT}(undef)
        n_roots = fill_depressed_cubic_roots!(roots, p, q)
        for i in 1:n_roots
            m1 = roots[i]
            if cubic_momentum_constraint_satisfied(m1, y, x, rho_floor, rho_e_floor)
                E1 = rho_e_floor + m1 * m1 / (2 * rho_floor)
                best_dist2, best_rho, best_m, best_E, has_candidate = consider_projection_candidate_1d!(best_dist2,
                                                                                                        best_rho,
                                                                                                        best_m,
                                                                                                        best_E,
                                                                                                        has_candidate,
                                                                                                        rho_floor,
                                                                                                        m1,
                                                                                                        E1,
                                                                                                        x,
                                                                                                        y,
                                                                                                        z)
            end
        end
    end

    # Case: mu > 0 and lambda = 0
    if abs(y) < artm_tol
        if x >= rho_floor && z < rho_e_floor
            best_dist2, best_rho, best_m, best_E, has_candidate = consider_projection_candidate_1d!(best_dist2,
                                                                                                    best_rho,
                                                                                                    best_m,
                                                                                                    best_E,
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
            for rho_c in (0.5 * (x - sqrt(delta2)), 0.5 * (x + sqrt(delta2)))
                delta3 = -8 * rho_c * rho_c + 8 * x * rho_c + y * y
                delta3 = clamp_small_negative_discriminant(delta3, artm_tol)
                if rho_c >= rho_floor - artm_tol && delta3 >= zero(delta3)
                    sqrt_delta3 = sqrt(delta3)
                    for m_c in (0.5 * (y - sqrt_delta3), 0.5 * (y + sqrt_delta3))
                        if rho_e_floor * rho_c + 0.5f0 * m_c * m_c >
                           z * rho_c * (1 - artm_tol)
                            E_c = rho_e_floor + 0.5f0 * m_c * m_c / rho_c
                            best_dist2, best_rho, best_m, best_E, has_candidate = consider_projection_candidate_1d!(best_dist2,
                                                                                                                    best_rho,
                                                                                                                    best_m,
                                                                                                                    best_E,
                                                                                                                    has_candidate,
                                                                                                                    rho_c,
                                                                                                                    m_c,
                                                                                                                    E_c,
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
              " with rho_floor = ", rho_floor, " and rho_e_floor = ", rho_e_floor, ".")
    end

    return SVector(best_rho, best_m, best_E)
end

function project_euler_2d_cubic_branch!(best_dist2, best_rho, best_m1, best_m2, best_E,
                                        has_candidate, x, y1, y2, z, rho_floor, rho_e_floor,
                                        artm_tol, use_v1_as_primary)
    roots = MVector{3, typeof(x)}(undef)
    if use_v1_as_primary
        a = 1 + (y2 / y1)^2
        p = rho_floor * (4 * rho_e_floor - 2 * z) / a
        q = -2 * rho_floor * rho_e_floor * y1 / a
        n_roots = fill_depressed_cubic_roots!(roots, p, q)
        for i in 1:n_roots
            m1 = roots[i]
            if cubic_momentum_constraint_satisfied_2d(m1, y1, x, a, rho_floor, rho_e_floor)
                E_c = rho_e_floor + a * m1 * m1 / (2 * rho_floor)
                best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate = consider_projection_candidate_2d!(best_dist2,
                                                                                                                  best_rho,
                                                                                                                  best_m1,
                                                                                                                  best_m2,
                                                                                                                  best_E,
                                                                                                                  has_candidate,
                                                                                                                  rho_floor,
                                                                                                                  m1,
                                                                                                                  (y2 /
                                                                                                                   y1) *
                                                                                                                  m1,
                                                                                                                  E_c,
                                                                                                                  x,
                                                                                                                  y1,
                                                                                                                  y2,
                                                                                                                  z)
            end
        end
    else
        a = 1 + (y1 / y2)^2
        p = rho_floor * (4 * rho_e_floor - 2 * z) / a
        q = -2 * rho_floor * rho_e_floor * y2 / a
        n_roots = fill_depressed_cubic_roots!(roots, p, q)
        for i in 1:n_roots
            m2 = roots[i]
            if cubic_momentum_constraint_satisfied_2d(m2, y2, x, a, rho_floor, rho_e_floor)
                E_c = rho_e_floor + a * m2 * m2 / (2 * rho_floor)
                best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate = consider_projection_candidate_2d!(best_dist2,
                                                                                                                  best_rho,
                                                                                                                  best_m1,
                                                                                                                  best_m2,
                                                                                                                  best_E,
                                                                                                                  has_candidate,
                                                                                                                  rho_floor,
                                                                                                                  (y1 /
                                                                                                                   y2) *
                                                                                                                  m2,
                                                                                                                  m2,
                                                                                                                  E_c,
                                                                                                                  x,
                                                                                                                  y1,
                                                                                                                  y2,
                                                                                                                  z)
            end
        end
    end
    return best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate
end

function project_euler_2d_lambda_zero_branch!(best_dist2, best_rho, best_m1, best_m2,
                                              best_E,
                                              has_candidate, x, y1, y2, z, rho_floor,
                                              rho_e_floor, artm_tol, use_v1_as_primary)
    if use_v1_as_primary
        a = 1 + (y2 / y1)^2
        delta_rho = x * x -
                    (2 * x * y1 * y1 * (z - rho_e_floor) - a * y1^4) /
                    (2 * y1 * y1 + (rho_e_floor + x - z)^2 / a)
        if delta_rho >= zero(delta_rho)
            for rho_c in (0.5 * (x - sqrt(delta_rho)), 0.5 * (x + sqrt(delta_rho)))
                delta_m1 = -8 * a * rho_c * rho_c + 8 * a * x * rho_c + (a * y1)^2
                delta_m1 = clamp_small_negative_discriminant(delta_m1, artm_tol)
                if rho_c >= rho_floor - artm_tol && delta_m1 >= zero(delta_m1)
                    sqrt_delta_m1 = sqrt(delta_m1) / a
                    for m1_c in (0.5 * (y1 - sqrt_delta_m1), 0.5 * (y1 + sqrt_delta_m1))
                        if (rho_e_floor * rho_c + 0.5f0 * a * m1_c * m1_c >
                            z * rho_c * (1 - artm_tol))
                            E_c = rho_e_floor + 0.5f0 * a * m1_c * m1_c / rho_c
                            best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate = consider_projection_candidate_2d!(best_dist2,
                                                                                                                              best_rho,
                                                                                                                              best_m1,
                                                                                                                              best_m2,
                                                                                                                              best_E,
                                                                                                                              has_candidate,
                                                                                                                              rho_c,
                                                                                                                              m1_c,
                                                                                                                              (y2 /
                                                                                                                               y1) *
                                                                                                                              m1_c,
                                                                                                                              E_c,
                                                                                                                              x,
                                                                                                                              y1,
                                                                                                                              y2,
                                                                                                                              z)
                        end
                    end
                end
            end
        end
    else
        a = 1 + (y1 / y2)^2
        delta_rho = x * x -
                    (2 * x * y2 * y2 * (z - rho_e_floor) - a * y2^4) /
                    (2 * y2 * y2 + (rho_e_floor + x - z)^2 / a)
        if delta_rho >= zero(delta_rho)
            for rho_c in (0.5 * (x - sqrt(delta_rho)), 0.5 * (x + sqrt(delta_rho)))
                delta_m2 = -8 * a * rho_c * rho_c + 8 * a * x * rho_c + (a * y2)^2
                delta_m2 = clamp_small_negative_discriminant(delta_m2, artm_tol)
                if rho_c >= rho_floor - artm_tol && delta_m2 >= zero(delta_m2)
                    sqrt_delta_m2 = sqrt(delta_m2) / a
                    for m2_c in (0.5 * (y2 - sqrt_delta_m2), 0.5 * (y2 + sqrt_delta_m2))
                        if (rho_e_floor * rho_c + 0.5f0 * a * m2_c * m2_c >
                            z * rho_c * (1 - artm_tol))
                            E_c = rho_e_floor + 0.5f0 * a * m2_c * m2_c / rho_c
                            best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate = consider_projection_candidate_2d!(best_dist2,
                                                                                                                              best_rho,
                                                                                                                              best_m1,
                                                                                                                              best_m2,
                                                                                                                              best_E,
                                                                                                                              has_candidate,
                                                                                                                              rho_c,
                                                                                                                              (y1 /
                                                                                                                               y2) *
                                                                                                                              m2_c,
                                                                                                                              m2_c,
                                                                                                                              E_c,
                                                                                                                              x,
                                                                                                                              y1,
                                                                                                                              y2,
                                                                                                                              z)
                        end
                    end
                end
            end
        end
    end
    return best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate
end

function project_euler_2d_to_admissible_set(u, rho_floor, rho_e_floor)
    rho, m1, m2, E = u
    RealT = typeof(rho)
    artm_tol = euler_projection_artm_tol(rho_floor, rho_e_floor, RealT)

    if euler_admissible_2d(rho, m1, m2, E, rho_floor, rho_e_floor)
        return u
    end

    x, y1, y2, z = rho, m1, m2, E
    best_dist2 = typemax(RealT)
    best_rho = zero(RealT)
    best_m1 = zero(RealT)
    best_m2 = zero(RealT)
    best_E = zero(RealT)
    has_candidate = false

    # Case: mu = 0 and lambda > 0
    if x < rho_floor && 2 * rho_floor * rho_e_floor + y1 * y1 + y2 * y2 <= 2 * rho_floor * z
        best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate = consider_projection_candidate_2d!(best_dist2,
                                                                                                          best_rho,
                                                                                                          best_m1,
                                                                                                          best_m2,
                                                                                                          best_E,
                                                                                                          has_candidate,
                                                                                                          rho_floor,
                                                                                                          y1,
                                                                                                          y2,
                                                                                                          z,
                                                                                                          x,
                                                                                                          y1,
                                                                                                          y2,
                                                                                                          z)
    end

    # Case: mu > 0 and lambda > 0
    if abs(y1) < artm_tol && abs(y2) < artm_tol
        if x < rho_floor && z < rho_e_floor
            best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate = consider_projection_candidate_2d!(best_dist2,
                                                                                                              best_rho,
                                                                                                              best_m1,
                                                                                                              best_m2,
                                                                                                              best_E,
                                                                                                              has_candidate,
                                                                                                              rho_floor,
                                                                                                              zero(RealT),
                                                                                                              zero(RealT),
                                                                                                              rho_e_floor,
                                                                                                              x,
                                                                                                              y1,
                                                                                                              y2,
                                                                                                              z)
        end
    else
        use_v1_as_primary = abs(y1) >= abs(y2)
        best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate = project_euler_2d_cubic_branch!(best_dist2,
                                                                                                       best_rho,
                                                                                                       best_m1,
                                                                                                       best_m2,
                                                                                                       best_E,
                                                                                                       has_candidate,
                                                                                                       x,
                                                                                                       y1,
                                                                                                       y2,
                                                                                                       z,
                                                                                                       rho_floor,
                                                                                                       rho_e_floor,
                                                                                                       artm_tol,
                                                                                                       use_v1_as_primary)
    end

    # Case: mu > 0 and lambda = 0
    if abs(y1) < artm_tol && abs(y2) < artm_tol
        if x >= rho_floor && z < rho_e_floor
            best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate = consider_projection_candidate_2d!(best_dist2,
                                                                                                              best_rho,
                                                                                                              best_m1,
                                                                                                              best_m2,
                                                                                                              best_E,
                                                                                                              has_candidate,
                                                                                                              x,
                                                                                                              zero(RealT),
                                                                                                              zero(RealT),
                                                                                                              rho_e_floor,
                                                                                                              x,
                                                                                                              y1,
                                                                                                              y2,
                                                                                                              z)
        end
    else
        use_v1_as_primary = abs(y1) >= abs(y2)
        best_dist2, best_rho, best_m1, best_m2, best_E, has_candidate = project_euler_2d_lambda_zero_branch!(best_dist2,
                                                                                                             best_rho,
                                                                                                             best_m1,
                                                                                                             best_m2,
                                                                                                             best_E,
                                                                                                             has_candidate,
                                                                                                             x,
                                                                                                             y1,
                                                                                                             y2,
                                                                                                             z,
                                                                                                             rho_floor,
                                                                                                             rho_e_floor,
                                                                                                             artm_tol,
                                                                                                             use_v1_as_primary)
    end

    if !has_candidate
        error("Failed to find projection onto Euler admissible set for state ", u,
              " with rho_floor = ", rho_floor, " and rho_e_floor = ", rho_e_floor, ".")
    end

    return SVector(best_rho, best_m1, best_m2, best_E)
end
