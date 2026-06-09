@inline function project_to_admissible_set(cell_average, lower_bound, variables,
                                           equations::CompressibleEulerEquations2D)
    rho_floor, rho_e_floor = variable_projection_floors(lower_bound, variables, equations)
    return project_euler_2d_to_admissible_set(cell_average, rho_floor, rho_e_floor)
end

@inline function euler_admissible_2d(rho, m1, m2, E, rho_floor, rho_e_floor)
    return rho >= rho_floor &&
           m1 * m1 + m2 * m2 + 2 * rho_e_floor * rho <= 2 * rho * E
end

@inline function projection_distance_squared_2d(rho, m1, m2, E, x, y1, y2, z)
    return (rho - x)^2 + (m1 - y1)^2 + (m2 - y2)^2 + (E - z)^2
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

@inline function cubic_momentum_constraint_satisfied_2d(m1, y1, x, a, rho_floor,
                                                        rho_e_floor)
    return ((m1 > zero(m1) && y1 > m1) || (m1 < zero(m1) && y1 < m1)) &&
           2 * rho_floor * x + a * m1 * (y1 - m1) < 2 * rho_floor * rho_e_floor
end

function project_euler_2d_cubic_branch!(best_dist2, best_rho, best_m1, best_m2, best_E,
                                        has_candidate, x, y1, y2, z, rho_floor, rho_e_floor,
                                        admissible_projection_tol, use_v1_as_primary)
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
                                              rho_e_floor, admissible_projection_tol,
                                              use_v1_as_primary)
    if use_v1_as_primary
        a = 1 + (y2 / y1)^2
        delta_rho = x * x -
                    (2 * x * y1 * y1 * (z - rho_e_floor) - a * y1^4) /
                    (2 * y1 * y1 + (rho_e_floor + x - z)^2 / a)
        if delta_rho >= zero(delta_rho)
            for rho_c in (0.5 * (x - sqrt(delta_rho)), 0.5 * (x + sqrt(delta_rho)))
                delta_m1 = -8 * a * rho_c * rho_c + 8 * a * x * rho_c + (a * y1)^2
                delta_m1 = clamp_small_negative_discriminant(delta_m1,
                                                             admissible_projection_tol)
                if rho_c >= rho_floor - admissible_projection_tol &&
                   delta_m1 >= zero(delta_m1)
                    sqrt_delta_m1 = sqrt(delta_m1) / a
                    for m1_c in (0.5 * (y1 - sqrt_delta_m1), 0.5 * (y1 + sqrt_delta_m1))
                        if (rho_e_floor * rho_c + 0.5f0 * a * m1_c * m1_c >
                            z * rho_c * (1 - admissible_projection_tol))
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
                delta_m2 = clamp_small_negative_discriminant(delta_m2,
                                                             admissible_projection_tol)
                if rho_c >= rho_floor - admissible_projection_tol &&
                   delta_m2 >= zero(delta_m2)
                    sqrt_delta_m2 = sqrt(delta_m2) / a
                    for m2_c in (0.5 * (y2 - sqrt_delta_m2), 0.5 * (y2 + sqrt_delta_m2))
                        if (rho_e_floor * rho_c + 0.5f0 * a * m2_c * m2_c >
                            z * rho_c * (1 - admissible_projection_tol))
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
    admissible_projection_tol = euler_admissible_projection_tol(rho_floor, rho_e_floor,
                                                                RealT)

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
    if abs(y1) < admissible_projection_tol && abs(y2) < admissible_projection_tol
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
                                                                                                       admissible_projection_tol,
                                                                                                       use_v1_as_primary)
    end

    # Case: mu > 0 and lambda = 0
    if abs(y1) < admissible_projection_tol && abs(y2) < admissible_projection_tol
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
                                                                                                             admissible_projection_tol,
                                                                                                             use_v1_as_primary)
    end

    if !has_candidate
        error("Failed to find projection onto Euler admissible set for state ", u,
              " with rho_floor = ", rho_floor, " and rho_e_floor = ", rho_e_floor, ".")
    end

    return SVector(best_rho, best_m1, best_m2, best_E)
end
