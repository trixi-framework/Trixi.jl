# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# 1D version
function correct_u!(u_dgfv::AbstractArray{<:Any, 3}, u_fv, u_dg_node,
                    delta_alpha, alpha, alpha_max,
                    element, semi)
    if delta_alpha > 0
        if alpha[element] + delta_alpha > alpha_max
            delta_alpha = alpha_max - alpha[element]
            alpha[element] = alpha_max
        else
            alpha[element] += delta_alpha
        end

        @unpack solver, equations = semi

        for i in eachnode(solver)
            u_fv_node = get_node_vars(u_fv, equations, solver, i, element)
            u_dgfv_node = get_node_vars(u_dgfv, equations, solver, i, element)
            for v in eachvariable(equations)
                u_dg_node[v] = compute_pure_dg(u_dgfv_node[v], u_fv_node[v],
                                               alpha[element])

                u_dgfv[v, i, element] = u_dgfv_node[v] +
                                        delta_alpha * (u_fv_node[v] - u_dg_node[v])
            end
        end
    end

    return nothing
end

# `mesh` passed in only for dispatch
function limiter_rueda_gassner!(u_dgfv, mesh::AbstractMesh{1}, semi, limiter!)
    @unpack equations, solver, cache = semi
    @unpack alpha = solver.volume_integral.indicator.cache

    @unpack beta_rho, beta_p, alpha_max, near_zero_tol,
    max_iterations, root_tol, damping, use_density_init,
    solver_fv, u_fv_ode,
    u_dg_node_threaded, du_dalpha_node_threaded, dp_du_node_threaded, u_newton_node_threaded = limiter!

    u_fv = wrap_array(u_fv_ode, semi)

    @threaded for element in eachelement(solver, cache)
        # If alpha is already maximized we are not permitted to correct
        if abs(alpha[element] - alpha_max) < near_zero_tol
            continue
        end

        delta_alpha = zero(eltype(u_dgfv)) # Element-wise correction

        ### Density correction process ###
        for i in eachnode(solver)
            u_fv_node = get_node_vars(u_fv, equations, solver, i, element)
            rho_fv = density(u_fv_node, equations)
            if rho_fv < 0 || isnan(rho_fv) || isinf(rho_fv)
                error("Finite-Volume solution produces invalid value for density!")
            end

            u_dgfv_node = get_node_vars(u_dgfv, equations, solver, i, element)
            rho_dgfv = density(u_dgfv_node, equations)

            a_rho = beta_rho * rho_fv - rho_dgfv
            if a_rho > root_tol # => non-zero `delta_alpha` required
                rho_dg = compute_pure_dg(rho_dgfv, rho_fv, alpha[element])

                # Avoid division (and correction due to this node) for densities close to each other
                if abs(rho_fv - rho_dg) < near_zero_tol
                    continue
                end

                delta_alpha_i = a_rho / (rho_fv - rho_dg)

                # `delta_alpha_i` is calculated at each node, use maximum for the entire element
                delta_alpha = max(delta_alpha, delta_alpha_i)
            end
        end

        # Get thread-local storage
        u_dg_node = u_dg_node_threaded[Threads.threadid()]
        du_dalpha_node = du_dalpha_node_threaded[Threads.threadid()]
        dp_du_node = dp_du_node_threaded[Threads.threadid()]
        u_newton_node = u_newton_node_threaded[Threads.threadid()]

        # Correct density
        correct_u!(u_dgfv, u_fv, u_dg_node,
                   delta_alpha, alpha, alpha_max,
                   element, semi)

        ### Pressure correction process ###
        delta_alpha_density = delta_alpha # Used for Newton init
        delta_alpha = zero(eltype(u_dgfv)) # Element-wise correction
        for i in eachnode(solver)
            u_fv_node = get_node_vars(u_fv, equations, solver, i, element)
            p_fv = pressure(u_fv_node, equations)
            if p_fv < 0 || isnan(p_fv) || isinf(p_fv)
                error("Finite-Volume solution produces invalid value for pressure!")
            end

            u_dgfv_node = get_node_vars(u_dgfv, equations, solver, i, element)
            p_dgfv = pressure(u_dgfv_node, equations)

            a_p = beta_p * p_fv - p_dgfv # This is -g(alpha_new) in the paper, see eq. (15)
            if a_p > root_tol # => non-zero `delta_alpha` required
                # Initial guess for Newton iteration
                delta_alpha_i = use_density_init ? delta_alpha_density :
                                zero(eltype(u_dgfv))

                # Newton's method to solve for pressure `delta_alpha`
                # Calculate ∂p/∂α using the chain rule
                # ∂p/∂α = ∂p/∂u ⋅ ∂u/∂α
                for newton_it in 1:max_iterations
                    # Compute corrected alpha of current (n'th) Newton iteration
                    alpha_n = alpha[element] + delta_alpha_i
                    # Compute ∂u/∂α
                    for v in eachvariable(equations)
                        u_dg_node[v] = compute_pure_dg(u_dgfv_node[v], u_fv_node[v],
                                                       alpha_n)

                        # Derivate follows simply from
                        # u_dgfv = (1 - α) u_dg + α * u_FV
                        du_dalpha_node[v] = u_fv_node[v] - u_dg_node[v]
                    end

                    # Compute ∂p/∂u, ...
                    dp_du_node = gradient_conservative(pressure, u_dgfv_node, equations)
                    # ... and combine with ∂u/∂α to get ∂p/∂α
                    dp_dalpha = dot(dp_du_node, du_dalpha_node)

                    # Avoid division by close to zero derivative
                    if abs(dp_dalpha) < near_zero_tol
                        continue
                    end

                    # Newton update to `delta_alpha_i`
                    # "+" due "-" of Newton formula and "-" in definition of a_p = beta_p * p_fv - p_dgfv
                    delta_alpha_i += damping * a_p / dp_dalpha

                    # Calculate corrected u
                    for v in eachvariable(equations)
                        u_newton_node[v] = u_dgfv_node[v] +
                                           delta_alpha_i * (u_fv_node[v] - u_dg_node[v])
                    end
                    # Compute new pressure value
                    p_newton = pressure(u_newton_node, equations)

                    # Check convergence
                    a_p = beta_p * p_fv - p_newton
                    if a_p <= root_tol
                        break
                    end

                    if newton_it == max_iterations
                        error("RRG Limiter: ($max_iterations) not enough to correct pressure!")
                    end
                end
                delta_alpha = max(delta_alpha, delta_alpha_i)
            end
        end

        # Correct pressure
        correct_u!(u_dgfv, u_fv, u_dg_node,
                   delta_alpha, alpha, alpha_max,
                   element, semi)
    end

    return nothing
end
end # @muladd
