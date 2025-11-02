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
        # TODO: Revisit this, maybe just clip at 1 and do not respect alpha_max?
        #if alpha[element] + delta_alpha > alpha_max
        #    delta_alpha = alpha_max - alpha[element]
        if alpha[element] + delta_alpha > 1
            delta_alpha = 1 - alpha[element]
            alpha[element] = 1
        else
            alpha[element] += delta_alpha
        end

        @unpack solver, equations = semi

        for i in eachnode(solver)
            u_fv_node = get_node_vars(u_fv, equations, solver, i, element)
            u_dgfv_node = get_node_vars(u_dgfv, equations, solver, i, element)
            for v in eachvariable(equations)
                # Compute pure DG solution given Hennemann-Gassner blending:
                #        u = (1 - α) u_dg + α * u_FV
                # <=> u_dg = (u - α * u_FV) / (1 - α)
                u_dg_node[v] = (u_dgfv_node[v] - alpha[element] * u_fv_node[v]) /
                               (1 - alpha[element])
                #u_dgfv_node[v] += delta_alpha * (u_fv_node[v] - u_dg_node[v])
                u_dgfv[v, i, element] = u_dgfv_node[v] +
                                        delta_alpha * (u_fv_node[v] - u_dg_node[v])
            end
        end
    end

    return nothing
end

# `mesh` passed in only for dispatch
function limiter_rueda_gassner!(u_dgfv, alpha, mesh::AbstractMesh{1}, semi,
                                limiter!)
    @unpack equations, solver, cache = semi

    @unpack beta, alpha_max, max_iterations, root_tol,
    solver_fv, u_fv_ode,
    u_dg_node, du_dalpha_node, dp_du_node, u_newton_node = limiter!

    u_fv = wrap_array(u_fv_ode, semi)

    @threaded for element in eachelement(solver, cache)
        # if alpha is already maximized, there is no need to correct
        if abs(alpha[element] - alpha_max) < root_tol # TODO: Different tolerance for this
            continue
        end

        # delta_alpha for this element
        delta_alpha = zero(eltype(u_dgfv))

        # Density
        for i in eachnode(solver)
            u_fv_node = get_node_vars(u_fv, equations, solver, i, element)
            rho_fv = density(u_fv_node, equations)
            if rho_fv < 0
                error("Finite-Volume solution produces negative value for density!")
            end

            u_dgfv_node = get_node_vars(u_dgfv, equations, solver, i, element)
            rho_dgfv = density(u_dgfv_node, equations)

            a_rho = beta * rho_fv - rho_dgfv
            if a_rho > root_tol # delta_alpha required # TODO: Different tolerance for this
                # Compute pure DG solution given Hennemann-Gassner blending:
                #        u = (1 - α) u_dg + α * u_FV
                # <=> u_dg = (u - α * u_FV) / (1 - α)
                rho_dg = (rho_dgfv - alpha[element] * rho_fv) / (1 - alpha[element])

                # avoid divison by 0
                if abs(rho_fv - rho_dg) < root_tol # TODO: Different tolerance for this
                    continue
                end

                delta_alpha_i = a_rho / (rho_fv - rho_dg) # TODO: Delta t required here?

                # delta_alpha is calculated for each node, use maximum for the entire element
                delta_alpha = max(delta_alpha, delta_alpha_i)
            end
        end

        # Correct density
        correct_u!(u_dgfv, u_fv, u_dg_node,
                   delta_alpha, alpha, alpha_max,
                   element, semi)

        # Pressure
        delta_alpha = zero(eltype(u_dgfv))
        for i in eachnode(solver)
            u_fv_node = get_node_vars(u_fv, equations, solver, i, element)
            p_fv = pressure(u_fv_node, equations)
            if p_fv < 0
                error("Finite-Volume solution produces negative value for pressure!")
            end

            u_dgfv_node = get_node_vars(u_dgfv, equations, solver, i, element)
            p_dgfv = pressure(u_dgfv_node, equations)

            a_p = beta * p_fv - p_dgfv # This is -g(alpha_new) in the paper, see eq. (15)
            if a_p > root_tol # delta_alpha required # TODO: Different tolerance for this
                # Initial guess for Newton iteration. By using zero, we try using the existing alpha first
                delta_alpha_i = zero(eltype(u_dgfv))

                # Newton's method to solve for pressure delta_alpha alpha_p
                # Calculate ∂p/∂α using the chain rule
                # ∂p/∂α = ∂p/∂u ⋅ ∂u/∂α
                for newton_it in 1:max_iterations
                    # compute  ∂u/∂α
                    for v in eachvariable(equations)
                        # Compute pure DG solution
                        u_dg_node[v] = (u_dgfv_node[v] - alpha[element] * u_fv_node[v]) /
                                       (1 - alpha[element])

                        # Derivate follows simply from
                        # u = (1 - α) u_dg + α * u_FV
                        du_dalpha_node[v] = u_fv_node[v] - u_dg_node[v] # TODO: Delta t here?
                    end

                    # compute ∂p/∂u
                    v1 = velocity(u_dgfv_node, equations)
                    dp_du_node[1] = (equations.gamma - 1) * (0.5 * v1^2)
                    dp_du_node[2] = (equations.gamma - 1) * (-v1)
                    dp_du_node[3] = (equations.gamma - 1)
                    #dp_du_node = pressure_gradient(u_dgfv_node, equations)

                    # CARE: Does this maybe allocate?
                    dp_dalpha = dot(dp_du_node, du_dalpha_node)

                    # avoid divison by 0
                    if abs(dp_dalpha) < root_tol
                        continue
                    end

                    # Newton update to alpha_p.
                    # Use "+" instead of "-" (as in the paper) since we have a_p = -g(alpha_new)
                    delta_alpha_i += alpha_p / dp_dalpha

                    # calc corrected u
                    for v in eachvariable(equations)
                        u_newton_node[v] = u_dgfv_node[v] +
                                           delta_alpha_i * (u_fv_node[v] - u_dg_node[v])
                    end

                    # get new pressure value
                    p_newton = pressure(u_newton_node, equations)

                    # Check convergence
                    alpha_p = beta * p_fv - p_newton
                    if alpha_p <= root_tol
                        break
                    end

                    if newton_it == max_iterations
                        error("Number of iterations ($max_iterations) not enough to correct pressure!")
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
