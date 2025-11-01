# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# 1D version
function correct_u!(u_dgfv::AbstractArray{<:Any, 3}, u_fv, u_dg_node,
                    correction, alpha, alpha_max,
                    element, semi)
    if correction > 0
        if alpha[element] + correction > alpha_max
            correction = alpha_max - alpha[element]
            alpha[element] = alpha_max
        else
            alpha[element] += correction
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
                #u_dgfv_node[v] += correction * (u_fv_node[v] - u_dg_node[v])
                u_dgfv[v, i, element] = u_dgfv_node[v] + correction * (u_fv_node[v] - u_dg_node[v])
            end
        end
    end

    return nothing
end

# `mesh` passed in only for dispatch
function limiter_rueda_gassner!(u_dgfv, alpha, mesh::AbstractMesh{1}, integrator, semi,
                                limiter!, stage)
    @unpack equations, solver, cache = semi

    @unpack beta, alpha_max, max_iterations, root_tol,
    solver_fv, u_fv_ode,
    u_dg_node, du_dalpha_node, dp_du_node, u_newton_node = limiter!

    # pure FV solution for stage s
    compute_u_fv!(limiter!, integrator, stage)

    u_fv = wrap_array(u_fv_ode, semi)

    @threaded for element in eachelement(solver, cache)
        # if alpha is already maximized, there is no need to correct
        if abs(alpha[element] - alpha_max) < root_tol # TODO: Different tolerance for this
            continue
        end

        # Correction for this element
        correction = zero(eltype(u_dgfv))

        # Density
        for i in eachnode(solver)
            u_fv_node = get_node_vars(u_fv, equations, solver, i, element)
            rho_fv = density(u_fv_node, equations)
            if rho_fv < 0
                error("Finite-Volume solution produces negative value for density!")
            end

            u_dgfv_node = get_node_vars(u_dgfv, equations, solver, i, element)
            rho_dgfv = density(u_dgfv_node, equations)

            alpha_rho = beta * rho_fv - rho_dgfv
            if alpha_rho > root_tol # Correction required # TODO: Different tolerance for this
                # Compute pure DG solution given Hennemann-Gassner blending:
                #        u = (1 - α) u_dg + α * u_FV
                # <=> u_dg = (u - α * u_FV) / (1 - α)
                rho_dg = (rho_dgfv - alpha[element] * rho_fv) / (1 - alpha[element])

                # avoid divison by 0
                if abs(rho_fv - rho_dg) < root_tol # TODO: Different tolerance for this
                    continue
                end

                correction_i = alpha_rho / (rho_fv - rho_dg)

                # Correction is calculated for each node, use maximum for the entire element
                correction = max(correction, correction_i)
            end
        end

        # Correct density
        correct_u!(u_dgfv, u_fv, u_dg_node,
                   correction, alpha, alpha_max,
                   element, semi)

        # Pressure
        correction = zero(eltype(u_dgfv))
        for i in eachnode(solver)
            u_fv_node = get_node_vars(u_fv, equations, solver, i, element)
            p_fv = pressure(u_fv_node, equations)
            if p_fv < 0
                error("Finite-Volume solution produces negative value for pressure!")
            end

            u_dgfv_node = get_node_vars(u_dgfv, equations, solver, i, element)
            p_dgfv = pressure(u_dgfv_node, equations)

            alpha_p = beta * p_fv - p_dgfv
            if alpha_p > root_tol # Correction required # TODO: Different tolerance for this
                correction_i = zero(eltype(u_dgfv)) # CARE: Revisit this guy!

                # Newton's method
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
                        du_dalpha_node[v] = u_fv_node[v] - u_dg_node[v]
                    end

                    # compute  ∂p/∂u
                    # TODO: Use `ForwardDiff.derivative` here
                    #=
                    dp_du_node[1] = (equations.gamma - 1) * (0.5 * v1^2)
                    dp_du_node[2] = (equations.gamma - 1) * (-v1)
                    dp_du_node[3] = (equations.gamma - 1)
                    =#
                    dp_du_node = pressure_gradient(u_dgfv_node, equations)

                    # CARE: Does this maybe allocate?
                    dp_dalpha = dot(dp_du_node, du_dalpha_node)

                    # avoid divison by 0
                    if abs(dp_dalpha) < root_tol
                        continue
                    end

                    # TODO: This looks odd (why the += and not only = ?)
                    correction_i += alpha_p / dp_dalpha

                    # calc corrected u in newton stage
                    for v in eachvariable(equations)
                        u_newton_node[v] = u_dgfv_node[v] +
                                           correction_i * (u_fv_node[v] - u_dg_node[v])
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
                correction = max(correction, correction_i)
            end
        end

        # Correct pressure
        correct_u!(u_dgfv, u_fv, u_dg_node,
                   correction, alpha, alpha_max,
                   element, semi)
    end

    return nothing
end
end # @muladd
