# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function max_dt(u, t, mesh::TreeMesh{1},
                constant_speed::False, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1 = zero(max_scaled_speed)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            lambda1, = max_abs_speeds(u_node, equations)
            max_lambda1 = max(max_lambda1, lambda1)
        end
        inv_jacobian = cache.elements.inverse_jacobian[element]
        max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_lambda1)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::TreeMesh{1},
                constant_speed::True, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1, = max_abs_speeds(equations)
        inv_jacobian = cache.elements.inverse_jacobian[element]
        max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_lambda1)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::StructuredMesh{1},
                constant_speed::False, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1 = zero(max_scaled_speed)

        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            lambda1, = max_abs_speeds(u_node, equations)

            inv_jacobian = cache.elements.inverse_jacobian[i, element]

            max_lambda1 = max(max_lambda1, inv_jacobian * lambda1)
        end

        max_scaled_speed = max(max_scaled_speed, max_lambda1)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::StructuredMesh{1},
                constant_speed::True, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1, = max_abs_speeds(equations)

        for i in eachnode(dg)
            inv_jacobian = cache.elements.inverse_jacobian[i, element]
            max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_lambda1)
        end
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end
end # @muladd
