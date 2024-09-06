# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function max_dt(u, t, mesh::TreeMesh{1},
                constant_speed::False, equations,
                dg::DG, cache)
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
        inv_jacobian = cache.elements.inverse_jacobian[element] # Δx
        max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_lambda1)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::TreeMesh{1},
                constant_diffusivity::False, equations,
                equations_parabolic::AbstractEquationsParabolic,
                dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1 = zero(max_scaled_speed)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            lambda1, = max_diffusivity(u_node, equations_parabolic)
            max_lambda1 = max(max_lambda1, lambda1)
        end
        inv_jacobian = cache.elements.inverse_jacobian[element] # Δx
        max_scaled_speed = max(max_scaled_speed, inv_jacobian^2 * max_lambda1)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::TreeMesh{1},
                constant_speed::True, equations,
                dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1, = max_abs_speeds(equations)
        inv_jacobian = cache.elements.inverse_jacobian[element] # Δx
        max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_lambda1)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::TreeMesh{1},
                constant_diffusivity::True, equations,
                equations_parabolic::AbstractEquationsParabolic,
                dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1, = max_diffusivity(equations_parabolic)
        inv_jacobian = cache.elements.inverse_jacobian[element] # Δx
        max_scaled_speed = max(max_scaled_speed, inv_jacobian^2 * max_lambda1)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::StructuredMesh{1},
                constant_speed::False, equations,
                dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1 = zero(max_scaled_speed)

        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            lambda1, = max_abs_speeds(u_node, equations)

            inv_jacobian = cache.elements.inverse_jacobian[i, element] # Δx

            max_lambda1 = max(max_lambda1, inv_jacobian * lambda1)
        end

        max_scaled_speed = max(max_scaled_speed, max_lambda1)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::StructuredMesh{1},
                constant_speed::True, equations,
                dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1, = max_abs_speeds(equations)

        for i in eachnode(dg)
            inv_jacobian = cache.elements.inverse_jacobian[i, element] # Δx
            max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_lambda1)
        end
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

# Note: `max_dt` is not implemented for `StructuredMesh{1}` since 
# for the `StructuredMesh` there is no support of parabolic terms (yet), see the overview in the docs:
# https://trixi-framework.github.io/Trixi.jl/stable/overview/#overview-semidiscretizations
# Thus, there is also no need to implement the `max_dt` function for the `StructuredMesh{1}` case (yet).

end # @muladd
