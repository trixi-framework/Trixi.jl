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

    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        max_lambda1 = zero(max_scaled_speed)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            lambda1, = max_abs_speeds(u_node, equations)
            max_lambda1 = Base.max(max_lambda1, lambda1)
        end
        inv_jacobian = cache.elements.inverse_jacobian[element] # 2 / Δx
        # Use `Base.max` to prevent silent failures, as `max` from `@fastmath` doesn't propagate
        # `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
        max_scaled_speed = Base.max(max_scaled_speed, inv_jacobian * max_lambda1)
    end

    # Factor 2 cancels with 2 from `inv_jacobian`, resulting in Δx
    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::TreeMesh{1},
                constant_diffusivity::False, equations,
                equations_parabolic::AbstractEquationsParabolic,
                dg::DG, cache)
    # to avoid a division by zero if the diffusivity vanishes everywhere
    max_scaled_speed = nextfloat(zero(t))

    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        max_lambda1 = zero(max_scaled_speed)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            lambda1 = max_diffusivity(u_node, equations_parabolic)
            max_lambda1 = max(max_lambda1, lambda1)
        end
        inv_jacobian = cache.elements.inverse_jacobian[element] # 2 / Δx
        max_scaled_speed = max(max_scaled_speed, inv_jacobian^2 * max_lambda1)
    end

    # Factor 4 cancels with 2^2 from `inv_jacobian^2`, resulting in Δx^2
    return 4 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::TreeMesh{1},
                constant_speed::True, equations,
                dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    max_lambda1, = max_abs_speeds(equations)

    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        inv_jacobian = cache.elements.inverse_jacobian[element] # 2 / Δx
        # Use `Base.max` to prevent silent failures, as `max` from `@fastmath` doesn't propagate
        # `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
        max_scaled_speed = Base.max(max_scaled_speed, inv_jacobian * max_lambda1)
    end

    # Factor 2 cancels with 2 from `inv_jacobian`, resulting in Δx
    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::TreeMesh,
                constant_diffusivity::True, equations,
                equations_parabolic::AbstractEquationsParabolic,
                dg::DG, cache)
    # to avoid a division by zero if the diffusivity vanishes everywhere
    max_scaled_speed = nextfloat(zero(t))

    max_lambda1 = max_diffusivity(equations_parabolic)

    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        inv_jacobian = cache.elements.inverse_jacobian[element] # 2 / Δx
        # Note: For the currently supported parabolic equations
        # Diffusion & Navier-Stokes, we only have one diffusivity,
        # so this is valid for 1D, 2D and 3D.
        max_scaled_speed = max(max_scaled_speed, inv_jacobian^2 * max_lambda1)
    end

    # Factor 4 cancels with 2^2 from `inv_jacobian^2`, resulting in Δx^2
    return 4 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::StructuredMesh{1},
                constant_speed::False, equations,
                dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        max_lambda1 = zero(max_scaled_speed)

        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            lambda1, = max_abs_speeds(u_node, equations)

            inv_jacobian = cache.elements.inverse_jacobian[i, element] # 2 / Δx

            max_lambda1 = Base.max(max_lambda1, inv_jacobian * lambda1)
        end

        # Use `Base.max` to prevent silent failures, as `max` from `@fastmath` doesn't propagate
        # `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
        max_scaled_speed = Base.max(max_scaled_speed, max_lambda1)
    end

    # Factor 2 cancels with 2 from `inv_jacobian`, resulting in Δx
    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::StructuredMesh{1},
                constant_speed::True, equations,
                dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    max_lambda1, = max_abs_speeds(equations)

    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        for i in eachnode(dg)
            inv_jacobian = cache.elements.inverse_jacobian[i, element] # 2 / Δx
            # Use `Base.max` to prevent silent failures, as `max` from `@fastmath` doesn't propagate
            # `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
            max_scaled_speed = Base.max(max_scaled_speed, inv_jacobian * max_lambda1)
        end
    end

    # Factor 2 cancels with 2 from `inv_jacobian`, resulting in Δx
    return 2 / (nnodes(dg) * max_scaled_speed)
end

# Note: `max_dt` is not implemented for `StructuredMesh{1}` and `equations_parabolic` since 
# for the `StructuredMesh` type there is no support of parabolic terms (yet), see the overview in the docs:
# https://trixi-framework.github.io/Trixi.jl/stable/overview/#overview-semidiscretizations

end # @muladd
