# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function limiter_zhang_shu!(u, threshold::Real, variable,
                            mesh::AbstractMesh{3}, equations, dg::DGSEM, cache)
    @threaded for element in eachelement(dg, cache)
        # determine minimum value
        value_min = typemax(eltype(u))
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            value_min = min(value_min, variable(u_node, equations))
        end

        # detect if limiting is necessary
        value_min < threshold || continue

        u_mean = compute_u_mean(u, element, mesh, equations, dg, cache)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                           equations, dg, i, j, k, element)
        end
    end

    return nothing
end

# Modified version of the limiter used in the refinement step of the AMR callback.
# To ensure admissibility after the refinement step, we compute a joint
# limiting coefficient for all children elements and then limit against the
# admissible mean value of the parent element.
# This strategy is described in Remark 3 of the paper:
# - Arpit Babbar, Praveen Chandrashekar (2025)
#   Lax-Wendroff flux reconstruction on adaptive curvilinear meshes with
#   error based time stepping for hyperbolic conservation laws
#   [doi: 10.1016/j.jcp.2024.113622](https://doi.org/10.1016/j.jcp.2024.113622)
function limiter_zhang_shu!(u, threshold::Real, variable, mesh::AbstractMesh{3},
                            equations, dg::DGSEM, cache,
                            element_ids_new::Vector{Int}, u_mean_refined_elements)
    @assert length(element_ids_new)==size(u_mean_refined_elements, 2) "The length of `element_ids_new` must match the second dimension of `u_mean_refined_elements`."

    # theta_sum = [zero(eltype(u)) for _ in 1:Threads.maxthreadid()]
    @threaded for idx in eachindex(element_ids_new)
        # Get the mean value from the parent element
        u_mean = get_node_vars(u_mean_refined_elements, equations, dg, idx)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = one(eltype(u)) # Limiting coefficient

        # Iterate over the children of the current element to determine a joint limiting coefficient `theta`
        for new_element_id in element_ids_new[idx]:(element_ids_new[idx] + 2^ndims(mesh) - 1)
            # determine minimum value
            value_min = typemax(eltype(u))
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                u_node = get_node_vars(u, equations, dg, i, j, k, new_element_id)
                value_min = min(value_min, variable(u_node, equations))
            end
            value_min < threshold || continue # Detect if limiting is necessary

            theta = min(theta, (value_mean - threshold) / (value_mean - value_min))
        end
        if theta < 0
            error("Negative theta encountered in Zhang-Shu limiter during refinement step: $theta")
        end

        theta < 1 || continue # Check if limiting action is necessary
        # if theta < 1
        #     theta_sum[Threads.threadid()] += theta
        # else
        #     theta_sum[Threads.threadid()] += 1.0
        #     continue # Check if limiting action is necessary
        # end

        # Iterate again over the children to apply joint shifting
        for new_element_id in element_ids_new[idx]:(element_ids_new[idx] + 2^ndims(mesh) - 1)
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                u_node = get_node_vars(u, equations, dg, i, j, k, new_element_id)
                set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                               equations, dg, i, j, k, new_element_id)
            end
        end
    end
    # thread_sum_total = sum(theta_sum)
    # mkpath("out")
    # open("out/shifting_alphas_refined_elements.txt", "a") do f
    #     println(f,
    #             "variable: $variable, n_refined elements: $(length(element_ids_new) * 8), avg: $(thread_sum_total / length(element_ids_new)), ")
    # end

    return nothing
end

# Modified version of the limiter used in the coarsening step of the AMR callback.
# To ensure admissibility after the coarsening step, we apply the limiter to
# the coarsened elements.
function limiter_zhang_shu!(u, threshold::Real, variable,
                            mesh::AbstractMesh{3}, equations, dg::DGSEM, cache,
                            element_ids_new::Vector{Int})
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    # theta_sum = [zero(eltype(u)) for _ in 1:Threads.maxthreadid()]
    # Apply limiter to coarsened elements
    @threaded for element in element_ids_new
        # determine minimum value
        value_min = typemax(eltype(u))
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            value_min = min(value_min, variable(u_node, equations))
        end
        value_min < threshold || continue # Detect if limiting is necessary
        # if value_min >= threshold
        #     theta_sum[Threads.threadid()] += 1.0
        #     continue # Check if limiting action is necessary
        # end

        u_mean = compute_u_mean(u, element, mesh, equations, dg, cache)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)

        if theta < 0
            error("Negative theta encountered in Zhang-Shu limiter during refinement step: $theta")
        end
        # theta_sum[Threads.threadid()] += theta
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                           equations, dg, i, j, k, element)
        end
    end
    # thread_sum_total = sum(theta_sum)
    # open("out/shifting_alphas_coarsened_elements.txt", "a") do f
    #     println(f,
    #             "variable: $variable, n_coarsened elements: $(length(element_ids_new)), avg: $(thread_sum_total / length(element_ids_new)), ")
    # end

    return nothing
end
end # @muladd
