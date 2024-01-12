# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Create arrays with DGSEM-specific structure to store the mean values and set them all to 0
function initialize_mean_values(mesh::TreeMesh{2},
                                equations::AbstractCompressibleEulerEquations{2},
                                dg::DGSEM, cache)
    uEltype = eltype(cache.elements)
    v_mean = zeros(uEltype,
                   (ndims(equations), nnodes(dg), nnodes(dg),
                    nelements(cache.elements)))
    c_mean = zeros(uEltype, (nnodes(dg), nnodes(dg), nelements(cache.elements)))
    rho_mean = zeros(uEltype, size(c_mean))
    vorticity_mean = zeros(uEltype, size(c_mean))

    return (; v_mean, c_mean, rho_mean, vorticity_mean)
end

# Create cache which holds the vorticity for the previous time step. This is needed due to the
# trapezoidal rule
function create_cache(::Type{AveragingCallback}, mesh::TreeMesh{2},
                      equations::AbstractCompressibleEulerEquations{2}, dg::DGSEM,
                      cache)
    # Cache vorticity from previous time step
    uEltype = eltype(cache.elements)
    vorticity_prev = zeros(uEltype, (nnodes(dg), nnodes(dg), nelements(cache.elements)))
    return (; vorticity_prev)
end

# Calculate vorticity for the initial solution and store it in the cache
function initialize_cache!(averaging_callback_cache, u,
                           mesh::TreeMesh{2},
                           equations::AbstractCompressibleEulerEquations{2},
                           dg::DGSEM, cache)
    @unpack vorticity_prev = averaging_callback_cache

    # Calculate vorticity for initial solution
    calc_vorticity!(vorticity_prev, u, mesh, equations, dg, cache)

    return nothing
end

# Update mean values using the trapezoidal rule
function calc_mean_values!(mean_values, averaging_callback_cache, u, u_prev,
                           integration_constant,
                           mesh::TreeMesh{2},
                           equations::AbstractCompressibleEulerEquations{2},
                           dg::DGSEM, cache)
    @unpack v_mean, c_mean, rho_mean, vorticity_mean = mean_values
    @unpack vorticity_prev = averaging_callback_cache

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            vorticity = calc_vorticity_node(u, mesh, equations, dg, cache, i, j,
                                            element)
            vorticity_prev_node = vorticity_prev[i, j, element]
            vorticity_prev[i, j, element] = vorticity # Cache current vorticity for the next time step

            u_node_prim = cons2prim(get_node_vars(u, equations, dg, i, j, element),
                                    equations)
            u_prev_node_prim = cons2prim(get_node_vars(u_prev, equations, dg, i, j,
                                                       element), equations)

            rho, v1, v2, p = u_node_prim
            rho_prev, v1_prev, v2_prev, p_prev = u_prev_node_prim

            c = sqrt(equations.gamma * p / rho)
            c_prev = sqrt(equations.gamma * p_prev / rho_prev)

            # Calculate the contribution to the mean values using the trapezoidal rule
            vorticity_mean[i, j, element] += integration_constant *
                                             (vorticity_prev_node + vorticity)
            v_mean[1, i, j, element] += integration_constant * (v1_prev + v1)
            v_mean[2, i, j, element] += integration_constant * (v2_prev + v2)
            c_mean[i, j, element] += integration_constant * (c_prev + c)
            rho_mean[i, j, element] += integration_constant * (rho_prev + rho)
        end
    end

    return nothing
end
end # @muladd
