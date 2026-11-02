# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_error_norms(func, u, t, analyzer,
                          mesh::Union{P4estMeshParallel{3}, T8codeMeshParallel{3}},
                          equations,
                          initial_condition, dg::DGSEM, cache, cache_analysis)
    @unpack vandermonde, weights = analyzer
    @unpack node_coordinates, inverse_jacobian = cache.elements
    @unpack u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2, jacobian_local, jacobian_tmp1, jacobian_tmp2 = cache_analysis

    # Set up data structures
    l2_error = zero_return_type(func, SVector{nvariables(equations), eltype(u)},
                                typeof(equations))
    linf_error = copy(l2_error)
    volume = zero(eltype(weights)) * zero(eltype(inverse_jacobian))

    # Iterate over all elements for error calculations
    for element in eachelement(dg, cache)
        # Interpolate solution and node locations to analysis nodes
        multiply_dimensionwise!(u_local, vandermonde, view(u, :, :, :, :, element),
                                u_tmp1, u_tmp2)
        multiply_dimensionwise!(x_local, vandermonde,
                                view(node_coordinates, :, :, :, :, element), x_tmp1,
                                x_tmp2)
        multiply_scalar_dimensionwise!(jacobian_local, vandermonde,
                                       inv.(view(inverse_jacobian, :, :, :, element)),
                                       jacobian_tmp1, jacobian_tmp2)

        # Calculate errors at each analysis node
        for k in eachnode(analyzer), j in eachnode(analyzer), i in eachnode(analyzer)
            u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j,
                                                        k), t, equations)
            diff = func(u_exact, equations) -
                   func(get_node_vars(u_local, equations, dg, i, j, k), equations)
            # We take absolute value as we need the Jacobian here for the volume calculation
            abs_jacobian_local_ijk = abs(jacobian_local[i, j, k])

            l2_error += diff .^ 2 *
                        (weights[i] * weights[j] * weights[k] * abs_jacobian_local_ijk)
            linf_error = @. max(linf_error, abs(diff))
            volume += weights[i] * weights[j] * weights[k] * abs_jacobian_local_ijk
        end
    end

    return reduce_error_norms_mpi(l2_error, linf_error, volume)
end

function integrate_via_indices(func::Func, u,
                               mesh::Union{P4estMeshParallel{3}, T8codeMeshParallel{3}},
                               equations,
                               dg::DGSEM, cache, args...; normalize = true) where {Func}
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    # Initialize integral with zeros of the right shape
    integral = zero_return_type(func, typeof(u), Int, Int, Int, Int,
                                typeof(equations), typeof(dg), map(typeof, args)...)
    volume = zero(eltype(weights)) * zero(eltype(inverse_jacobian))

    # Use quadrature to numerically integrate over entire domain
    @batch reduction=((+, integral), (+, volume)) for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            volume_jacobian = abs(inv(inverse_jacobian[i, j, k, element]))
            integral += volume_jacobian * weights[i] * weights[j] * weights[k] *
                        func(u, i, j, k, element, equations, dg, args...)
            volume += volume_jacobian * weights[i] * weights[j] * weights[k]
        end
    end

    return reduce_integral_and_volume_mpi(integral, volume, normalize)
end
end # @muladd
