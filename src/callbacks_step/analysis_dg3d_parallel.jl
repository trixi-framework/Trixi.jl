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

    # Calculate error norms on the CPU, to ensure the order of summation is the same.
    if trixi_backend(u) !== nothing
        node_coordinates = Array(node_coordinates)
        inverse_jacobian = Array(inverse_jacobian)
        u = Array(u)
    end

    # Set up data structures
    l2_error = zero(func(get_node_vars(u, equations, dg, 1, 1, 1, 1), equations))
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

function integrate_via_indices(func::Func, backend::Nothing, u,
                               mesh::Union{P4estMeshParallel{3}, T8codeMeshParallel{3}},
                               equations,
                               dg::DGSEM, cache, args...; normalize = true) where {Func}
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    # Initialize integral with zeros of the right shape
    # Pass `zeros(eltype(u), nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg), 1)`
    # to `func` since `u` might be empty, if the current rank has no elements.
    # See also https://github.com/trixi-framework/Trixi.jl/issues/1096, and
    # https://github.com/trixi-framework/Trixi.jl/pull/2126/files/7cbc57cfcba93e67353566e10fce1f3edac27330#r1814483243.
    integral = zero(func(zeros(eltype(u), nvariables(equations), nnodes(dg), nnodes(dg),
                               nnodes(dg), 1), 1, 1, 1, 1, equations, dg, args...))
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

function integrate_via_indices(func::Func, backend::Backend, u,
                               mesh::Union{P4estMeshParallel{3}, T8codeMeshParallel{3}},
                               equations,
                               dg::DGSEM, cache, args...; normalize = true) where {Func}
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    function local_plus((integral, total_volume), (integral_element, volume_element))
        return (integral + integral_element, total_volume + volume_element)
    end

    # `func(u, 1,1,1,1, equations, dg, args...)` might access GPU memory
    # so we have to rely on the compiler to correctly infer the type of the integral here.
    # TODO: Technically we need device_promote_op here that "infers" the function within the context of the GPU.
    integral0 = zero(Base.promote_op(func, typeof(u), Int, Int, Int, Int,
                                     typeof(equations), typeof(dg),
                                     map(typeof, args)...))
    volume0 = zero(eltype(weights)) * zero(eltype(inverse_jacobian))
    init = neutral = (integral0, volume0)

    # Use quadrature to numerically integrate over entire domain
    num_elements = nelements(dg, cache)
    integral, volume = AcceleratedKernels.mapreduce(local_plus, 1:num_elements,
                                                    backend; init,
                                                    neutral) do element
        # Initialize integral with zeros of the right shape
        local_integral, local_total_volume = neutral

        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            volume_jacobian = abs(inv(inverse_jacobian[i, j, k, element]))
            local_integral += volume_jacobian * weights[i] * weights[j] * weights[k] *
                              func(u, i, j, k, element, equations, dg, args...)
            local_total_volume += volume_jacobian * weights[i] * weights[j] * weights[k]
        end

        return (local_integral, local_total_volume)
    end

    global_integral = MPI.Reduce!(Ref(integral), +, mpi_root(), mpi_comm())
    total_volume = MPI.Reduce(volume, +, mpi_root(), mpi_comm())
    if mpi_isroot()
        integral = convert(typeof(integral), global_integral[])
    else
        integral = convert(typeof(integral), NaN * integral)
        total_volume = volume # non-root processes receive nothing from reduce -> overwrite
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume
    end

    return integral
end
end # @muladd
