# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function max_dt(u, t, mesh::TreeMesh{3},
                constant_speed::False, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1 = max_lambda2 = max_lambda3 = zero(max_scaled_speed)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            lambda1, lambda2, lambda3 = max_abs_speeds(u_node, equations)
            max_lambda1 = max(max_lambda1, lambda1)
            max_lambda2 = max(max_lambda2, lambda2)
            max_lambda3 = max(max_lambda3, lambda3)
        end
        inv_jacobian = cache.elements.inverse_jacobian[element]
        max_scaled_speed = max(max_scaled_speed,
                               inv_jacobian * (max_lambda1 + max_lambda2 + max_lambda3))
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::TreeMesh{3},
                constant_speed::True, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1, max_lambda2, max_lambda3 = max_abs_speeds(equations)
        inv_jacobian = cache.elements.inverse_jacobian[element]
        max_scaled_speed = max(max_scaled_speed,
                               inv_jacobian * (max_lambda1 + max_lambda2 + max_lambda3))
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::Union{StructuredMesh{3}, P4estMesh{3}, T8codeMesh{3}},
                constant_speed, equations, dg::DG, cache)
    backend = backend_or_nothing(cache.elements)
    _max_dt(backend, u, t, mesh, constant_speed, equations, dg, cache)
end

@inline function _max_dt(backend::Backend, u, t,
                         mesh::Union{StructuredMesh{3}, P4estMesh{3}, T8codeMesh{3}},
                         constant_speed::False, equations, dg::DG, cache)
    @unpack contravariant_vectors, inverse_jacobian = cache.elements
    num_elements = nelements(dg, cache)
    nodes = eachnode(dg)
    kernel! = max_scaled_speed_kernel!(backend)

    max_scaled_speeds = allocate(backend, eltype(t), num_elements)
    kernel!(max_scaled_speeds, u, constant_speed, equations, nodes,
            contravariant_vectors,
            inverse_jacobian; ndrange = num_elements)

    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = max(nextfloat(zero(t)), maximum(max_scaled_speeds))

    return 2 / (nnodes(dg) * max_scaled_speed)
end

@kernel function max_scaled_speed_kernel!(max_scaled_speeds, u,
                                          constant_speed::False, equations, nodes,
                                          contravariant_vectors, inverse_jacobian)
    element = @index(Global, Linear)
    NVARS = Val(nvariables(equations))

    max_lambda1 = max_lambda2 = max_lambda3 = zero(eltype(max_scaled_speeds))
    for k in nodes, j in nodes, i in nodes
        u_node = get_svector(u, NVARS, i, j, k, element)
        lambda1, lambda2, lambda3 = max_abs_speeds(u_node, equations)

        Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                    k, element)
        lambda1_transformed = abs(Ja11 * lambda1 + Ja12 * lambda2 + Ja13 * lambda3)
        Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                    k, element)
        lambda2_transformed = abs(Ja21 * lambda1 + Ja22 * lambda2 + Ja23 * lambda3)
        Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors, i, j,
                                                    k, element)
        lambda3_transformed = abs(Ja31 * lambda1 + Ja32 * lambda2 + Ja33 * lambda3)

        inv_jacobian = abs(inverse_jacobian[i, j, k, element])

        max_lambda1 = max(max_lambda1, inv_jacobian * lambda1_transformed)
        max_lambda2 = max(max_lambda2, inv_jacobian * lambda2_transformed)
        max_lambda3 = max(max_lambda3, inv_jacobian * lambda3_transformed)
    end

    max_scaled_speeds[element] = max_lambda1 + max_lambda2 + max_lambda3
end

@inline function _max_dt(backend::Backend, u, t,
                         mesh::Union{StructuredMesh{3}, P4estMesh{3}, T8codeMesh{3}},
                         constant_speed::True, equations, dg::DG, cache)
    @unpack contravariant_vectors, inverse_jacobian = cache.elements
    num_elements = nelements(dg, cache)
    nodes = eachnode(dg)
    kernel! = max_scaled_speed_kernel!(backend)

    max_lambda1.max_lambda2.max_lambda3 = max_abs_speeds(equations)
    max_scaled_speeds = allocate(backend, eltype(t), num_elements)
    kernel!(max_scaled_speeds, constant_speed, nodes, contravariant_vectors,
            inverse_jacobian, max_lambda1, max_lambda2, max_lambda3;
            ndrange = num_elements)

    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = max(nextfloat(zero(t)), maximum(max_scaled_speeds))

    return 2 / (nnodes(dg) * max_scaled_speed)
end

@kernel function max_scaled_speed_kernel!(max_scaled_speeds,
                                          constant_speed::True, nodes,
                                          contravariant_vectors, inverse_jacobian,
                                          max_lambda1, max_lambda2, max_lambda3)
    element = @index(Global, Linear)
    for k in nodes, j in nodes, i in nodes
        Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                    k, element)
        lambda1_transformed = abs(Ja11 * max_lambda1 + Ja12 * max_lambda2 +
                                  Ja13 * max_lambda3)
        Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                    k, element)
        lambda2_transformed = abs(Ja21 * max_lambda1 + Ja22 * max_lambda2 +
                                  Ja23 * max_lambda3)
        Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors, i, j,
                                                    k, element)
        lambda3_transformed = abs(Ja31 * max_lambda1 + Ja32 * max_lambda2 +
                                  Ja33 * max_lambda3)

        inv_jacobian = abs(inverse_jacobian[i, j, k, element])

        max_scaled_speeds[element] = inv_jacobian *
                                     (lambda1_transformed + lambda2_transformed +
                                      lambda3_transformed)
    end
end

@inline function _max_dt(::Nothing, u, t,
                         mesh::Union{StructuredMesh{3}, P4estMesh{3}, T8codeMesh{3}},
                         constant_speed::False, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    @unpack contravariant_vectors = cache.elements

    for element in eachelement(dg, cache)
        max_lambda1 = max_lambda2 = max_lambda3 = zero(max_scaled_speed)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            lambda1, lambda2, lambda3 = max_abs_speeds(u_node, equations)

            Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                        k, element)
            lambda1_transformed = abs(Ja11 * lambda1 + Ja12 * lambda2 + Ja13 * lambda3)
            Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                        k, element)
            lambda2_transformed = abs(Ja21 * lambda1 + Ja22 * lambda2 + Ja23 * lambda3)
            Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors, i, j,
                                                        k, element)
            lambda3_transformed = abs(Ja31 * lambda1 + Ja32 * lambda2 + Ja33 * lambda3)

            inv_jacobian = abs(cache.elements.inverse_jacobian[i, j, k, element])

            max_lambda1 = max(max_lambda1, inv_jacobian * lambda1_transformed)
            max_lambda2 = max(max_lambda2, inv_jacobian * lambda2_transformed)
            max_lambda3 = max(max_lambda3, inv_jacobian * lambda3_transformed)
        end

        max_scaled_speed = max(max_scaled_speed,
                               max_lambda1 + max_lambda2 + max_lambda3)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

@inline function _max_dt(::Nothing, u, t,
                         mesh::Union{StructuredMesh{3}, P4estMesh{3}, T8codeMesh{3}},
                         constant_speed::True, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    @unpack contravariant_vectors = cache.elements

    max_lambda1, max_lambda2, max_lambda3 = max_abs_speeds(equations)

    for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                        k, element)
            lambda1_transformed = abs(Ja11 * max_lambda1 + Ja12 * max_lambda2 +
                                      Ja13 * max_lambda3)
            Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                        k, element)
            lambda2_transformed = abs(Ja21 * max_lambda1 + Ja22 * max_lambda2 +
                                      Ja23 * max_lambda3)
            Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors, i, j,
                                                        k, element)
            lambda3_transformed = abs(Ja31 * max_lambda1 + Ja32 * max_lambda2 +
                                      Ja33 * max_lambda3)

            inv_jacobian = abs(cache.elements.inverse_jacobian[i, j, k, element])

            max_scaled_speed = max(max_scaled_speed,
                                   inv_jacobian *
                                   (lambda1_transformed + lambda2_transformed +
                                    lambda3_transformed))
        end
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::ParallelP4estMesh{3},
                constant_speed::False, equations, dg::DG, cache)
    # call the method accepting a general `mesh::P4estMesh{3}`
    # TODO: MPI, we should improve this; maybe we should dispatch on `u`
    #       and create some MPI array type, overloading broadcasting and mapreduce etc.
    #       Then, this specific array type should also work well with DiffEq etc.
    dt = invoke(max_dt,
                Tuple{typeof(u), typeof(t), P4estMesh{3},
                      typeof(constant_speed), typeof(equations), typeof(dg),
                      typeof(cache)},
                u, t, mesh, constant_speed, equations, dg, cache)
    dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]

    return dt
end

function max_dt(u, t, mesh::ParallelP4estMesh{3},
                constant_speed::True, equations, dg::DG, cache)
    # call the method accepting a general `mesh::P4estMesh{3}`
    # TODO: MPI, we should improve this; maybe we should dispatch on `u`
    #       and create some MPI array type, overloading broadcasting and mapreduce etc.
    #       Then, this specific array type should also work well with DiffEq etc.
    dt = invoke(max_dt,
                Tuple{typeof(u), typeof(t), P4estMesh{3},
                      typeof(constant_speed), typeof(equations), typeof(dg),
                      typeof(cache)},
                u, t, mesh, constant_speed, equations, dg, cache)
    dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]

    return dt
end

function max_dt(u, t, mesh::ParallelT8codeMesh{3},
                constant_speed::False, equations, dg::DG, cache)
    # call the method accepting a general `mesh::T8codeMesh{3}`
    # TODO: MPI, we should improve this; maybe we should dispatch on `u`
    #       and create some MPI array type, overloading broadcasting and mapreduce etc.
    #       Then, this specific array type should also work well with DiffEq etc.
    dt = invoke(max_dt,
                Tuple{typeof(u), typeof(t), T8codeMesh{3},
                      typeof(constant_speed), typeof(equations), typeof(dg),
                      typeof(cache)},
                u, t, mesh, constant_speed, equations, dg, cache)
    dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]

    return dt
end

function max_dt(u, t, mesh::ParallelT8codeMesh{3},
                constant_speed::True, equations, dg::DG, cache)
    # call the method accepting a general `mesh::T8codeMesh{3}`
    # TODO: MPI, we should improve this; maybe we should dispatch on `u`
    #       and create some MPI array type, overloading broadcasting and mapreduce etc.
    #       Then, this specific array type should also work well with DiffEq etc.
    dt = invoke(max_dt,
                Tuple{typeof(u), typeof(t), T8codeMesh{3},
                      typeof(constant_speed), typeof(equations), typeof(dg),
                      typeof(cache)},
                u, t, mesh, constant_speed, equations, dg, cache)
    dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]

    return dt
end
end # @muladd
