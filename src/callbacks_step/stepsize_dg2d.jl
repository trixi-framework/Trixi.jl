# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function max_dt(u, t, mesh::TreeMesh{2},
                constant_speed::False, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1 = max_lambda2 = zero(max_scaled_speed)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            lambda1, lambda2 = max_abs_speeds(u_node, equations)
            max_lambda1 = max(max_lambda1, lambda1)
            max_lambda2 = max(max_lambda2, lambda2)
        end
        inv_jacobian = cache.elements.inverse_jacobian[element]
        max_scaled_speed = max(max_scaled_speed,
                               inv_jacobian * (max_lambda1 + max_lambda2))
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::TreeMesh{2},
                constant_speed::True, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    for element in eachelement(dg, cache)
        max_lambda1, max_lambda2 = max_abs_speeds(equations)
        inv_jacobian = cache.elements.inverse_jacobian[element]
        max_scaled_speed = max(max_scaled_speed,
                               inv_jacobian * (max_lambda1 + max_lambda2))
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::ParallelTreeMesh{2},
                constant_speed::False, equations, dg::DG, cache)
    # call the method accepting a general `mesh::TreeMesh{2}`
    # TODO: MPI, we should improve this; maybe we should dispatch on `u`
    #       and create some MPI array type, overloading broadcasting and mapreduce etc.
    #       Then, this specific array type should also work well with DiffEq etc.
    dt = invoke(max_dt,
                Tuple{typeof(u), typeof(t), TreeMesh{2},
                      typeof(constant_speed), typeof(equations), typeof(dg),
                      typeof(cache)},
                u, t, mesh, constant_speed, equations, dg, cache)
    # Base.min instead of min needed, see comment in src/auxiliary/math.jl
    dt = MPI.Allreduce!(Ref(dt), Base.min, mpi_comm())[]

    return dt
end

function max_dt(u, t, mesh::ParallelTreeMesh{2},
                constant_speed::True, equations, dg::DG, cache)
    # call the method accepting a general `mesh::TreeMesh{2}`
    # TODO: MPI, we should improve this; maybe we should dispatch on `u`
    #       and create some MPI array type, overloading broadcasting and mapreduce etc.
    #       Then, this specific array type should also work well with DiffEq etc.
    dt = invoke(max_dt,
                Tuple{typeof(u), typeof(t), TreeMesh{2},
                      typeof(constant_speed), typeof(equations), typeof(dg),
                      typeof(cache)},
                u, t, mesh, constant_speed, equations, dg, cache)
    # Base.min instead of min needed, see comment in src/auxiliary/math.jl
    dt = MPI.Allreduce!(Ref(dt), Base.min, mpi_comm())[]

    return dt
end

function max_dt(u, t,
                mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2},
                            T8codeMesh{2}, StructuredMeshView{2}},
                constant_speed::False, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    for element in eachelement(dg, cache)
        max_lambda1 = max_lambda2 = zero(max_scaled_speed)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            lambda1, lambda2 = max_abs_speeds(u_node, equations)

            # Local speeds transformed to the reference element
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                  element)
            lambda1_transformed = abs(Ja11 * lambda1 + Ja12 * lambda2)
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                  element)
            lambda2_transformed = abs(Ja21 * lambda1 + Ja22 * lambda2)

            inv_jacobian = abs(inverse_jacobian[i, j, element])

            max_lambda1 = max(max_lambda1, lambda1_transformed * inv_jacobian)
            max_lambda2 = max(max_lambda2, lambda2_transformed * inv_jacobian)
        end

        max_scaled_speed = max(max_scaled_speed, max_lambda1 + max_lambda2)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t,
                mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2},
                            T8codeMesh{2}, StructuredMeshView{2}},
                constant_speed::True, equations, dg::DG, cache)
    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    max_lambda1, max_lambda2 = max_abs_speeds(equations)

    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            # Local speeds transformed to the reference element
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                  element)
            lambda1_transformed = abs(Ja11 * max_lambda1 + Ja12 * max_lambda2)
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                  element)
            lambda2_transformed = abs(Ja21 * max_lambda1 + Ja22 * max_lambda2)

            inv_jacobian = abs(inverse_jacobian[i, j, element])
            max_scaled_speed = max(max_scaled_speed,
                                   inv_jacobian *
                                   (lambda1_transformed + lambda2_transformed))
        end
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u, t, mesh::ParallelP4estMesh{2},
                constant_speed::False, equations, dg::DG, cache)
    # call the method accepting a general `mesh::P4estMesh{2}`
    # TODO: MPI, we should improve this; maybe we should dispatch on `u`
    #       and create some MPI array type, overloading broadcasting and mapreduce etc.
    #       Then, this specific array type should also work well with DiffEq etc.
    dt = invoke(max_dt,
                Tuple{typeof(u), typeof(t), P4estMesh{2},
                      typeof(constant_speed), typeof(equations), typeof(dg),
                      typeof(cache)},
                u, t, mesh, constant_speed, equations, dg, cache)
    # Base.min instead of min needed, see comment in src/auxiliary/math.jl
    dt = MPI.Allreduce!(Ref(dt), Base.min, mpi_comm())[]

    return dt
end

function max_dt(u, t, mesh::ParallelP4estMesh{2},
                constant_speed::True, equations, dg::DG, cache)
    # call the method accepting a general `mesh::P4estMesh{2}`
    # TODO: MPI, we should improve this; maybe we should dispatch on `u`
    #       and create some MPI array type, overloading broadcasting and mapreduce etc.
    #       Then, this specific array type should also work well with DiffEq etc.
    dt = invoke(max_dt,
                Tuple{typeof(u), typeof(t), P4estMesh{2},
                      typeof(constant_speed), typeof(equations), typeof(dg),
                      typeof(cache)},
                u, t, mesh, constant_speed, equations, dg, cache)
    # Base.min instead of min needed, see comment in src/auxiliary/math.jl
    dt = MPI.Allreduce!(Ref(dt), Base.min, mpi_comm())[]

    return dt
end

function max_dt(u, t, mesh::ParallelT8codeMesh{2},
                constant_speed::False, equations, dg::DG, cache)
    # call the method accepting a general `mesh::T8codeMesh{2}`
    # TODO: MPI, we should improve this; maybe we should dispatch on `u`
    #       and create some MPI array type, overloading broadcasting and mapreduce etc.
    #       Then, this specific array type should also work well with DiffEq etc.
    dt = invoke(max_dt,
                Tuple{typeof(u), typeof(t), T8codeMesh{2},
                      typeof(constant_speed), typeof(equations), typeof(dg),
                      typeof(cache)},
                u, t, mesh, constant_speed, equations, dg, cache)
    # Base.min instead of min needed, see comment in src/auxiliary/math.jl
    dt = MPI.Allreduce!(Ref(dt), Base.min, mpi_comm())[]

    return dt
end

function max_dt(u, t, mesh::ParallelT8codeMesh{2},
                constant_speed::True, equations, dg::DG, cache)
    # call the method accepting a general `mesh::T8codeMesh{2}`
    # TODO: MPI, we should improve this; maybe we should dispatch on `u`
    #       and create some MPI array type, overloading broadcasting and mapreduce etc.
    #       Then, this specific array type should also work well with DiffEq etc.
    dt = invoke(max_dt,
                Tuple{typeof(u), typeof(t), T8codeMesh{2},
                      typeof(constant_speed), typeof(equations), typeof(dg),
                      typeof(cache)},
                u, t, mesh, constant_speed, equations, dg, cache)
    # Base.min instead of min needed, see comment in src/auxiliary/math.jl
    dt = MPI.Allreduce!(Ref(dt), Base.min, mpi_comm())[]

    return dt
end
end # @muladd
