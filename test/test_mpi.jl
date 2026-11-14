# The MPI example tests live in the `test_mpi_*.jl` files and are tagged `:mpi`.
# They are discovered automatically by `@run_package_tests` and run inside the
# `mpiexec`-launched worker process set up in `runtests.jl`. Items that fail
# often on Windows CI are additionally tagged `:mpi_skip_windows`.

@testitem "MPI supporting functionality" setup=[Setup] tags=[:mpi] begin
    using Trixi: Trixi, ode_norm, SVector
    t = 0.5
    let u = 1.0
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
    let u = [1.0, -2.0]
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
    let u = [SVector(1.0, -2.0), SVector(0.5, -0.1)]
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
end

@testitem "MPI communication of wrapped arrays and views of them" setup=[Setup] tags=[:mpi] begin
    # Saving solution and restart files without parallel HDF5 gathers the data of each
    # variable on the root rank by sending strided views `@view(data[v, .., :])`, and
    # rebalancing the solver with AMR sends and receives contiguous views
    # `@view(u[:, .., element])`. Here, we check that MPI.jl communicates these views
    # correctly for all array types returned by `Trixi.wrap_array`.
    using Trixi: Trixi, MPI, FixedSizeArrays, FixedSizeArrayDims, VectorBackedArray,
                 mpi_comm, mpi_rank, mpi_nranks, mpi_root, mpi_isroot
    comm = mpi_comm()
    rank = mpi_rank()
    n_ranks = mpi_nranks()
    n_vars = 3
    n_nodes = 4
    # Different numbers of elements on each rank
    n_elements_by_rank = [r + 2 for r in 0:(n_ranks - 1)]
    n_elements = n_elements_by_rank[rank + 1]
    element_offsets = cumsum(n_elements_by_rank) .- n_elements_by_rank
    node_counts = convert(Vector{Cint}, n_elements_by_rank .* n_nodes^2)
    # Unique value for each variable, node, and (global) element
    function value(v, i, j, global_element)
        return v + 10.0 * i + 100.0 * j + 1000.0 * global_element
    end
    dims = (n_vars, n_nodes, n_nodes, n_elements)
    function init_vector()
        u_ode = Vector{Float64}(undef, prod(dims))
        u = reshape(u_ode, dims)
        for element in 1:n_elements, j in 1:n_nodes, i in 1:n_nodes, v in 1:n_vars
            u[v, i, j, element] = value(v, i, j, element_offsets[rank + 1] + element)
        end
        return u_ode
    end

    # The array types returned by `wrap_array`
    wrappers = (
                # `Vector` with compile-time sizes (`DGSEM`)
                u_ode -> FixedSizeArrays.wrap(FixedSizeArrayDims, u_ode,
                                              (Val(n_vars), Val(n_nodes), Val(n_nodes),
                                               n_elements)),
                # `Vector` with a compile-time number of variables (`FDSBP`)
                u_ode -> FixedSizeArrays.wrap(FixedSizeArrayDims, u_ode,
                                              (Val(n_vars), n_nodes, n_nodes,
                                               n_elements)),
                # `Vector` without compile-time sizes
                u_ode -> Trixi.wrap_storage(u_ode, dims),
                # view of a part of a bigger array, e.g., in coupled semidiscretizations
                u_ode -> Trixi.wrap_storage(view(vcat(u_ode, u_ode),
                                                 (length(u_ode) + 1):(2 * length(u_ode))),
                                            dims))

    @testset "wrapper $k" for (k, wrapper) in enumerate(wrappers)
        u = wrapper(init_vector())
        k == 1 && @test u isa FixedSizeArrayDims{Float64, 4, <:Tuple{Val, Val, Val, Int}}
        k == 2 && @test u isa FixedSizeArrayDims{Float64, 4, <:Tuple{Val, Int, Int, Int}}
        k == 3 && @test u isa VectorBackedArray{Float64, 4}
        k == 4 && @test u isa Base.ReshapedArray

        # Gather strided views as in `save_solution_file_on_root`
        for v in 1:n_vars
            if mpi_isroot()
                recv = fill(NaN, sum(node_counts))
                MPI.Gatherv!(@view(u[v, .., :]), MPI.VBuffer(recv, node_counts),
                             mpi_root(), comm)
                expected = [value(v, i, j, element)
                            for i in 1:n_nodes, j in 1:n_nodes,
                                element in 1:sum(n_elements_by_rank)]
                @test recv == vec(expected)
            else
                MPI.Gatherv!(@view(u[v, .., :]), nothing, mpi_root(), comm)
            end
        end
        # The data is unchanged
        @test u == reshape(init_vector(), dims)

        # Send and receive contiguous views of elements as in `rebalance_solver!`:
        # every rank sends its last element to the next rank, which stores it as its
        # first element.
        dest = mod(rank + 1, n_ranks)
        src = mod(rank - 1, n_ranks)
        requests = Vector{MPI.Request}()
        u_new = wrapper(fill(NaN, prod(dims)))
        push!(requests, MPI.Isend(@view(u[:, .., n_elements]), dest, k, comm))
        push!(requests, MPI.Irecv!(@view(u_new[:, .., 1]), src, k, comm))
        MPI.Waitall(requests, MPI.Status)
        src_element = element_offsets[src + 1] + n_elements_by_rank[src + 1]
        @test u_new[:, :, :, 1] == [value(v, i, j, src_element)
               for v in 1:n_vars, i in 1:n_nodes, j in 1:n_nodes]
        @test all(isnan, @view(u_new[:, .., 2:n_elements]))
    end
end
