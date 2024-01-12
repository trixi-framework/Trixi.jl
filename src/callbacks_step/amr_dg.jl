# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Redistribute data for load balancing after partitioning the mesh
function rebalance_solver!(u_ode::AbstractVector,
                           mesh::Union{ParallelP4estMesh, ParallelT8codeMesh},
                           equations,
                           dg::DGSEM, cache, old_global_first_quadrant)

    # MPI ranks are 0-based. This array uses 1-based indices.
    global_first_quadrant = get_global_first_element_ids(mesh)

    if global_first_quadrant[mpi_rank() + 1] ==
       old_global_first_quadrant[mpi_rank() + 1] &&
       global_first_quadrant[mpi_rank() + 2] ==
       old_global_first_quadrant[mpi_rank() + 2]
        # Global ids of first and last local quadrants are the same for newly partitioned mesh so the
        # solver does not need to be rebalanced on this rank.
        # Container init uses all-to-all communication -> reinitialize even if there is nothing to do
        # locally (there are other MPI ranks that need to be rebalanced if this function is called)
        reinitialize_containers!(mesh, equations, dg, cache)
        return
    end
    # Retain current solution data
    old_n_elements = nelements(dg, cache)
    old_u_ode = copy(u_ode)
    GC.@preserve old_u_ode begin # OBS! If we don't GC.@preserve old_u_ode, it might be GC'ed
        # Use `wrap_array_native` instead of `wrap_array` since MPI might not interact
        # nicely with non-base array types
        old_u = wrap_array_native(old_u_ode, mesh, equations, dg, cache)

        @trixi_timeit timer() "reinitialize data structures" begin
            reinitialize_containers!(mesh, equations, dg, cache)
        end

        resize!(u_ode,
                nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
        u = wrap_array_native(u_ode, mesh, equations, dg, cache)

        @trixi_timeit timer() "exchange data" begin
            # Collect MPI requests for MPI_Waitall
            requests = Vector{MPI.Request}()
            # Find elements that will change their rank and send their data to the new rank
            for old_element_id in 1:old_n_elements
                # Get global quad ID of old element; local quad id is element id - 1
                global_quad_id = old_global_first_quadrant[mpi_rank() + 1] +
                                 old_element_id - 1
                if !(global_first_quadrant[mpi_rank() + 1] <= global_quad_id <
                     global_first_quadrant[mpi_rank() + 2])
                    # Send element data to new rank, use global_quad_id as tag (non-blocking)
                    dest = findfirst(r -> global_first_quadrant[r] <= global_quad_id <
                                          global_first_quadrant[r + 1],
                                     1:mpi_nranks()) - 1 # mpi ranks 0-based
                    request = MPI.Isend(@view(old_u[:, .., old_element_id]), dest,
                                        global_quad_id, mpi_comm())
                    push!(requests, request)
                end
            end

            # Loop over all elements in new container and either copy them from old container
            # or receive them with MPI
            for element in eachelement(dg, cache)
                # Get global quad ID of element; local quad id is element id - 1
                global_quad_id = global_first_quadrant[mpi_rank() + 1] + element - 1
                if old_global_first_quadrant[mpi_rank() + 1] <= global_quad_id <
                   old_global_first_quadrant[mpi_rank() + 2]
                    # Quad ids are 0-based, element ids are 1-based, hence add 1
                    old_element_id = global_quad_id -
                                     old_global_first_quadrant[mpi_rank() + 1] + 1
                    # Copy old element data to new element container
                    @views u[:, .., element] .= old_u[:, .., old_element_id]
                else
                    # Receive old element data
                    src = findfirst(r -> old_global_first_quadrant[r] <=
                                         global_quad_id <
                                         old_global_first_quadrant[r + 1],
                                    1:mpi_nranks()) - 1 # mpi ranks 0-based
                    request = MPI.Irecv!(@view(u[:, .., element]), src, global_quad_id,
                                         mpi_comm())
                    push!(requests, request)
                end
            end

            # Wait for all non-blocking MPI send/receive operations to finish
            MPI.Waitall(requests, MPI.Status)
        end
    end # GC.@preserve old_u_ode
end
end # @muladd
