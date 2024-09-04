# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    partition!(mesh::ParallelTreeMesh, allow_coarsening=true)

Partition `mesh` using a static domain decomposition algorithm
based on leaf cell count and tree structure.
If `allow_coarsening` is `true`, the algorithm will keep leaf cells together
on one rank when needed for local coarsening (i.e. when all children of a cell are leaves).
"""
function partition!(mesh::ParallelTreeMesh; allow_coarsening = true)
    # Determine number of leaf cells per rank
    leaves = leaf_cells(mesh.tree)
    @assert length(leaves)>mpi_nranks() "Too many ranks to properly partition the mesh!"
    n_leaves_per_rank = OffsetArray(fill(div(length(leaves), mpi_nranks()),
                                         mpi_nranks()),
                                    0:(mpi_nranks() - 1))
    for rank in 0:(rem(length(leaves), mpi_nranks()) - 1)
        n_leaves_per_rank[rank] += 1
    end
    @assert sum(n_leaves_per_rank) == length(leaves)

    # Assign MPI ranks to all cells such that all ancestors of each cell - if not yet assigned to a
    # rank - belong to the same rank
    mesh.first_cell_by_rank = similar(n_leaves_per_rank)
    mesh.n_cells_by_rank = similar(n_leaves_per_rank)

    leaf_count = 0
    # Assign first cell to rank 0 (employ depth-first indexing of cells)
    mesh.first_cell_by_rank[0] = 1
    # Iterate over all ranks
    for rank in 0:(mpi_nranks() - 1)
        leaf_count += n_leaves_per_rank[rank]
        last_id = leaves[leaf_count]
        parent_id = mesh.tree.parent_ids[last_id]

        # If coarsening is allowed, we need to make sure that parents of leaves 
        # are on the same rank as the leaves when coarsened.
        if allow_coarsening &&
           # Check if all children of the last parent are leaves
           all(id -> is_leaf(mesh.tree, id), @view mesh.tree.child_ids[:, parent_id]) &&
           rank < length(n_leaves_per_rank) - 1 # Make sure there is another rank

            # To keep children of parent together if they are all leaves,
            # all children are added to this rank
            additional_cells = (last_id + 1):mesh.tree.child_ids[end, parent_id]
            if length(additional_cells) > 0
                last_id = additional_cells[end]

                additional_leaves = count(id -> is_leaf(mesh.tree, id),
                                          additional_cells)
                leaf_count += additional_leaves
                # Add leaves to this rank, remove from next rank
                n_leaves_per_rank[rank] += additional_leaves
                n_leaves_per_rank[rank + 1] -= additional_leaves
            end
        end

        @assert all(n -> n > 0, n_leaves_per_rank) "Too many ranks to properly partition the mesh!"

        mesh.n_cells_by_rank[rank] = last_id - mesh.first_cell_by_rank[rank] + 1
        # Use depth-first indexing of cells again to assign also non leaf cells
        mesh.tree.mpi_ranks[mesh.first_cell_by_rank[rank]:last_id] .= rank

        # Set first cell of next rank
        if rank < length(n_leaves_per_rank) - 1 # Make sure there is another rank
            mesh.first_cell_by_rank[rank + 1] = mesh.first_cell_by_rank[rank] +
                                                mesh.n_cells_by_rank[rank]
        end
    end

    @assert all(x -> x >= 0, mesh.tree.mpi_ranks[1:length(mesh.tree)])
    @assert sum(mesh.n_cells_by_rank) == length(mesh.tree)

    return nothing
end

function get_restart_mesh_filename(restart_filename, mpi_parallel::True)
    # Get directory name
    dirname, _ = splitdir(restart_filename)

    if mpi_isroot()
        # Read mesh filename from restart file
        mesh_file = ""
        h5open(restart_filename, "r") do file
            mesh_file = read(attributes(file)["mesh_file"])
        end

        buffer = Vector{UInt8}(mesh_file)
        MPI.Bcast!(Ref(length(buffer)), mpi_root(), mpi_comm())
        MPI.Bcast!(buffer, mpi_root(), mpi_comm())
    else # non-root ranks
        count = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())
        buffer = Vector{UInt8}(undef, count[])
        MPI.Bcast!(buffer, mpi_root(), mpi_comm())
        mesh_file = String(buffer)
    end

    # Construct and return filename
    return joinpath(dirname, mesh_file)
end
end # @muladd
