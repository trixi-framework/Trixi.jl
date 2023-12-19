function reinitialize_containers!(mesh::ParallelT8codeMesh, equations, dg::DGSEM, cache)
    @unpack elements = cache
    resize!(elements, ncells(mesh))
    init_elements!(elements, mesh, dg.basis)

    count_required_surfaces!(mesh)
    required = count_required_surfaces(mesh)

    @unpack interfaces = cache
    resize!(interfaces, required.interfaces)

    @unpack boundaries = cache
    resize!(boundaries, required.boundaries)

    @unpack mortars = cache
    resize!(mortars, required.mortars)

    @unpack mpi_interfaces = cache
    resize!(mpi_interfaces, required.mpi_interfaces)

    @unpack mpi_mortars = cache
    resize!(mpi_mortars, required.mpi_mortars)

    mpi_mesh_info = (
      mpi_mortars = mpi_mortars,
      mpi_interfaces = mpi_interfaces,

      # Temporary arrays for updating `mpi_cache`.
      global_mortar_ids = fill(UInt64(0), nmpimortars(mpi_mortars)),
      global_interface_ids = fill(UInt64(0), nmpiinterfaces(mpi_interfaces)),
      neighbor_ranks_mortar = Vector{Vector{Int}}(undef, nmpimortars(mpi_mortars)),
      neighbor_ranks_interface = fill(-1, nmpiinterfaces(mpi_interfaces)),
    )

    trixi_t8_fill_mesh_info(mesh, elements, interfaces, mortars, boundaries,
                            mesh.boundary_names; mpi_mesh_info = mpi_mesh_info)

    @unpack mpi_cache = cache
    init_mpi_cache!(mpi_cache, mesh, mpi_mesh_info, nvariables(equations), nnodes(dg), eltype(elements))

    empty!(mpi_mesh_info.global_mortar_ids)
    empty!(mpi_mesh_info.global_interface_ids)
    empty!(mpi_mesh_info.neighbor_ranks_mortar)
    empty!(mpi_mesh_info.neighbor_ranks_interface)

    # Re-initialize and distribute normal directions of MPI mortars; requires
    # MPI communication, so the MPI cache must be re-initialized beforehand.
    init_normal_directions!(mpi_mortars, dg.basis, elements)
    exchange_normal_directions!(mpi_mortars, mpi_cache, mesh, nnodes(dg))

    return nothing
end

# Compatibility to `dgsem_p4est/containers.jl`.
function init_mpi_interfaces!(interfaces, mesh::ParallelT8codeMesh)
    # Do nothing.
    return nothing
end

# Compatibility to `dgsem_p4est/containers.jl`.
function init_mpi_mortars!(mortars, mesh::ParallelT8codeMesh)
    # Do nothing.
    return nothing
end

# Compatibility to `dgsem_p4est/containers_parallel.jl`.
function init_mpi_mortars!(mpi_mortars, mesh::ParallelT8codeMesh, basis, elements)
    # Do nothing.
    return nothing
end
