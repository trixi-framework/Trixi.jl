function reinitialize_containers!(mesh::T8codeMeshParallel, equations, dg::DGSEM, cache)
    @unpack elements, interfaces, boundaries, mortars, mpi_interfaces, mpi_mortars,
    mpi_cache = cache

    n_cells = ncells(mesh)
    resize!(elements, n_cells)
    init_elements!(elements, mesh, dg.basis)

    # Resize volume integral and related datastructures
    @unpack volume_integral = cache
    resize!(cache, mesh, volume_integral, n_cells)
    init_volume_integral!(cache, mesh, dg, volume_integral, n_cells)

    count_required_surfaces!(mesh)
    required = count_required_surfaces(mesh)

    resize!(interfaces, required.interfaces)

    resize!(boundaries, required.boundaries)

    resize!(mortars, required.mortars)

    resize!(mpi_interfaces, required.mpi_interfaces)

    resize!(mpi_mortars, required.mpi_mortars)

    mpi_mesh_info = (mpi_mortars = mpi_mortars,
                     mpi_interfaces = mpi_interfaces,

                     # Temporary arrays for updating `mpi_cache`.
                     global_mortar_ids = fill(UInt128(0), nmpimortars(mpi_mortars)),
                     global_interface_ids = fill(UInt128(0),
                                                 nmpiinterfaces(mpi_interfaces)),
                     neighbor_ranks_mortar = Vector{Vector{Int}}(undef,
                                                                 nmpimortars(mpi_mortars)),
                     neighbor_ranks_interface = fill(-1, nmpiinterfaces(mpi_interfaces)))

    fill_mesh_info!(mesh, interfaces, mortars, boundaries,
                    mesh.boundary_names; mpi_mesh_info = mpi_mesh_info)

    init_mpi_cache!(mpi_cache, mesh, mpi_mesh_info, nvariables(equations), nnodes(dg),
                    eltype(elements))

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
function init_mpi_interfaces!(interfaces, mesh::T8codeMeshParallel)
    # Do nothing.
    return nothing
end

# Compatibility to `dgsem_p4est/containers.jl`.
function init_mpi_mortars!(mortars, mesh::T8codeMeshParallel)
    # Do nothing.
    return nothing
end

# Compatibility to `dgsem_p4est/containers_parallel.jl`.
function init_mpi_mortars!(mpi_mortars, mesh::T8codeMeshParallel, basis, elements)
    # Do nothing.
    return nothing
end
