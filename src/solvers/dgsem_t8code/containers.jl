function reinitialize_containers!(mesh::T8codeMesh, equations, dg::DGSEM, cache)
    # Re-initialize elements container.
    @unpack elements = cache
    resize!(elements, ncells(mesh))
    init_elements!(elements, mesh, dg.basis)

    count_required_surfaces!(mesh)

    # Resize interfaces container.
    @unpack interfaces = cache
    resize!(interfaces, mesh.ninterfaces)

    # Resize mortars container.
    @unpack mortars = cache
    resize!(mortars, mesh.nmortars)

    # Resize boundaries container.
    @unpack boundaries = cache
    resize!(boundaries, mesh.nboundaries)

    fill_mesh_info!(mesh, interfaces, mortars, boundaries,
                    mesh.boundary_names)

    return nothing
end

function count_required_surfaces!(mesh::T8codeMesh)
    counts = count_interfaces(mesh)

    mesh.nmortars = counts.mortars
    mesh.ninterfaces = counts.interfaces
    mesh.nboundaries = counts.boundaries

    mesh.nmpimortars = counts.mpi_mortars
    mesh.nmpiinterfaces = counts.mpi_interfaces

    return counts
end

# Compatibility to `dgsem_p4est/containers.jl`.
function count_required_surfaces(mesh::T8codeMesh)
    return (interfaces = mesh.ninterfaces,
            mortars = mesh.nmortars,
            boundaries = mesh.nboundaries,
            mpi_interfaces = mesh.nmpiinterfaces,
            mpi_mortars = mesh.nmpimortars)
end

# Compatibility to `dgsem_p4est/containers.jl`.
function init_interfaces!(interfaces, mesh::T8codeMesh)
    # Already computed. Do nothing.
    return nothing
end

# Compatibility to `dgsem_p4est/containers.jl`.
function init_mortars!(mortars, mesh::T8codeMesh)
    # Already computed. Do nothing.
    return nothing
end

# Compatibility to `dgsem_p4est/containers.jl`.
function init_boundaries!(boundaries, mesh::T8codeMesh)
    # Already computed. Do nothing.
    return nothing
end
