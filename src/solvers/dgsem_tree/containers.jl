
# Dimension independent code related to containers of the DG solver
# with the mesh type TreeMesh

function reinitialize_containers!(mesh::TreeMesh, equations, dg::DGSEM, cache)
  # Get new list of leaf cells
  leaf_cell_ids = local_leaf_cells(mesh.tree)

  # re-initialize elements container
  @unpack elements = cache
  resize!(elements, length(leaf_cell_ids))
  init_elements!(elements, leaf_cell_ids, mesh, dg.basis)

  # re-initialize interfaces container
  @unpack interfaces = cache
  resize!(interfaces, count_required_interfaces(mesh, leaf_cell_ids))
  init_interfaces!(interfaces, elements, mesh)

  # re-initialize boundaries container
  @unpack boundaries = cache
  resize!(boundaries, count_required_boundaries(mesh, leaf_cell_ids))
  init_boundaries!(boundaries, elements, mesh)

  # re-initialize mortars container
  @unpack mortars = cache
  resize!(mortars, count_required_mortars(mesh, leaf_cell_ids))
  init_mortars!(mortars, elements, mesh)

  if mpi_isparallel()
    # re-initialize mpi_interfaces container
    @unpack mpi_interfaces = cache
    resize!(mpi_interfaces, count_required_mpi_interfaces(mesh, leaf_cell_ids))
    init_mpi_interfaces!(mpi_interfaces, elements, mesh)

    # re-initialize mpi cache
    @unpack mpi_cache = cache
    init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, nvariables(equations), nnodes(dg))
  end
end


# Dimension-specific implementations
include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
