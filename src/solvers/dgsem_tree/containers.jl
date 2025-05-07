# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

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
    if hasproperty(cache, :mortars) # cache_parabolic does not carry mortars
        @unpack mortars = cache
        resize!(mortars, count_required_mortars(mesh, leaf_cell_ids))
        init_mortars!(mortars, elements, mesh)
    end

    # re-initialize auxiliary variables container
    if hasproperty(cache, :aux_vars)
        @unpack aux_vars = cache
        resize!(aux_vars, length(leaf_cell_ids),
                count_required_interfaces(mesh, leaf_cell_ids))
        init_aux_node_vars!(aux_vars, mesh, equations, dg, cache)
        init_auxiliary_surface_node_variables!(aux_vars, mesh, equations, dg,
                                               cache)
    end

    if mpi_isparallel()
        # re-initialize mpi_interfaces container
        @unpack mpi_interfaces = cache
        resize!(mpi_interfaces, count_required_mpi_interfaces(mesh, leaf_cell_ids))
        init_mpi_interfaces!(mpi_interfaces, elements, mesh)

        # re-initialize mpi_mortars container
        @unpack mpi_mortars = cache
        resize!(mpi_mortars, count_required_mpi_mortars(mesh, leaf_cell_ids))
        init_mpi_mortars!(mpi_mortars, elements, mesh)

        # re-initialize mpi cache
        @unpack mpi_cache = cache
        init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, mpi_mortars,
                        nvariables(equations), nnodes(dg), eltype(elements))
    end
end

# Container for storing values of auxiliary variables at volume/surface quadrature nodes
mutable struct AuxiliaryNodeVariablesContainer{NDIMS, uEltype <: Real, NDIMSP2,
                                               AuxiliaryVariables}
    aux_node_vars::Array{uEltype, NDIMSP2}         # [var, i, j, element]
    aux_surface_node_vars::Array{uEltype, NDIMSP2} # [leftright, var, i, interface]

    # internal `resize!`able storage
    _aux_node_vars::Vector{uEltype}
    _aux_surface_node_vars::Vector{uEltype}

    # save initialization function
    auxiliary_field::AuxiliaryVariables
end

nvariables(aux_vars::AuxiliaryNodeVariablesContainer) = size(aux_vars.aux_node_vars,
                                                                        1)
nnodes(aux_vars::AuxiliaryNodeVariablesContainer) = size(aux_vars.aux_node_vars,
                                                                    2)

# Create auxiliary node variable container
function init_aux_node_vars(mesh, equations, solver, cache,
                                       auxiliary_field)
    @unpack elements, interfaces = cache

    n_elements = nelements(elements)
    n_interfaces = ninterfaces(interfaces)
    NDIMS = ndims(mesh)
    uEltype = eltype(elements)
    nan_uEltype = convert(uEltype, NaN)

    _aux_node_vars = fill(nan_uEltype,
                                n_aux_node_vars(equations) *
                                nnodes(solver)^NDIMS * n_elements)
    aux_node_vars = unsafe_wrap(Array, pointer(_aux_node_vars),
                                      (n_aux_node_vars(equations),
                                       ntuple(_ -> nnodes(solver), NDIMS)...,
                                       n_elements))
    _aux_surface_node_vars = fill(nan_uEltype,
                                        2 * n_aux_node_vars(equations) *
                                        nnodes(solver)^(NDIMS - 1) *
                                        n_interfaces)
    aux_surface_node_vars = unsafe_wrap(Array,
                                              pointer(_aux_surface_node_vars),
                                              (2, n_aux_node_vars(equations),
                                               ntuple(_ -> nnodes(solver),
                                                      NDIMS - 1)...,
                                               n_interfaces))

    aux_vars = AuxiliaryNodeVariablesContainer{NDIMS, uEltype, NDIMS + 2,
                                                          typeof(auxiliary_field)}(aux_node_vars,
                                                                                   aux_surface_node_vars,
                                                                                   _aux_node_vars,
                                                                                   _aux_surface_node_vars,
                                                                                   auxiliary_field)

    init_aux_node_vars!(aux_vars, mesh, equations, solver, cache)
    init_auxiliary_surface_node_variables!(aux_vars, mesh, equations, solver,
                                           cache)
    return aux_vars
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(aux_vars::AuxiliaryNodeVariablesContainer{NDIMS},
                      capacity_node_vars, capacity_node_surface_vars) where {NDIMS}
    @unpack _aux_node_vars, _aux_surface_node_vars = aux_vars
    n_nodes = nnodes(aux_vars)
    n_variables = nvariables(aux_vars)

    resize!(_aux_node_vars, n_variables * n_nodes^NDIMS * capacity_node_vars)
    aux_vars.aux_node_vars = unsafe_wrap(Array,
                                                          pointer(_aux_node_vars),
                                                          (n_variables,
                                                           ntuple(_ -> n_nodes,
                                                                  NDIMS)...,
                                                           capacity_node_vars))

    resize!(_aux_surface_node_vars,
            2 * n_variables * n_nodes^(NDIMS - 1) *
            capacity_node_surface_vars)
    aux_vars.aux_surface_node_vars = unsafe_wrap(Array,
                                                                  pointer(_aux_surface_node_vars),
                                                                  (2, n_variables,
                                                                   ntuple(_ -> n_nodes,
                                                                          NDIMS - 1)...,
                                                                   capacity_node_surface_vars))
    return nothing
end

# Dimension-specific implementations
include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
end # @muladd
