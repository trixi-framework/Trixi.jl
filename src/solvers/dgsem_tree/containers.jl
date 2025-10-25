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

    return nothing
end

# Container data structure (structure-of-arrays style) for variables used for IDP limiting
mutable struct ContainerSubcellLimiterIDP{NDIMS, uEltype <: Real, NDIMSP1} <:
               AbstractContainer
    alpha::Array{uEltype, NDIMSP1} # [i, j, k, element]
    variable_bounds::Dict{Symbol, Array{uEltype, NDIMSP1}}
    # internal `resize!`able storage
    _alpha::Vector{uEltype}
    _variable_bounds::Dict{Symbol, Vector{uEltype}}
end

function ContainerSubcellLimiterIDP{NDIMS, uEltype}(capacity::Integer, n_nodes,
                                                    bound_keys) where {NDIMS,
                                                                       uEltype <: Real}
    nan_uEltype = convert(uEltype, NaN)

    # Initialize fields with defaults
    _alpha = fill(nan_uEltype, prod(ntuple(_ -> n_nodes, NDIMS)) * capacity)
    alpha = unsafe_wrap(Array, pointer(_alpha),
                        (ntuple(_ -> n_nodes, NDIMS)..., capacity))

    _variable_bounds = Dict{Symbol, Vector{uEltype}}()
    variable_bounds = Dict{Symbol, Array{uEltype, NDIMS + 1}}()
    for key in bound_keys
        _variable_bounds[key] = fill(nan_uEltype,
                                     prod(ntuple(_ -> n_nodes, NDIMS)) * capacity)
        variable_bounds[key] = unsafe_wrap(Array, pointer(_variable_bounds[key]),
                                           (ntuple(_ -> n_nodes, NDIMS)..., capacity))
    end

    return ContainerSubcellLimiterIDP{NDIMS, uEltype, NDIMS + 1}(alpha,
                                                                 variable_bounds,
                                                                 _alpha,
                                                                 _variable_bounds)
end

@inline nnodes(container::ContainerSubcellLimiterIDP) = size(container.alpha, 1)
@inline Base.ndims(::ContainerSubcellLimiterIDP{NDIMS}) where {NDIMS} = NDIMS

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(container::ContainerSubcellLimiterIDP, capacity)
    n_nodes = nnodes(container)
    n_dims = ndims(container)

    (; _alpha) = container
    resize!(_alpha, prod(ntuple(_ -> n_nodes, n_dims)) * capacity)
    container.alpha = unsafe_wrap(Array, pointer(_alpha),
                                  (ntuple(_ -> n_nodes, n_dims)..., capacity))
    container.alpha .= convert(eltype(container.alpha), NaN)

    (; _variable_bounds) = container
    for (key, _) in _variable_bounds
        resize!(_variable_bounds[key], prod(ntuple(_ -> n_nodes, n_dims)) * capacity)
        container.variable_bounds[key] = unsafe_wrap(Array,
                                                     pointer(_variable_bounds[key]),
                                                     (ntuple(_ -> n_nodes, n_dims)...,
                                                      capacity))
    end

    return nothing
end

# Dimension-specific implementations
include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
end # @muladd
