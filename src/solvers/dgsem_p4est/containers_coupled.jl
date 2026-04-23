# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    P4estCoupledMortarContainer{NDIMS, uEltype <: Real, RealT <: Real, NDIMSP1, NDIMSP2, NDIMSP3, uArray <: DenseArray{uEltype, NDIMSP3}, uVector <: DenseVector{uEltype}}

Container for mortars at coupled mesh view boundaries. Similar to `P4estMPIMortarContainer`
but for coupling between mesh views instead of MPI ranks.

Stores solution values and connectivity information for elements at different refinement
levels across mesh view boundaries (hanging nodes).

# Fields
- `u`: Solution values at mortar [small/large side, variable, position, i, j, mortar]
- `local_neighbor_ids`: Element IDs local to each mesh view [mortar][neighbor_indices]
- `local_neighbor_positions`: Position in mortar structure [mortar][positions]
- `global_neighbor_ids`: Element IDs in parent mesh (global) [mortar][neighbor_indices]
- `node_indices`: Face orientation for small and large sides [small/large, mortar]
- `normal_directions`: Normal vectors on small element faces [dimension, i, j, position, mortar]
"""
mutable struct P4estCoupledMortarContainer{NDIMS, uEltype <: Real, RealT <: Real,
                                           NDIMSP1, NDIMSP2, NDIMSP3,
                                           uArray <: DenseArray{uEltype, NDIMSP3},
                                           uVector <: DenseVector{uEltype}}
    u::uArray                                      # [small/large side, variable, position, i, j, mortar]
    local_neighbor_ids::Vector{Vector{Int}}        # [mortar][ids]
    local_neighbor_positions::Vector{Vector{Int}}  # [mortar][positions]
    global_neighbor_ids::Vector{Vector{Int}}       # [mortar][global ids]
    node_indices::Matrix{NTuple{NDIMS, Symbol}}    # [small/large, mortar]
    normal_directions::Array{RealT, NDIMSP2}       # [dimension, i, j, position, mortar]
    # internal `resize!`able storage
    _u::uVector
    _node_indices::Vector{NTuple{NDIMS, Symbol}}
    _normal_directions::Vector{RealT}
end

@inline function ncoupledmortars(coupled_mortars::P4estCoupledMortarContainer)
    length(coupled_mortars.local_neighbor_ids)
end

@inline Base.ndims(::P4estCoupledMortarContainer{NDIMS}) where {NDIMS} = NDIMS

function Base.resize!(coupled_mortars::P4estCoupledMortarContainer, capacity)
    @unpack _u, _node_indices, _normal_directions = coupled_mortars

    n_dims = ndims(coupled_mortars)
    n_nodes = size(coupled_mortars.u, 4)
    n_variables = size(coupled_mortars.u, 2)

    resize!(_u, 2 * n_variables * 2^(n_dims - 1) * n_nodes^(n_dims - 1) * capacity)
    coupled_mortars.u = unsafe_wrap(Array, pointer(_u),
                                    (2, n_variables, 2^(n_dims - 1),
                                     ntuple(_ -> n_nodes, n_dims - 1)..., capacity))

    resize!(coupled_mortars.local_neighbor_ids, capacity)
    resize!(coupled_mortars.local_neighbor_positions, capacity)
    resize!(coupled_mortars.global_neighbor_ids, capacity)

    resize!(_node_indices, 2 * capacity)
    coupled_mortars.node_indices = unsafe_wrap(Array, pointer(_node_indices),
                                               (2, capacity))

    # Normal directions for all positions: small + large
    n_positions = 2^(n_dims - 1) + 1
    resize!(_normal_directions,
            n_dims * n_nodes^(n_dims - 1) * n_positions * capacity)
    coupled_mortars.normal_directions = unsafe_wrap(Array, pointer(_normal_directions),
                                                    (n_dims,
                                                     ntuple(_ -> n_nodes, n_dims - 1)...,
                                                     n_positions, capacity))

    return nothing
end

"""
    init_coupled_mortars(mesh::P4estMeshView, equations, basis, elements)

Initialize coupled mortar container for a mesh view. This function creates an empty
container that will be populated during mesh view extraction.

For the minimal prototype, this returns an empty container. Full initialization
happens in `extract_coupled_mortars` in mesh view extraction.
"""
function init_coupled_mortars(mesh, equations, basis, elements)
    NDIMS = ndims(mesh)
    uEltype = eltype(elements)
    RealT = real(mesh)

    n_nodes = nnodes(basis)
    n_variables = nvariables(equations)

    # Start with zero capacity - will be resized when mortars are found
    _u = Vector{uEltype}(undef, 0)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, n_variables, 2^(NDIMS - 1),
                     ntuple(_ -> n_nodes, NDIMS - 1)..., 0))

    local_neighbor_ids = Vector{Vector{Int}}(undef, 0)
    local_neighbor_positions = Vector{Vector{Int}}(undef, 0)
    global_neighbor_ids = Vector{Vector{Int}}(undef, 0)

    _node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, 0)
    node_indices = unsafe_wrap(Array, pointer(_node_indices), (2, 0))

    _normal_directions = Vector{RealT}(undef, 0)
    # Normal directions for all mortar positions: small elements + large element
    # In 2D: 2 small + 1 large = 3 positions
    # In 3D: 4 small + 1 large = 5 positions
    n_positions = 2^(NDIMS - 1) + 1
    normal_directions = unsafe_wrap(Array, pointer(_normal_directions),
                                    (NDIMS, ntuple(_ -> n_nodes, NDIMS - 1)...,
                                     n_positions, 0))

    NDIMSP1, NDIMSP2, NDIMSP3 = NDIMS + 1, NDIMS + 2, NDIMS + 3
    uArray = typeof(u)
    uVector = typeof(_u)

    return P4estCoupledMortarContainer{NDIMS, uEltype, RealT, NDIMSP1, NDIMSP2,
                                       NDIMSP3, uArray, uVector}(u,
                                                                 local_neighbor_ids,
                                                                 local_neighbor_positions,
                                                                 global_neighbor_ids,
                                                                 node_indices,
                                                                 normal_directions,
                                                                 _u,
                                                                 _node_indices,
                                                                 _normal_directions)
end

# Iterator for coupled mortars
@inline eachcoupledmortar(dg, cache) = Base.OneTo(ncoupledmortars(cache.coupled_mortars))

end # @muladd
