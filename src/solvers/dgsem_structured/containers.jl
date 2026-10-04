# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct StructuredElementContainer{NDIMS, RealT <: Real, uEltype <: Real,
                                  NDIMSP1, NDIMSP2, NDIMSP3} <: AbstractElementContainer
    # Physical coordinates at each node
    node_coordinates::Array{RealT, NDIMSP2} # [orientation, node_i, node_j, node_k, element]

    # Physical coordinates at boundary nodes
    boundary_node_coordinates::Array{RealT, NDIMSP1} # [orientation, node_i, node_j, direction/face]

    # ID of neighbor element in negative direction in orientation
    left_neighbors::Array{Int, 2} # [orientation, elements]

    # Jacobian matrix of the transformation
    # [jacobian_i, jacobian_j, node_i, node_j, node_k, element] where jacobian_i is the first index of the Jacobian matrix
    jacobian_matrix::Array{RealT, NDIMSP3}

    # Contravariant vectors, scaled by J, in Kopriva's blue book called Ja^i_n (i index, n dimension)
    contravariant_vectors::Array{RealT, NDIMSP3} # [dimension, index, node_i, node_j, node_k, element]

    # 1/J where J is the Jacobian determinant (determinant of Jacobian matrix)
    inverse_jacobian::Array{RealT, NDIMSP1} # [node_i, node_j, node_k, element]

    # Buffer for solution values at interfaces (filled by `prolong2interfaces!`)
    interfaces_u::Array{uEltype, NDIMSP2} # [variable, i, j, direction, element]

    # Buffer for calculated surface flux
    surface_flux_values::Array{uEltype, NDIMSP2} # [variable, i, j, direction, element]
end

# Create element container and initialize element data
function init_elements(mesh::Union{StructuredMesh{NDIMS, RealT},
                                   StructuredMeshView{NDIMS, RealT}},
                       equations::AbstractEquations,
                       basis,
                       ::Type{uEltype}) where {NDIMS, RealT <: Real, uEltype <: Real}
    nelements = prod(size(mesh))
    node_coordinates = Array{RealT, NDIMS + 2}(undef, NDIMS,
                                               ntuple(_ -> nnodes(basis), NDIMS)...,
                                               nelements)
    boundary_node_coordinates = Array{RealT, NDIMS + 1}(undef, NDIMS,
                                                        ntuple(_ -> nnodes(basis),
                                                               NDIMS - 1)...,
                                                        NDIMS * 2)
    left_neighbors = Array{Int, 2}(undef, NDIMS, nelements)
    jacobian_matrix = Array{RealT, NDIMS + 3}(undef, NDIMS, NDIMS,
                                              ntuple(_ -> nnodes(basis), NDIMS)...,
                                              nelements)
    contravariant_vectors = similar(jacobian_matrix)
    inverse_jacobian = Array{RealT, NDIMS + 1}(undef,
                                               ntuple(_ -> nnodes(basis), NDIMS)...,
                                               nelements)
    interfaces_u = Array{uEltype, NDIMS + 2}(undef, nvariables(equations),
                                             ntuple(_ -> nnodes(basis),
                                                    NDIMS - 1)..., NDIMS * 2,
                                             nelements)
    surface_flux_values = Array{uEltype, NDIMS + 2}(undef, nvariables(equations),
                                                    ntuple(_ -> nnodes(basis),
                                                           NDIMS - 1)..., NDIMS * 2,
                                                    nelements)

    elements = StructuredElementContainer{NDIMS, RealT, uEltype,
                                          NDIMS + 1, NDIMS + 2, NDIMS + 3}(node_coordinates,
                                                                           boundary_node_coordinates,
                                                                           left_neighbors,
                                                                           jacobian_matrix,
                                                                           contravariant_vectors,
                                                                           inverse_jacobian,
                                                                           interfaces_u,
                                                                           surface_flux_values)

    init_elements!(elements, mesh, basis)
    return elements
end

@inline nelements(elements::StructuredElementContainer) = size(elements.left_neighbors,
                                                               2)

function Base.eltype(::StructuredElementContainer{NDIMS, RealT, uEltype}) where {NDIMS,
                                                                                 RealT,
                                                                                 uEltype
                                                                                 }
    return uEltype
end

# Check whether the arrays in `elements` have the axes we assume it must have in the inner loops
# of Trixi.jl.
function check_axes(elements::StructuredElementContainer{NDIMS}, equations,
                    solver::DG, cache) where {NDIMS}
    node_axes = ntuple(_ -> eachnode(solver), NDIMS)
    surface_node_axes = ntuple(_ -> eachnode(solver), NDIMS - 1)
    check_axes(elements.node_coordinates,
               (Base.OneTo(NDIMS),
                node_axes...,
                eachelement(solver, cache)))
    check_axes(elements.boundary_node_coordinates,
               (Base.OneTo(NDIMS),
                surface_node_axes...,
                Base.OneTo(2 * NDIMS)))
    check_axes(elements.left_neighbors,
               (Base.OneTo(NDIMS), eachelement(solver, cache)))
    check_axes(elements.jacobian_matrix,
               (Base.OneTo(NDIMS), Base.OneTo(NDIMS),
                node_axes...,
                eachelement(solver, cache)))
    check_axes(elements.contravariant_vectors,
               (Base.OneTo(NDIMS), Base.OneTo(NDIMS),
                node_axes...,
                eachelement(solver, cache)))
    check_axes(elements.inverse_jacobian,
               (node_axes...,
                eachelement(solver, cache)))
    surface_axes = (eachvariable(equations),
                    surface_node_axes...,
                    Base.OneTo(2 * NDIMS),
                    eachelement(solver, cache))
    check_axes(elements.interfaces_u, surface_axes)
    check_axes(elements.surface_flux_values, surface_axes)
    return nothing
end

# Essentially equivalent to `get_contravariant_vector` and `get_node_coords`
@inline function get_normal_vector(normal_vectors, indices...)
    # Returns SVector{NDIMS} where NDIMS is 2 or 3.
    # Can be deduced at compile time from (number of dims - 2) from `normal_vectors` since
    # for 2d we have 4 dims (2 two dims for nodes) - 2 => 2
    # and for 3d we have 5 dims (3 three dims for nodes) - 2 = > 3
    return SVector(ntuple(@inline(dim->normal_vectors[dim, indices...]),
                          Val(ndims(normal_vectors) - 2)))
end

@inline storage_type(::AbstractNormalVectorContainer) = Array

include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
end # @muladd
