# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Container data structure (structure-of-arrays style) for DG elements
mutable struct ElementContainer1D{RealT <: Real, uEltype <: Real} <: AbstractContainer
    inverse_jacobian::Vector{RealT}        # [elements]
    node_coordinates::Array{RealT, 3}      # [orientation, i, elements]
    surface_flux_values::Array{uEltype, 3} # [variables, direction, elements]
    cell_ids::Vector{Int}                  # [elements]
    # internal `resize!`able storage
    _node_coordinates::Vector{RealT}
    _surface_flux_values::Vector{uEltype}
end

nvariables(elements::ElementContainer1D) = size(elements.surface_flux_values, 1)
nnodes(elements::ElementContainer1D) = size(elements.node_coordinates, 2)
Base.eltype(elements::ElementContainer1D) = eltype(elements.surface_flux_values)

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(elements::ElementContainer1D, capacity)
    n_nodes = nnodes(elements)
    n_variables = nvariables(elements)
    @unpack _node_coordinates, _surface_flux_values,
    inverse_jacobian, cell_ids = elements

    resize!(inverse_jacobian, capacity)

    resize!(_node_coordinates, 1 * n_nodes * capacity)
    elements.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                            (1, n_nodes, capacity))

    resize!(_surface_flux_values, n_variables * 2 * 1 * capacity)
    elements.surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                               (n_variables, 2 * 1, capacity))

    resize!(cell_ids, capacity)

    return nothing
end

function ElementContainer1D{RealT, uEltype}(capacity::Integer, n_variables,
                                            n_nodes) where {RealT <: Real,
                                                            uEltype <: Real}
    nan_RealT = convert(RealT, NaN)
    nan_uEltype = convert(uEltype, NaN)

    # Initialize fields with defaults
    inverse_jacobian = fill(nan_RealT, capacity)

    _node_coordinates = fill(nan_RealT, 1 * n_nodes * capacity)
    node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                   (1, n_nodes, capacity))

    _surface_flux_values = fill(nan_uEltype, n_variables * 2 * 1 * capacity)
    surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                      (n_variables, 2 * 1, capacity))

    cell_ids = fill(typemin(Int), capacity)

    return ElementContainer1D{RealT, uEltype}(inverse_jacobian, node_coordinates,
                                              surface_flux_values, cell_ids,
                                              _node_coordinates, _surface_flux_values)
end

# Return number of elements
@inline nelements(elements::ElementContainer1D) = length(elements.cell_ids)
# TODO: Taal performance, 1:nelements(elements) vs. Base.OneTo(nelements(elements))
"""
    eachelement(elements::ElementContainer1D)

Return an iterator over the indices that specify the location in relevant data structures
for the elements in `elements`. 
In particular, not the elements themselves are returned.
"""
@inline eachelement(elements::ElementContainer1D) = Base.OneTo(nelements(elements))
@inline Base.real(elements::ElementContainer1D) = eltype(elements.node_coordinates)

# Create element container and initialize element data
function init_elements(cell_ids, mesh::TreeMesh1D,
                       equations::AbstractEquations{1},
                       basis, ::Type{RealT},
                       ::Type{uEltype}) where {RealT <: Real, uEltype <: Real}
    # Initialize container
    n_elements = length(cell_ids)
    elements = ElementContainer1D{RealT, uEltype}(n_elements, nvariables(equations),
                                                  nnodes(basis))

    init_elements!(elements, cell_ids, mesh, basis)
    return elements
end

function init_elements!(elements, cell_ids, mesh::TreeMesh1D, basis)
    nodes = get_nodes(basis)
    # Compute the length of the 1D reference interval by integrating
    # the function with constant value unity on the corresponding
    # element data type (using \circ)
    reference_length = integrate(one ∘ eltype, nodes, basis)
    # Compute the offset of the midpoint of the 1D reference interval
    # (its difference from zero)
    reference_offset = (first(nodes) + last(nodes)) / 2

    # Store cell ids
    elements.cell_ids .= cell_ids

    # Calculate inverse Jacobian and node coordinates
    for element in eachelement(elements)
        # Get cell id
        cell_id = cell_ids[element]

        # Get cell length
        dx = length_at_cell(mesh.tree, cell_id)

        # Calculate inverse Jacobian
        jacobian = dx / reference_length
        elements.inverse_jacobian[element] = inv(jacobian)

        # Calculate node coordinates
        # Note that the `tree_coordinates` are the midpoints of the cells.
        # Hence, we need to add an offset for `nodes` with a midpoint
        # different from zero.
        for i in eachnode(basis)
            elements.node_coordinates[1, i, element] = (mesh.tree.coordinates[1,
                                                                              cell_id] +
                                                        jacobian *
                                                        (nodes[i] - reference_offset))
        end
    end

    return elements
end

# Container data structure (structure-of-arrays style) for DG interfaces
mutable struct InterfaceContainer1D{uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 3}      # [leftright, variables, interfaces]
    neighbor_ids::Matrix{Int} # [leftright, interfaces]
    orientations::Vector{Int} # [interfaces]
    # internal `resize!`able storage
    _u::Vector{uEltype}
    _neighbor_ids::Vector{Int}
end

nvariables(interfaces::InterfaceContainer1D) = size(interfaces.u, 2)
Base.eltype(interfaces::InterfaceContainer1D) = eltype(interfaces.u)

# See explanation of Base.resize! for the element container
function Base.resize!(interfaces::InterfaceContainer1D, capacity)
    n_variables = nvariables(interfaces)
    @unpack _u, _neighbor_ids, orientations = interfaces

    resize!(_u, 2 * n_variables * capacity)
    interfaces.u = unsafe_wrap(Array, pointer(_u),
                               (2, n_variables, capacity))

    resize!(_neighbor_ids, 2 * capacity)
    interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                                          (2, capacity))

    resize!(orientations, capacity)

    return nothing
end

function InterfaceContainer1D{uEltype}(capacity::Integer, n_variables,
                                       n_nodes) where {uEltype <: Real}
    nan = convert(uEltype, NaN)

    # Initialize fields with defaults
    _u = fill(nan, 2 * n_variables * capacity)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, n_variables, capacity))

    _neighbor_ids = fill(typemin(Int), 2 * capacity)
    neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                               (2, capacity))

    orientations = fill(typemin(Int), capacity)

    return InterfaceContainer1D{uEltype}(u, neighbor_ids, orientations,
                                         _u, _neighbor_ids)
end

# Return number of interfaces
@inline ninterfaces(interfaces::InterfaceContainer1D) = length(interfaces.orientations)

# Create interface container and initialize interface data in `elements`.
function init_interfaces(cell_ids, mesh::TreeMesh1D,
                         elements::ElementContainer1D)
    # Initialize container
    n_interfaces = count_required_interfaces(mesh, cell_ids)
    interfaces = InterfaceContainer1D{eltype(elements)}(n_interfaces,
                                                        nvariables(elements),
                                                        nnodes(elements))

    # Connect elements with interfaces
    init_interfaces!(interfaces, elements, mesh)
    return interfaces
end

# Count the number of interfaces that need to be created
function count_required_interfaces(mesh::TreeMesh1D, cell_ids)
    count = 0

    # Iterate over all cells
    for cell_id in cell_ids
        for direction in eachdirection(mesh.tree)
            # Only count interfaces in positive direction to avoid double counting
            if direction == 1
                continue
            end

            # Skip if no neighbor exists
            if !has_any_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            count += 1
        end
    end

    return count
end

# Initialize connectivity between elements and interfaces
function init_interfaces!(interfaces, elements, mesh::TreeMesh1D)
    # Construct cell -> element mapping for easier algorithm implementation
    tree = mesh.tree
    c2e = zeros(Int, length(tree))
    for element in eachelement(elements)
        c2e[elements.cell_ids[element]] = element
    end

    # Reset interface count
    count = 0

    # Iterate over all elements to find neighbors and to connect via interfaces
    for element in eachelement(elements)
        # Get cell id
        cell_id = elements.cell_ids[element]

        # Loop over directions
        for direction in eachdirection(mesh.tree)
            # Only create interfaces in positive direction
            if direction == 1
                continue
            end

            # Skip if no neighbor exists and current cell is not small
            if !has_any_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            count += 1

            if has_neighbor(mesh.tree, cell_id, direction)
                neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
                if has_children(mesh.tree, neighbor_cell_id) # Cell has small neighbor
                    interfaces.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[1,
                                                                                neighbor_cell_id]]
                else # Cell has same refinement level neighbor
                    interfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
                end
            else # Cell is small and has large neighbor
                parent_id = mesh.tree.parent_ids[cell_id]
                neighbor_cell_id = mesh.tree.neighbor_ids[direction, parent_id]
                interfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
            end

            interfaces.neighbor_ids[1, count] = element
            # Set orientation (x -> 1)
            interfaces.orientations[count] = 1
        end
    end

    @assert count==ninterfaces(interfaces) ("Actual interface count ($count) does not match "*
                                            "expectations $(ninterfaces(interfaces))")
end

# Container data structure (structure-of-arrays style) for DG boundaries
mutable struct BoundaryContainer1D{RealT <: Real, uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 3}              # [leftright, variables, boundaries]
    neighbor_ids::Vector{Int}         # [boundaries]
    orientations::Vector{Int}         # [boundaries]
    neighbor_sides::Vector{Int}       # [boundaries]
    node_coordinates::Array{RealT, 2} # [orientation, elements]
    n_boundaries_per_direction::SVector{2, Int} # [direction]
    # internal `resize!`able storage
    _u::Vector{uEltype}
    _node_coordinates::Vector{RealT}
end

nvariables(boundaries::BoundaryContainer1D) = size(boundaries.u, 2)
Base.eltype(boundaries::BoundaryContainer1D) = eltype(boundaries.u)

# See explanation of Base.resize! for the element container
function Base.resize!(boundaries::BoundaryContainer1D, capacity)
    n_variables = nvariables(boundaries)
    @unpack _u, _node_coordinates,
    neighbor_ids, orientations, neighbor_sides = boundaries

    resize!(_u, 2 * n_variables * capacity)
    boundaries.u = unsafe_wrap(Array, pointer(_u),
                               (2, n_variables, capacity))

    resize!(_node_coordinates, 1 * capacity)
    boundaries.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                              (1, capacity))

    resize!(neighbor_ids, capacity)

    resize!(orientations, capacity)

    resize!(neighbor_sides, capacity)

    return nothing
end

function BoundaryContainer1D{RealT, uEltype}(capacity::Integer, n_variables,
                                             n_nodes) where {RealT <: Real,
                                                             uEltype <: Real}
    nan_RealT = convert(RealT, NaN)
    nan_uEltype = convert(uEltype, NaN)

    # Initialize fields with defaults
    _u = fill(nan_uEltype, 2 * n_variables * capacity)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, n_variables, capacity))

    neighbor_ids = fill(typemin(Int), capacity)

    orientations = fill(typemin(Int), capacity)

    neighbor_sides = fill(typemin(Int), capacity)

    _node_coordinates = fill(nan_RealT, 1 * capacity)
    node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                   (1, capacity))

    n_boundaries_per_direction = SVector(0, 0)

    return BoundaryContainer1D{RealT, uEltype}(u, neighbor_ids, orientations,
                                               neighbor_sides,
                                               node_coordinates,
                                               n_boundaries_per_direction,
                                               _u, _node_coordinates)
end

# Return number of boundaries
nboundaries(boundaries::BoundaryContainer1D) = length(boundaries.orientations)

# Create boundaries container and initialize boundary data in `elements`.
function init_boundaries(cell_ids, mesh::TreeMesh1D,
                         elements::ElementContainer1D)
    # Initialize container
    n_boundaries = count_required_boundaries(mesh, cell_ids)
    boundaries = BoundaryContainer1D{real(elements), eltype(elements)}(n_boundaries,
                                                                       nvariables(elements),
                                                                       nnodes(elements))

    # Connect elements with boundaries
    init_boundaries!(boundaries, elements, mesh)
    return boundaries
end

# Count the number of boundaries that need to be created
function count_required_boundaries(mesh::TreeMesh1D, cell_ids)
    count = 0

    # Iterate over all cells
    for cell_id in cell_ids
        for direction in eachdirection(mesh.tree)
            # If neighbor exists, current cell is not at a boundary
            if has_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            # If coarse neighbor exists, current cell is not at a boundary
            if has_coarse_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            # No neighbor exists in this direction -> must be a boundary
            count += 1
        end
    end

    return count
end

# Initialize connectivity between elements and boundaries
function init_boundaries!(boundaries, elements, mesh::TreeMesh1D)
    # Reset boundaries count
    count = 0

    # Initialize boundary counts
    counts_per_direction = MVector(0, 0)

    # OBS! Iterate over directions first, then over elements, and count boundaries in each direction
    # Rationale: This way the boundaries are internally sorted by the directions -x, +x, -y etc.,
    #            obviating the need to store the boundary condition to be applied explicitly.
    # Loop over directions
    for direction in eachdirection(mesh.tree)
        # Iterate over all elements to find missing neighbors and to connect to boundaries
        for element in eachelement(elements)
            # Get cell id
            cell_id = elements.cell_ids[element]

            # If neighbor exists, current cell is not at a boundary
            if has_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            # If coarse neighbor exists, current cell is not at a boundary
            if has_coarse_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            # Create boundary
            count += 1
            counts_per_direction[direction] += 1

            # Set neighbor element id
            boundaries.neighbor_ids[count] = element

            # Set neighbor side, which denotes the direction (1 -> negative, 2 -> positive) of the element
            if direction == 2
                boundaries.neighbor_sides[count] = 1
            else
                boundaries.neighbor_sides[count] = 2
            end

            # Set orientation (x -> 1)
            boundaries.orientations[count] = 1

            # Store node coordinates
            enc = elements.node_coordinates
            if direction == 1 # -x direction
                boundaries.node_coordinates[:, count] .= enc[:, 1, element]
            elseif direction == 2 # +x direction
                boundaries.node_coordinates[:, count] .= enc[:, end, element]
            else
                error("should not happen")
            end
        end
    end

    @assert count==nboundaries(boundaries) ("Actual boundaries count ($count) does not match "*
                                            "expectations $(nboundaries(boundaries))")
    @assert sum(counts_per_direction) == count

    boundaries.n_boundaries_per_direction = SVector(counts_per_direction)

    return boundaries.n_boundaries_per_direction
end
end # @muladd
