# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Container data structure (structure-of-arrays style) for DG elements
mutable struct ElementContainer3D{RealT <: Real, uEltype <: Real} <: AbstractContainer
    inverse_jacobian::Vector{RealT}        # [elements]
    node_coordinates::Array{RealT, 5}      # [orientation, i, j, k, elements]
    surface_flux_values::Array{uEltype, 5} # [variables, i, j, direction, elements]
    cell_ids::Vector{Int}                  # [elements]
    # internal `resize!`able storage
    _node_coordinates::Vector{RealT}
    _surface_flux_values::Vector{uEltype}
end

nvariables(elements::ElementContainer3D) = size(elements.surface_flux_values, 1)
nnodes(elements::ElementContainer3D) = size(elements.node_coordinates, 2)
Base.eltype(elements::ElementContainer3D) = eltype(elements.surface_flux_values)

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(elements::ElementContainer3D, capacity)
    n_nodes = nnodes(elements)
    n_variables = nvariables(elements)
    @unpack _node_coordinates, _surface_flux_values,
    inverse_jacobian, cell_ids = elements

    resize!(inverse_jacobian, capacity)

    resize!(_node_coordinates, 3 * n_nodes * n_nodes * n_nodes * capacity)
    elements.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                            (3, n_nodes, n_nodes, n_nodes, capacity))

    resize!(_surface_flux_values, n_variables * n_nodes * n_nodes * 2 * 3 * capacity)
    elements.surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                               (n_variables, n_nodes, n_nodes, 2 * 3,
                                                capacity))

    resize!(cell_ids, capacity)

    return nothing
end

function ElementContainer3D{RealT, uEltype}(capacity::Integer, n_variables,
                                            n_nodes) where {RealT <: Real,
                                                            uEltype <: Real}
    nan_RealT = convert(RealT, NaN)
    nan_uEltype = convert(uEltype, NaN)

    # Initialize fields with defaults
    inverse_jacobian = fill(nan_RealT, capacity)

    _node_coordinates = fill(nan_RealT, 3 * n_nodes * n_nodes * n_nodes * capacity)
    node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                   (3, n_nodes, n_nodes, n_nodes, capacity))

    _surface_flux_values = fill(nan_uEltype,
                                n_variables * n_nodes * n_nodes * 2 * 3 * capacity)
    surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                      (n_variables, n_nodes, n_nodes, 2 * 3, capacity))

    cell_ids = fill(typemin(Int), capacity)

    return ElementContainer3D{RealT, uEltype}(inverse_jacobian, node_coordinates,
                                              surface_flux_values, cell_ids,
                                              _node_coordinates, _surface_flux_values)
end

# Return number of elements
nelements(elements::ElementContainer3D) = length(elements.cell_ids)
# TODO: Taal performance, 1:nelements(elements) vs. Base.OneTo(nelements(elements))
"""
    eachelement(elements::ElementContainer3D)

Return an iterator over the indices that specify the location in relevant data structures
for the elements in `elements`. 
In particular, not the elements themselves are returned.
"""
@inline eachelement(elements::ElementContainer3D) = Base.OneTo(nelements(elements))
@inline Base.real(elements::ElementContainer3D) = eltype(elements.node_coordinates)

# Create element container and initialize element data
function init_elements(cell_ids, mesh::TreeMesh3D,
                       equations::AbstractEquations{3},
                       basis, ::Type{RealT},
                       ::Type{uEltype}) where {RealT <: Real, uEltype <: Real}
    # Initialize container
    n_elements = length(cell_ids)
    elements = ElementContainer3D{RealT, uEltype}(n_elements, nvariables(equations),
                                                  nnodes(basis))

    init_elements!(elements, cell_ids, mesh, basis)
    return elements
end

function init_elements!(elements, cell_ids, mesh::TreeMesh3D, basis)
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
        for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            elements.node_coordinates[1, i, j, k, element] = (mesh.tree.coordinates[1,
                                                                                    cell_id] +
                                                              jacobian * (nodes[i] -
                                                               reference_offset))
            elements.node_coordinates[2, i, j, k, element] = (mesh.tree.coordinates[2,
                                                                                    cell_id] +
                                                              jacobian * (nodes[j] -
                                                               reference_offset))
            elements.node_coordinates[3, i, j, k, element] = (mesh.tree.coordinates[3,
                                                                                    cell_id] +
                                                              jacobian * (nodes[k] -
                                                               reference_offset))
        end
    end

    return elements
end

# Container data structure (structure-of-arrays style) for DG interfaces
mutable struct InterfaceContainer3D{uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 5}      # [leftright, variables, i, j, interfaces]
    neighbor_ids::Matrix{Int} # [leftright, interfaces]
    orientations::Vector{Int} # [interfaces]
    # internal `resize!`able storage
    _u::Vector{uEltype}
    _neighbor_ids::Vector{Int}
end

nvariables(interfaces::InterfaceContainer3D) = size(interfaces.u, 2)
nnodes(interfaces::InterfaceContainer3D) = size(interfaces.u, 3)
Base.eltype(interfaces::InterfaceContainer3D) = eltype(interfaces.u)

# See explanation of Base.resize! for the element container
function Base.resize!(interfaces::InterfaceContainer3D, capacity)
    n_nodes = nnodes(interfaces)
    n_variables = nvariables(interfaces)
    @unpack _u, _neighbor_ids, orientations = interfaces

    resize!(_u, 2 * n_variables * n_nodes * n_nodes * capacity)
    interfaces.u = unsafe_wrap(Array, pointer(_u),
                               (2, n_variables, n_nodes, n_nodes, capacity))

    resize!(_neighbor_ids, 2 * capacity)
    interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                                          (2, capacity))

    resize!(orientations, capacity)

    return nothing
end

function InterfaceContainer3D{uEltype}(capacity::Integer, n_variables,
                                       n_nodes) where {uEltype <: Real}
    nan = convert(uEltype, NaN)

    # Initialize fields with defaults
    _u = fill(nan, 2 * n_variables * n_nodes * n_nodes * capacity)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, n_variables, n_nodes, n_nodes, capacity))

    _neighbor_ids = fill(typemin(Int), 2 * capacity)
    neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                               (2, capacity))

    orientations = fill(typemin(Int), capacity)

    return InterfaceContainer3D{uEltype}(u, neighbor_ids, orientations,
                                         _u, _neighbor_ids)
end

# Return number of interfaces
ninterfaces(interfaces::InterfaceContainer3D) = length(interfaces.orientations)

# Create interface container and initialize interface data in `elements`.
function init_interfaces(cell_ids, mesh::TreeMesh3D,
                         elements::ElementContainer3D)
    # Initialize container
    n_interfaces = count_required_interfaces(mesh, cell_ids)
    interfaces = InterfaceContainer3D{eltype(elements)}(n_interfaces,
                                                        nvariables(elements),
                                                        nnodes(elements))

    # Connect elements with interfaces
    init_interfaces!(interfaces, elements, mesh)
    return interfaces
end

# Count the number of interfaces that need to be created
function count_required_interfaces(mesh::TreeMesh3D, cell_ids)
    count = 0

    # Iterate over all cells
    for cell_id in cell_ids
        for direction in eachdirection(mesh.tree)
            # Only count interfaces in positive direction to avoid double counting
            if direction % 2 == 1
                continue
            end

            # If no neighbor exists, current cell is small or at boundary and thus we need a mortar
            if !has_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            # Skip if neighbor has children
            neighbor_id = mesh.tree.neighbor_ids[direction, cell_id]
            if has_children(mesh.tree, neighbor_id)
                continue
            end

            count += 1
        end
    end

    return count
end

# Initialize connectivity between elements and interfaces
function init_interfaces!(interfaces, elements, mesh::TreeMesh3D)
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
            if direction % 2 == 1
                continue
            end

            # If no neighbor exists, current cell is small and thus we need a mortar
            if !has_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            # Skip if neighbor has children
            neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
            if has_children(mesh.tree, neighbor_cell_id)
                continue
            end

            # Create interface between elements (1 -> "left" of interface, 2 -> "right" of interface)
            count += 1
            interfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
            interfaces.neighbor_ids[1, count] = element

            # Set orientation (x -> 1, y -> 2, z -> 3)
            if direction in (1, 2)
                interfaces.orientations[count] = 1
            elseif direction in (3, 4)
                interfaces.orientations[count] = 2
            else
                interfaces.orientations[count] = 3
            end
        end
    end

    @assert count==ninterfaces(interfaces) ("Actual interface count ($count) does not match "*
                                            "expectations $(ninterfaces(interfaces))")
end

# Container data structure (structure-of-arrays style) for DG boundaries
mutable struct BoundaryContainer3D{RealT <: Real, uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 5}              # [leftright, variables, i, j, boundaries]
    neighbor_ids::Vector{Int}         # [boundaries]
    orientations::Vector{Int}         # [boundaries]
    neighbor_sides::Vector{Int}       # [boundaries]
    node_coordinates::Array{RealT, 4} # [orientation, i, j, elements]
    n_boundaries_per_direction::SVector{6, Int} # [direction]
    # internal `resize!`able storage
    _u::Vector{uEltype}
    _node_coordinates::Vector{RealT}
end

nvariables(boundaries::BoundaryContainer3D) = size(boundaries.u, 2)
nnodes(boundaries::BoundaryContainer3D) = size(boundaries.u, 3)
Base.eltype(boundaries::BoundaryContainer3D) = eltype(boundaries.u)

# See explanation of Base.resize! for the element container
function Base.resize!(boundaries::BoundaryContainer3D, capacity)
    n_nodes = nnodes(boundaries)
    n_variables = nvariables(boundaries)
    @unpack _u, _node_coordinates,
    neighbor_ids, orientations, neighbor_sides = boundaries

    resize!(_u, 2 * n_variables * n_nodes * n_nodes * capacity)
    boundaries.u = unsafe_wrap(Array, pointer(_u),
                               (2, n_variables, n_nodes, n_nodes, capacity))

    resize!(_node_coordinates, 3 * n_nodes * n_nodes * capacity)
    boundaries.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                              (3, n_nodes, n_nodes, capacity))

    resize!(neighbor_ids, capacity)

    resize!(orientations, capacity)

    resize!(neighbor_sides, capacity)

    return nothing
end

function BoundaryContainer3D{RealT, uEltype}(capacity::Integer, n_variables,
                                             n_nodes) where {RealT <: Real,
                                                             uEltype <: Real}
    nan_RealT = convert(RealT, NaN)
    nan_uEltype = convert(uEltype, NaN)

    # Initialize fields with defaults
    _u = fill(nan_uEltype, 2 * n_variables * n_nodes * n_nodes * capacity)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, n_variables, n_nodes, n_nodes, capacity))

    neighbor_ids = fill(typemin(Int), capacity)

    orientations = fill(typemin(Int), capacity)

    neighbor_sides = fill(typemin(Int), capacity)

    _node_coordinates = fill(nan_RealT, 3 * n_nodes * n_nodes * capacity)
    node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                   (3, n_nodes, n_nodes, capacity))

    n_boundaries_per_direction = SVector(0, 0, 0, 0, 0, 0)

    return BoundaryContainer3D{RealT, uEltype}(u, neighbor_ids, orientations,
                                               neighbor_sides,
                                               node_coordinates,
                                               n_boundaries_per_direction,
                                               _u, _node_coordinates)
end

# Return number of boundaries
nboundaries(boundaries::BoundaryContainer3D) = length(boundaries.orientations)

# Create boundaries container and initialize boundary data in `elements`.
function init_boundaries(cell_ids, mesh::TreeMesh3D,
                         elements::ElementContainer3D)
    # Initialize container
    n_boundaries = count_required_boundaries(mesh, cell_ids)
    boundaries = BoundaryContainer3D{real(elements), eltype(elements)}(n_boundaries,
                                                                       nvariables(elements),
                                                                       nnodes(elements))

    # Connect elements with boundaries
    init_boundaries!(boundaries, elements, mesh)
    return boundaries
end

# Count the number of boundaries that need to be created
function count_required_boundaries(mesh::TreeMesh3D, cell_ids)
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
function init_boundaries!(boundaries, elements, mesh::TreeMesh3D)
    # Reset boundaries count
    count = 0

    # Initialize boundary counts
    counts_per_direction = MVector(0, 0, 0, 0, 0, 0)

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
            if iseven(direction)
                boundaries.neighbor_sides[count] = 1
            else
                boundaries.neighbor_sides[count] = 2
            end

            # Set orientation (x -> 1, y -> 2)
            if direction in (1, 2)
                boundaries.orientations[count] = 1
            elseif direction in (3, 4)
                boundaries.orientations[count] = 2
            else
                boundaries.orientations[count] = 3
            end

            # Store node coordinates
            enc = elements.node_coordinates
            if direction == 1 # -x direction
                boundaries.node_coordinates[:, :, :, count] .= enc[:, 1, :, :, element]
            elseif direction == 2 # +x direction
                boundaries.node_coordinates[:, :, :, count] .= enc[:, end, :, :,
                                                                   element]
            elseif direction == 3 # -y direction
                boundaries.node_coordinates[:, :, :, count] .= enc[:, :, 1, :, element]
            elseif direction == 4 # +y direction
                boundaries.node_coordinates[:, :, :, count] .= enc[:, :, end, :,
                                                                   element]
            elseif direction == 5 # -z direction
                boundaries.node_coordinates[:, :, :, count] .= enc[:, :, :, 1, element]
            elseif direction == 6 # +z direction
                boundaries.node_coordinates[:, :, :, count] .= enc[:, :, :, end,
                                                                   element]
            else
                error("should not happen")
            end
        end
    end

    @assert count==nboundaries(boundaries) ("Actual boundaries count ($count) does not match "*
                                            "expectations $(nboundaries(boundaries))")
    @assert sum(counts_per_direction) == count

    boundaries.n_boundaries_per_direction = SVector(counts_per_direction)

    return SVector(counts_per_direction)
end

# Container data structure (structure-of-arrays style) for DG L2 mortars
# Positions/directions for orientations = 1, large_sides = 2:
# mortar is orthogonal to x-axis, large side is in positive coordinate direction wrt mortar
#   /----------------------------\  /----------------------------\
#   |             |              |  |                            |
#   | upper, left | upper, right |  |                            |
#   |      3      |      4       |  |                            |
#   |             |              |  |           large            |
#   |-------------|--------------|  |             5              |
# z |             |              |  |                            |
#   | lower, left | lower, right |  |                            |
# ^ |      1      |      2       |  |                            |
# | |             |              |  |                            |
# | \----------------------------/  \----------------------------/
# |
# ⋅----> y
# Left and right are always wrt to a coordinate direction:
# * left is always the negative direction
# * right is always the positive direction
#
# Left and right are used *both* for the numbering of the mortar faces *and* for the position of the
# elements with respect to the axis orthogonal to the mortar.
mutable struct L2MortarContainer3D{uEltype <: Real} <: AbstractContainer
    u_upper_left::Array{uEltype, 5}  # [leftright, variables, i, j, mortars]
    u_upper_right::Array{uEltype, 5} # [leftright, variables, i, j, mortars]
    u_lower_left::Array{uEltype, 5}  # [leftright, variables, i, j, mortars]
    u_lower_right::Array{uEltype, 5} # [leftright, variables, i, j, mortars]
    neighbor_ids::Array{Int, 2}      # [position, mortars]
    # Large sides: left -> 1, right -> 2
    large_sides::Vector{Int}  # [mortars]
    orientations::Vector{Int} # [mortars]
    # internal `resize!`able storage
    _u_upper_left::Vector{uEltype}
    _u_upper_right::Vector{uEltype}
    _u_lower_left::Vector{uEltype}
    _u_lower_right::Vector{uEltype}
    _neighbor_ids::Vector{Int}
end

nvariables(mortars::L2MortarContainer3D) = size(mortars.u_upper_left, 2)
nnodes(mortars::L2MortarContainer3D) = size(mortars.u_upper_left, 3)
Base.eltype(mortars::L2MortarContainer3D) = eltype(mortars.u_upper_left)

# See explanation of Base.resize! for the element container
function Base.resize!(mortars::L2MortarContainer3D, capacity)
    n_nodes = nnodes(mortars)
    n_variables = nvariables(mortars)
    @unpack _u_upper_left, _u_upper_right, _u_lower_left, _u_lower_right,
    _neighbor_ids, large_sides, orientations = mortars

    resize!(_u_upper_left, 2 * n_variables * n_nodes * n_nodes * capacity)
    mortars.u_upper_left = unsafe_wrap(Array, pointer(_u_upper_left),
                                       (2, n_variables, n_nodes, n_nodes, capacity))

    resize!(_u_upper_right, 2 * n_variables * n_nodes * n_nodes * capacity)
    mortars.u_upper_right = unsafe_wrap(Array, pointer(_u_upper_right),
                                        (2, n_variables, n_nodes, n_nodes, capacity))

    resize!(_u_lower_left, 2 * n_variables * n_nodes * n_nodes * capacity)
    mortars.u_lower_left = unsafe_wrap(Array, pointer(_u_lower_left),
                                       (2, n_variables, n_nodes, n_nodes, capacity))

    resize!(_u_lower_right, 2 * n_variables * n_nodes * n_nodes * capacity)
    mortars.u_lower_right = unsafe_wrap(Array, pointer(_u_lower_right),
                                        (2, n_variables, n_nodes, n_nodes, capacity))

    resize!(_neighbor_ids, 5 * capacity)
    mortars.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                                       (5, capacity))

    resize!(large_sides, capacity)

    resize!(orientations, capacity)

    return nothing
end

function L2MortarContainer3D{uEltype}(capacity::Integer, n_variables,
                                      n_nodes) where {uEltype <: Real}
    nan = convert(uEltype, NaN)

    # Initialize fields with defaults
    _u_upper_left = fill(nan, 2 * n_variables * n_nodes * n_nodes * capacity)
    u_upper_left = unsafe_wrap(Array, pointer(_u_upper_left),
                               (2, n_variables, n_nodes, n_nodes, capacity))

    _u_upper_right = fill(nan, 2 * n_variables * n_nodes * n_nodes * capacity)
    u_upper_right = unsafe_wrap(Array, pointer(_u_upper_right),
                                (2, n_variables, n_nodes, n_nodes, capacity))

    _u_lower_left = fill(nan, 2 * n_variables * n_nodes * n_nodes * capacity)
    u_lower_left = unsafe_wrap(Array, pointer(_u_lower_left),
                               (2, n_variables, n_nodes, n_nodes, capacity))

    _u_lower_right = fill(nan, 2 * n_variables * n_nodes * n_nodes * capacity)
    u_lower_right = unsafe_wrap(Array, pointer(_u_lower_right),
                                (2, n_variables, n_nodes, n_nodes, capacity))

    _neighbor_ids = fill(typemin(Int), 5 * capacity)
    neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                               (5, capacity))

    large_sides = fill(typemin(Int), capacity)

    orientations = fill(typemin(Int), capacity)

    return L2MortarContainer3D{uEltype}(u_upper_left, u_upper_right,
                                        u_lower_left, u_lower_right,
                                        neighbor_ids, large_sides, orientations,
                                        _u_upper_left, _u_upper_right,
                                        _u_lower_left, _u_lower_right,
                                        _neighbor_ids)
end

# Return number of L2 mortars
nmortars(l2mortars::L2MortarContainer3D) = length(l2mortars.orientations)

# Allow printing container contents
function Base.show(io::IO, ::MIME"text/plain", c::L2MortarContainer3D)
    @nospecialize c # reduce precompilation time

    println(io, '*'^20)
    for idx in CartesianIndices(c.u_upper_left)
        println(io, "c.u_upper_left[$idx] = $(c.u_upper_left[idx])")
    end
    for idx in CartesianIndices(c.u_upper_right)
        println(io, "c.u_upper_right[$idx] = $(c.u_upper_right[idx])")
    end
    for idx in CartesianIndices(c.u_lower_left)
        println(io, "c.u_lower_left[$idx] = $(c.u_lower_left[idx])")
    end
    for idx in CartesianIndices(c.u_lower_right)
        println(io, "c.u_lower_right[$idx] = $(c.u_lower_right[idx])")
    end
    println(io, "transpose(c.neighbor_ids) = $(transpose(c.neighbor_ids))")
    println(io, "c.large_sides = $(c.large_sides)")
    println(io, "c.orientations = $(c.orientations)")
    print(io, '*'^20)
end

# Create mortar container and initialize mortar data in `elements`.
function init_mortars(cell_ids, mesh::TreeMesh3D,
                      elements::ElementContainer3D,
                      mortar::LobattoLegendreMortarL2)
    # Initialize containers
    n_mortars = count_required_mortars(mesh, cell_ids)
    mortars = L2MortarContainer3D{eltype(elements)}(n_mortars, nvariables(elements),
                                                    nnodes(elements))

    # Connect elements with mortars
    init_mortars!(mortars, elements, mesh)
    return mortars
end

# Count the number of mortars that need to be created
function count_required_mortars(mesh::TreeMesh3D, cell_ids)
    count = 0

    # Iterate over all cells and count mortars from perspective of coarse cells
    for cell_id in cell_ids
        for direction in eachdirection(mesh.tree)
            # If no neighbor exists, cell is small with large neighbor or at boundary -> do nothing
            if !has_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            # If neighbor has no children, this is a conforming interface -> do nothing
            neighbor_id = mesh.tree.neighbor_ids[direction, cell_id]
            if !has_children(mesh.tree, neighbor_id)
                continue
            end

            count += 1
        end
    end

    return count
end

# Initialize connectivity between elements and mortars
function init_mortars!(mortars, elements, mesh::TreeMesh3D)
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

        for direction in eachdirection(mesh.tree)
            # If no neighbor exists, cell is small with large neighbor -> do nothing
            if !has_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            # If neighbor has no children, this is a conforming interface -> do nothing
            neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
            if !has_children(mesh.tree, neighbor_cell_id)
                continue
            end

            # Create mortar between elements (3 possible orientations):
            #
            # mortar in x-direction:
            # 1 -> small element in lower, left position  (-y, -z)
            # 2 -> small element in lower, right position (+y, -z)
            # 3 -> small element in upper, left position  (-y, +z)
            # 4 -> small element in upper, right position (+y, +z)
            #
            # mortar in y-direction:
            # 1 -> small element in lower, left position  (-x, -z)
            # 2 -> small element in lower, right position (+x, -z)
            # 3 -> small element in upper, left position  (-x, +z)
            # 4 -> small element in upper, right position (+x, +z)
            #
            # mortar in z-direction:
            # 1 -> small element in lower, left position  (-x, -y)
            # 2 -> small element in lower, right position (+x, -y)
            # 3 -> small element in upper, left position  (-x, +y)
            # 4 -> small element in upper, right position (+x, +y)
            #
            # Always the case:
            # 5 -> large element
            #
            count += 1
            mortars.neighbor_ids[5, count] = element

            # Directions are from the perspective of the large element
            # ("Where are the small elements? Ah, in the ... direction!")
            if direction == 1 # -x
                mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[2,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[4,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[3, count] = c2e[mesh.tree.child_ids[6,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[4, count] = c2e[mesh.tree.child_ids[8,
                                                                         neighbor_cell_id]]
            elseif direction == 2 # +x
                mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[3,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[3, count] = c2e[mesh.tree.child_ids[5,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[4, count] = c2e[mesh.tree.child_ids[7,
                                                                         neighbor_cell_id]]
            elseif direction == 3 # -y
                mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[3,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[4,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[3, count] = c2e[mesh.tree.child_ids[7,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[4, count] = c2e[mesh.tree.child_ids[8,
                                                                         neighbor_cell_id]]
            elseif direction == 4 # +y
                mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[2,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[3, count] = c2e[mesh.tree.child_ids[5,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[4, count] = c2e[mesh.tree.child_ids[6,
                                                                         neighbor_cell_id]]
            elseif direction == 5 # -z
                mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[5,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[6,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[3, count] = c2e[mesh.tree.child_ids[7,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[4, count] = c2e[mesh.tree.child_ids[8,
                                                                         neighbor_cell_id]]
            elseif direction == 6 # +z
                mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[2,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[3, count] = c2e[mesh.tree.child_ids[3,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[4, count] = c2e[mesh.tree.child_ids[4,
                                                                         neighbor_cell_id]]
            else
                error("should not happen")
            end

            # Set large side, which denotes the direction (1 -> negative, 2 -> positive) of the large side
            if iseven(direction)
                mortars.large_sides[count] = 1
            else
                mortars.large_sides[count] = 2
            end

            # Set orientation (x -> 1, y -> 2, z -> 3)
            if direction in (1, 2)
                mortars.orientations[count] = 1
            elseif direction in (3, 4)
                mortars.orientations[count] = 2
            else
                mortars.orientations[count] = 3
            end
        end
    end

    @assert count==nmortars(mortars) ("Actual mortar count ($count) does not match "*
                                      "expectations $(nmortars(mortars))")
end
end # @muladd
