# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Container data structure (structure-of-arrays style) for DG elements
mutable struct ElementContainer2D{RealT <: Real, uEltype <: Real} <: AbstractContainer
    inverse_jacobian::Vector{RealT}        # [elements]
    node_coordinates::Array{RealT, 4}      # [orientation, i, j, elements]
    surface_flux_values::Array{uEltype, 4} # [variables, i, direction, elements]
    cell_ids::Vector{Int}                  # [elements]
    # internal `resize!`able storage
    _node_coordinates::Vector{RealT}
    _surface_flux_values::Vector{uEltype}
end

nvariables(elements::ElementContainer2D) = size(elements.surface_flux_values, 1)
nnodes(elements::ElementContainer2D) = size(elements.node_coordinates, 2)
Base.eltype(elements::ElementContainer2D) = eltype(elements.surface_flux_values)

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(elements::ElementContainer2D, capacity)
    n_nodes = nnodes(elements)
    n_variables = nvariables(elements)
    @unpack _node_coordinates, _surface_flux_values,
    inverse_jacobian, cell_ids = elements

    resize!(inverse_jacobian, capacity)

    resize!(_node_coordinates, 2 * n_nodes * n_nodes * capacity)
    elements.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                            (2, n_nodes, n_nodes, capacity))

    resize!(_surface_flux_values, n_variables * n_nodes * 2 * 2 * capacity)
    elements.surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                               (n_variables, n_nodes, 2 * 2, capacity))

    resize!(cell_ids, capacity)

    return nothing
end

function ElementContainer2D{RealT, uEltype}(capacity::Integer, n_variables,
                                            n_nodes) where {RealT <: Real,
                                                            uEltype <: Real}
    nan_RealT = convert(RealT, NaN)
    nan_uEltype = convert(uEltype, NaN)

    # Initialize fields with defaults
    inverse_jacobian = fill(nan_RealT, capacity)

    _node_coordinates = fill(nan_RealT, 2 * n_nodes * n_nodes * capacity)
    node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                   (2, n_nodes, n_nodes, capacity))

    _surface_flux_values = fill(nan_uEltype, n_variables * n_nodes * 2 * 2 * capacity)
    surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                      (n_variables, n_nodes, 2 * 2, capacity))

    cell_ids = fill(typemin(Int), capacity)

    return ElementContainer2D{RealT, uEltype}(inverse_jacobian, node_coordinates,
                                              surface_flux_values, cell_ids,
                                              _node_coordinates, _surface_flux_values)
end

# Return number of elements
@inline nelements(elements::ElementContainer2D) = length(elements.cell_ids)
# TODO: Taal performance, 1:nelements(elements) vs. Base.OneTo(nelements(elements))
"""
    eachelement(elements::ElementContainer2D)

Return an iterator over the indices that specify the location in relevant data structures
for the elements in `elements`.
In particular, not the elements themselves are returned.
"""
@inline eachelement(elements::ElementContainer2D) = Base.OneTo(nelements(elements))
@inline Base.real(elements::ElementContainer2D) = eltype(elements.node_coordinates)

# Create element container and initialize element data
function init_elements(cell_ids, mesh::TreeMesh2D,
                       equations::AbstractEquations{2},
                       basis, ::Type{RealT},
                       ::Type{uEltype}) where {RealT <: Real, uEltype <: Real}
    # Initialize container
    n_elements = length(cell_ids)
    elements = ElementContainer2D{RealT, uEltype}(n_elements, nvariables(equations),
                                                  nnodes(basis))

    init_elements!(elements, cell_ids, mesh, basis)
    return elements
end

function init_elements!(elements, cell_ids, mesh::TreeMesh2D, basis)
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
        for j in eachnode(basis), i in eachnode(basis)
            elements.node_coordinates[1, i, j, element] = (mesh.tree.coordinates[1,
                                                                                 cell_id] +
                                                           jacobian *
                                                           (nodes[i] - reference_offset))
            elements.node_coordinates[2, i, j, element] = (mesh.tree.coordinates[2,
                                                                                 cell_id] +
                                                           jacobian *
                                                           (nodes[j] - reference_offset))
        end
    end

    return elements
end

# Container data structure (structure-of-arrays style) for DG interfaces
mutable struct InterfaceContainer2D{uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 4}        # [leftright, variables, i, interfaces]
    neighbor_ids::Array{Int, 2} # [leftright, interfaces]
    orientations::Vector{Int}   # [interfaces]
    # internal `resize!`able storage
    _u::Vector{uEltype}
    _neighbor_ids::Vector{Int}
end

nvariables(interfaces::InterfaceContainer2D) = size(interfaces.u, 2)
nnodes(interfaces::InterfaceContainer2D) = size(interfaces.u, 3)
Base.eltype(interfaces::InterfaceContainer2D) = eltype(interfaces.u)

# See explanation of Base.resize! for the element container
function Base.resize!(interfaces::InterfaceContainer2D, capacity)
    n_nodes = nnodes(interfaces)
    n_variables = nvariables(interfaces)
    @unpack _u, _neighbor_ids, orientations = interfaces

    resize!(_u, 2 * n_variables * n_nodes * capacity)
    interfaces.u = unsafe_wrap(Array, pointer(_u),
                               (2, n_variables, n_nodes, capacity))

    resize!(_neighbor_ids, 2 * capacity)
    interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                                          (2, capacity))

    resize!(orientations, capacity)

    return nothing
end

function InterfaceContainer2D{uEltype}(capacity::Integer, n_variables,
                                       n_nodes) where {uEltype <: Real}
    nan = convert(uEltype, NaN)

    # Initialize fields with defaults
    _u = fill(nan, 2 * n_variables * n_nodes * capacity)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, n_variables, n_nodes, capacity))

    _neighbor_ids = fill(typemin(Int), 2 * capacity)
    neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                               (2, capacity))

    orientations = fill(typemin(Int), capacity)

    return InterfaceContainer2D{uEltype}(u, neighbor_ids, orientations,
                                         _u, _neighbor_ids)
end

# Return number of interfaces
@inline ninterfaces(interfaces::InterfaceContainer2D) = length(interfaces.orientations)

# Create interface container and initialize interface data in `elements`.
function init_interfaces(cell_ids, mesh::TreeMesh2D,
                         elements::ElementContainer2D)
    # Initialize container
    n_interfaces = count_required_interfaces(mesh, cell_ids)
    interfaces = InterfaceContainer2D{eltype(elements)}(n_interfaces,
                                                        nvariables(elements),
                                                        nnodes(elements))

    # Connect elements with interfaces
    init_interfaces!(interfaces, elements, mesh)
    return interfaces
end

# Count the number of interfaces that need to be created
function count_required_interfaces(mesh::TreeMesh2D, cell_ids)
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
            neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
            if has_children(mesh.tree, neighbor_cell_id)
                continue
            end

            # Skip if neighbor is on different rank -> create MPI interface instead
            if mpi_isparallel() && !is_own_cell(mesh.tree, neighbor_cell_id)
                continue
            end

            count += 1
        end
    end

    return count
end

# Initialize connectivity between elements and interfaces
function init_interfaces!(interfaces, elements, mesh::TreeMesh2D)
    # Exit early if there are no interfaces to initialize
    if ninterfaces(interfaces) == 0
        return nothing
    end

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

            # Skip if neighbor is on different rank -> create MPI interface instead
            if mpi_isparallel() && !is_own_cell(mesh.tree, neighbor_cell_id)
                continue
            end

            # Create interface between elements (1 -> "left" of interface, 2 -> "right" of interface)
            count += 1
            interfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
            interfaces.neighbor_ids[1, count] = element

            # Set orientation (x -> 1, y -> 2)
            interfaces.orientations[count] = div(direction, 2)
        end
    end

    @assert count==ninterfaces(interfaces) ("Actual interface count ($count) does not match "*
                                            "expectations $(ninterfaces(interfaces))")
end

# Container data structure (structure-of-arrays style) for DG boundaries
mutable struct BoundaryContainer2D{RealT <: Real, uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 4}              # [leftright, variables, i, boundaries]
    neighbor_ids::Vector{Int}         # [boundaries]
    orientations::Vector{Int}         # [boundaries]
    neighbor_sides::Vector{Int}       # [boundaries]
    node_coordinates::Array{RealT, 3} # [orientation, i, elements]
    n_boundaries_per_direction::SVector{4, Int} # [direction]
    # internal `resize!`able storage
    _u::Vector{uEltype}
    _node_coordinates::Vector{RealT}
end

nvariables(boundaries::BoundaryContainer2D) = size(boundaries.u, 2)
nnodes(boundaries::BoundaryContainer2D) = size(boundaries.u, 3)
Base.eltype(boundaries::BoundaryContainer2D) = eltype(boundaries.u)

# See explanation of Base.resize! for the element container
function Base.resize!(boundaries::BoundaryContainer2D, capacity)
    n_nodes = nnodes(boundaries)
    n_variables = nvariables(boundaries)
    @unpack _u, _node_coordinates,
    neighbor_ids, orientations, neighbor_sides = boundaries

    resize!(_u, 2 * n_variables * n_nodes * capacity)
    boundaries.u = unsafe_wrap(Array, pointer(_u),
                               (2, n_variables, n_nodes, capacity))

    resize!(_node_coordinates, 2 * n_nodes * capacity)
    boundaries.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                              (2, n_nodes, capacity))

    resize!(neighbor_ids, capacity)

    resize!(orientations, capacity)

    resize!(neighbor_sides, capacity)

    return nothing
end

function BoundaryContainer2D{RealT, uEltype}(capacity::Integer, n_variables,
                                             n_nodes) where {RealT <: Real,
                                                             uEltype <: Real}
    nan_RealT = convert(RealT, NaN)
    nan_uEltype = convert(uEltype, NaN)

    # Initialize fields with defaults
    _u = fill(nan_uEltype, 2 * n_variables * n_nodes * capacity)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, n_variables, n_nodes, capacity))

    neighbor_ids = fill(typemin(Int), capacity)

    orientations = fill(typemin(Int), capacity)

    neighbor_sides = fill(typemin(Int), capacity)

    _node_coordinates = fill(nan_RealT, 2 * n_nodes * capacity)
    node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                   (2, n_nodes, capacity))

    n_boundaries_per_direction = SVector(0, 0, 0, 0)

    return BoundaryContainer2D{RealT, uEltype}(u, neighbor_ids, orientations,
                                               neighbor_sides,
                                               node_coordinates,
                                               n_boundaries_per_direction,
                                               _u, _node_coordinates)
end

# Return number of boundaries
@inline nboundaries(boundaries::BoundaryContainer2D) = length(boundaries.orientations)

# Create boundaries container and initialize boundary data in `elements`.
function init_boundaries(cell_ids, mesh::TreeMesh2D,
                         elements::ElementContainer2D)
    # Initialize container
    n_boundaries = count_required_boundaries(mesh, cell_ids)
    boundaries = BoundaryContainer2D{real(elements), eltype(elements)}(n_boundaries,
                                                                       nvariables(elements),
                                                                       nnodes(elements))

    # Connect elements with boundaries
    init_boundaries!(boundaries, elements, mesh)
    return boundaries
end

# Count the number of boundaries that need to be created
function count_required_boundaries(mesh::TreeMesh2D, cell_ids)
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
function init_boundaries!(boundaries, elements, mesh::TreeMesh2D)
    # Exit early if there are no boundaries to initialize
    if nboundaries(boundaries) == 0
        # In this case n_boundaries_per_direction still needs to be reset!
        boundaries.n_boundaries_per_direction = SVector(0, 0, 0, 0)
        return nothing
    end

    # Reset boundaries count
    count = 0

    # Initialize boundary counts
    counts_per_direction = MVector(0, 0, 0, 0)

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
            else
                boundaries.orientations[count] = 2
            end

            # Store node coordinates
            enc = elements.node_coordinates
            if direction == 1 # -x direction
                boundaries.node_coordinates[:, :, count] .= enc[:, 1, :, element]
            elseif direction == 2 # +x direction
                boundaries.node_coordinates[:, :, count] .= enc[:, end, :, element]
            elseif direction == 3 # -y direction
                boundaries.node_coordinates[:, :, count] .= enc[:, :, 1, element]
            elseif direction == 4 # +y direction
                boundaries.node_coordinates[:, :, count] .= enc[:, :, end, element]
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

# Container data structure (structure-of-arrays style) for DG L2 mortars
# Positions/directions for orientations = 1, large_sides = 2:
# mortar is orthogonal to x-axis, large side is in positive coordinate direction wrt mortar
#           |    |
# upper = 2 |    |
#           |    |
#                | 3 = large side
#           |    |
# lower = 1 |    |
#           |    |
mutable struct L2MortarContainer2D{uEltype <: Real} <: AbstractContainer
    u_upper::Array{uEltype, 4}  # [leftright, variables, i, mortars]
    u_lower::Array{uEltype, 4}  # [leftright, variables, i, mortars]
    neighbor_ids::Array{Int, 2} # [position, mortars]
    # Large sides: left -> 1, right -> 2
    large_sides::Vector{Int}  # [mortars]
    orientations::Vector{Int} # [mortars]
    # internal `resize!`able storage
    _u_upper::Vector{uEltype}
    _u_lower::Vector{uEltype}
    _neighbor_ids::Vector{Int}
end

nvariables(mortars::L2MortarContainer2D) = size(mortars.u_upper, 2)
nnodes(mortars::L2MortarContainer2D) = size(mortars.u_upper, 3)
Base.eltype(mortars::L2MortarContainer2D) = eltype(mortars.u_upper)

# See explanation of Base.resize! for the element container
function Base.resize!(mortars::L2MortarContainer2D, capacity)
    n_nodes = nnodes(mortars)
    n_variables = nvariables(mortars)
    @unpack _u_upper, _u_lower, _neighbor_ids,
    large_sides, orientations = mortars

    resize!(_u_upper, 2 * n_variables * n_nodes * capacity)
    mortars.u_upper = unsafe_wrap(Array, pointer(_u_upper),
                                  (2, n_variables, n_nodes, capacity))

    resize!(_u_lower, 2 * n_variables * n_nodes * capacity)
    mortars.u_lower = unsafe_wrap(Array, pointer(_u_lower),
                                  (2, n_variables, n_nodes, capacity))

    resize!(_neighbor_ids, 3 * capacity)
    mortars.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                                       (3, capacity))

    resize!(large_sides, capacity)

    resize!(orientations, capacity)

    return nothing
end

function L2MortarContainer2D{uEltype}(capacity::Integer, n_variables,
                                      n_nodes) where {uEltype <: Real}
    nan = convert(uEltype, NaN)

    # Initialize fields with defaults
    _u_upper = fill(nan, 2 * n_variables * n_nodes * capacity)
    u_upper = unsafe_wrap(Array, pointer(_u_upper),
                          (2, n_variables, n_nodes, capacity))

    _u_lower = fill(nan, 2 * n_variables * n_nodes * capacity)
    u_lower = unsafe_wrap(Array, pointer(_u_lower),
                          (2, n_variables, n_nodes, capacity))

    _neighbor_ids = fill(typemin(Int), 3 * capacity)
    neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                               (3, capacity))

    large_sides = fill(typemin(Int), capacity)

    orientations = fill(typemin(Int), capacity)

    return L2MortarContainer2D{uEltype}(u_upper, u_lower, neighbor_ids, large_sides,
                                        orientations,
                                        _u_upper, _u_lower, _neighbor_ids)
end

# Return number of L2 mortars
@inline nmortars(l2mortars::L2MortarContainer2D) = length(l2mortars.orientations)

# Allow printing container contents
function Base.show(io::IO, ::MIME"text/plain", c::L2MortarContainer2D)
    @nospecialize c # reduce precompilation time

    println(io, '*'^20)
    for idx in CartesianIndices(c.u_upper)
        println(io, "c.u_upper[$idx] = $(c.u_upper[idx])")
    end
    for idx in CartesianIndices(c.u_lower)
        println(io, "c.u_lower[$idx] = $(c.u_lower[idx])")
    end
    println(io, "transpose(c.neighbor_ids) = $(transpose(c.neighbor_ids))")
    println(io, "c.large_sides = $(c.large_sides)")
    println(io, "c.orientations = $(c.orientations)")
    print(io, '*'^20)
end

# Create mortar container and initialize mortar data in `elements`.
function init_mortars(cell_ids, mesh::TreeMesh2D,
                      elements::ElementContainer2D,
                      ::LobattoLegendreMortarL2)
    # Initialize containers
    n_mortars = count_required_mortars(mesh, cell_ids)
    mortars = L2MortarContainer2D{eltype(elements)}(n_mortars, nvariables(elements),
                                                    nnodes(elements))

    # Connect elements with mortars
    init_mortars!(mortars, elements, mesh)
    return mortars
end

# Count the number of mortars that need to be created
function count_required_mortars(mesh::TreeMesh2D, cell_ids)
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

            # Skip if one of the small cells is on different rank -> create mpi mortar instead
            # (the coarse cell is always on the local rank)
            if mpi_isparallel()
                if direction == 1 # small cells left, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[2, neighbor_id]
                    upper_cell_id = mesh.tree.child_ids[4, neighbor_id]
                elseif direction == 2 # small cells right, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[1, neighbor_id]
                    upper_cell_id = mesh.tree.child_ids[3, neighbor_id]
                elseif direction == 3 # small cells left, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[3, neighbor_id]
                    upper_cell_id = mesh.tree.child_ids[4, neighbor_id]
                else # direction == 4, small cells right, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[1, neighbor_id]
                    upper_cell_id = mesh.tree.child_ids[2, neighbor_id]
                end
                small_cell_ids = (lower_cell_id, upper_cell_id)
                if any(cell -> !is_own_cell(mesh.tree, cell), small_cell_ids)
                    continue
                end
            end

            count += 1
        end
    end

    return count
end

# Initialize connectivity between elements and mortars
function init_mortars!(mortars, elements, mesh::TreeMesh2D)
    # Exit early if there are no mortars to initialize
    if nmortars(mortars) == 0
        return nothing
    end

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

            # Skip if one of the small cells is on different rank -> create mpi mortar instead
            # (the coarse cell is always on the local rank)
            if mpi_isparallel()
                if direction == 1 # small cells left, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[2, neighbor_cell_id]
                    upper_cell_id = mesh.tree.child_ids[4, neighbor_cell_id]
                elseif direction == 2 # small cells right, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[1, neighbor_cell_id]
                    upper_cell_id = mesh.tree.child_ids[3, neighbor_cell_id]
                elseif direction == 3 # small cells left, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[3, neighbor_cell_id]
                    upper_cell_id = mesh.tree.child_ids[4, neighbor_cell_id]
                else # direction == 4, small cells right, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[1, neighbor_cell_id]
                    upper_cell_id = mesh.tree.child_ids[2, neighbor_cell_id]
                end
                small_cell_ids = (lower_cell_id, upper_cell_id)
                if any(cell -> !is_own_cell(mesh.tree, cell), small_cell_ids)
                    continue
                end
            end

            # Create mortar between elements:
            # 1 -> small element in negative coordinate direction
            # 2 -> small element in positive coordinate direction
            # 3 -> large element
            count += 1
            mortars.neighbor_ids[3, count] = element
            if direction == 1
                mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[2,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[4,
                                                                         neighbor_cell_id]]
            elseif direction == 2
                mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[3,
                                                                         neighbor_cell_id]]
            elseif direction == 3
                mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[3,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[4,
                                                                         neighbor_cell_id]]
            elseif direction == 4
                mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1,
                                                                         neighbor_cell_id]]
                mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[2,
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

            # Set orientation (x -> 1, y -> 2)
            if direction in (1, 2)
                mortars.orientations[count] = 1
            else
                mortars.orientations[count] = 2
            end
        end
    end

    @assert count==nmortars(mortars) ("Actual mortar count ($count) does not match "*
                                      "expectations $(nmortars(mortars))")
end

# Container data structure (structure-of-arrays style) for DG MPI interfaces
mutable struct MPIInterfaceContainer2D{uEltype <: Real} <: AbstractContainer
    u::Array{uEltype, 4}            # [leftright, variables, i, interfaces]
    # Note: `local_neighbor_ids` stores the MPI-local neighbors, but with globally valid index!
    local_neighbor_ids::Vector{Int} # [interfaces]
    orientations::Vector{Int}       # [interfaces]
    remote_sides::Vector{Int}       # [interfaces]
    # internal `resize!`able storage
    _u::Vector{uEltype}
end

nvariables(mpi_interfaces::MPIInterfaceContainer2D) = size(mpi_interfaces.u, 2)
nnodes(mpi_interfaces::MPIInterfaceContainer2D) = size(mpi_interfaces.u, 3)
Base.eltype(mpi_interfaces::MPIInterfaceContainer2D) = eltype(mpi_interfaces.u)

# See explanation of Base.resize! for the element container
function Base.resize!(mpi_interfaces::MPIInterfaceContainer2D, capacity)
    n_nodes = nnodes(mpi_interfaces)
    n_variables = nvariables(mpi_interfaces)
    @unpack _u, local_neighbor_ids, orientations, remote_sides = mpi_interfaces

    resize!(_u, 2 * n_variables * n_nodes * capacity)
    mpi_interfaces.u = unsafe_wrap(Array, pointer(_u),
                                   (2, n_variables, n_nodes, capacity))

    resize!(local_neighbor_ids, capacity)

    resize!(orientations, capacity)

    resize!(remote_sides, capacity)

    return nothing
end

function MPIInterfaceContainer2D{uEltype}(capacity::Integer, n_variables,
                                          n_nodes) where {uEltype <: Real}
    nan = convert(uEltype, NaN)

    # Initialize fields with defaults
    _u = fill(nan, 2 * n_variables * n_nodes * capacity)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, n_variables, n_nodes, capacity))

    local_neighbor_ids = fill(typemin(Int), capacity)

    orientations = fill(typemin(Int), capacity)

    remote_sides = fill(typemin(Int), capacity)

    return MPIInterfaceContainer2D{uEltype}(u, local_neighbor_ids, orientations,
                                            remote_sides,
                                            _u)
end

# TODO: Taal, rename to ninterfaces?
# Return number of interfaces
function nmpiinterfaces(mpi_interfaces::MPIInterfaceContainer2D)
    length(mpi_interfaces.orientations)
end

# Create MPI interface container and initialize MPI interface data in `elements`.
function init_mpi_interfaces(cell_ids, mesh::TreeMesh2D,
                             elements::ElementContainer2D)
    # Initialize container
    n_mpi_interfaces = count_required_mpi_interfaces(mesh, cell_ids)
    mpi_interfaces = MPIInterfaceContainer2D{eltype(elements)}(n_mpi_interfaces,
                                                               nvariables(elements),
                                                               nnodes(elements))

    # Connect elements with interfaces
    init_mpi_interfaces!(mpi_interfaces, elements, mesh)
    return mpi_interfaces
end

# Count the number of MPI interfaces that need to be created
function count_required_mpi_interfaces(mesh::TreeMesh2D, cell_ids)
    # No MPI interfaces needed if MPI is not used
    if !mpi_isparallel()
        return 0
    end

    count = 0

    # Iterate over all cells
    for cell_id in cell_ids
        for direction in eachdirection(mesh.tree)
            # If no neighbor exists, current cell is small or at boundary and thus we need a mortar
            if !has_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            # Skip if neighbor has children
            neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
            if has_children(mesh.tree, neighbor_cell_id)
                continue
            end

            # Skip if neighbor is on this rank -> create regular interface instead
            if is_own_cell(mesh.tree, neighbor_cell_id)
                continue
            end

            count += 1
        end
    end

    return count
end

# Initialize connectivity between elements and interfaces
function init_mpi_interfaces!(mpi_interfaces, elements, mesh::TreeMesh2D)
    # Exit early if there are no MPI interfaces to initialize
    if nmpiinterfaces(mpi_interfaces) == 0
        return nothing
    end

    # Reset interface count
    count = 0

    # Iterate over all elements to find neighbors and to connect via mpi_interfaces
    for element in eachelement(elements)
        # Get cell id
        cell_id = elements.cell_ids[element]

        # Loop over directions
        for direction in eachdirection(mesh.tree)
            # If no neighbor exists, current cell is small and thus we need a mortar
            if !has_neighbor(mesh.tree, cell_id, direction)
                continue
            end

            # Skip if neighbor has children
            neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
            if has_children(mesh.tree, neighbor_cell_id)
                continue
            end

            # Skip if neighbor is on this MPI rank -> create regular interface instead
            if is_own_cell(mesh.tree, neighbor_cell_id)
                continue
            end

            # Create interface between elements
            count += 1
            # Note: `local_neighbor_ids` stores the MPI-local neighbors, 
            # but with globally valid index!
            mpi_interfaces.local_neighbor_ids[count] = element

            if iseven(direction) # element is "left" of interface, remote cell is "right" of interface
                mpi_interfaces.remote_sides[count] = 2
            else
                mpi_interfaces.remote_sides[count] = 1
            end

            # Set orientation (x -> 1, y -> 2)
            if direction in (1, 2) # x-direction
                mpi_interfaces.orientations[count] = 1
            else # y-direction
                mpi_interfaces.orientations[count] = 2
            end
        end
    end

    @assert count==nmpiinterfaces(mpi_interfaces) ("Actual interface count ($count) does not match "
                                                   *"expectations $(nmpiinterfaces(mpi_interfaces))")
end

# Container data structure (structure-of-arrays style) for DG L2 mortars
# Positions/directions for orientations = 1, large_sides = 2:
# mortar is orthogonal to x-axis, large side is in positive coordinate direction wrt mortar
#           |    |
# upper = 2 |    |
#           |    |
#                | 3 = large side
#           |    |
# lower = 1 |    |
#           |    |
mutable struct MPIL2MortarContainer2D{uEltype <: Real} <: AbstractContainer
    u_upper::Array{uEltype, 4} # [leftright, variables, i, mortars]
    u_lower::Array{uEltype, 4} # [leftright, variables, i, mortars]
    # Note: `local_neighbor_ids` stores the MPI-local neighbors, but with globally valid index!
    local_neighbor_ids::Vector{Vector{Int}}       # [mortars][ids]
    local_neighbor_positions::Vector{Vector{Int}} # [mortars][positions]
    # Large sides: left -> 1, right -> 2
    large_sides::Vector{Int}  # [mortars]
    orientations::Vector{Int} # [mortars]
    # internal `resize!`able storage
    _u_upper::Vector{uEltype}
    _u_lower::Vector{uEltype}
end

nvariables(mpi_mortars::MPIL2MortarContainer2D) = size(mpi_mortars.u_upper, 2)
nnodes(mpi_mortars::MPIL2MortarContainer2D) = size(mpi_mortars.u_upper, 3)
Base.eltype(mpi_mortars::MPIL2MortarContainer2D) = eltype(mpi_mortars.u_upper)

# See explanation of Base.resize! for the element container
function Base.resize!(mpi_mortars::MPIL2MortarContainer2D, capacity)
    n_nodes = nnodes(mpi_mortars)
    n_variables = nvariables(mpi_mortars)
    @unpack _u_upper, _u_lower, local_neighbor_ids, local_neighbor_positions,
    large_sides, orientations = mpi_mortars

    resize!(_u_upper, 2 * n_variables * n_nodes * capacity)
    mpi_mortars.u_upper = unsafe_wrap(Array, pointer(_u_upper),
                                      (2, n_variables, n_nodes, capacity))

    resize!(_u_lower, 2 * n_variables * n_nodes * capacity)
    mpi_mortars.u_lower = unsafe_wrap(Array, pointer(_u_lower),
                                      (2, n_variables, n_nodes, capacity))

    resize!(local_neighbor_ids, capacity)
    resize!(local_neighbor_positions, capacity)

    resize!(large_sides, capacity)

    resize!(orientations, capacity)

    return nothing
end

function MPIL2MortarContainer2D{uEltype}(capacity::Integer, n_variables,
                                         n_nodes) where {uEltype <: Real}
    nan = convert(uEltype, NaN)

    # Initialize fields with defaults
    _u_upper = fill(nan, 2 * n_variables * n_nodes * capacity)
    u_upper = unsafe_wrap(Array, pointer(_u_upper),
                          (2, n_variables, n_nodes, capacity))

    _u_lower = fill(nan, 2 * n_variables * n_nodes * capacity)
    u_lower = unsafe_wrap(Array, pointer(_u_lower),
                          (2, n_variables, n_nodes, capacity))

    local_neighbor_ids = fill(Vector{Int}(), capacity)
    local_neighbor_positions = fill(Vector{Int}(), capacity)

    large_sides = fill(typemin(Int), capacity)

    orientations = fill(typemin(Int), capacity)

    return MPIL2MortarContainer2D{uEltype}(u_upper, u_lower, local_neighbor_ids,
                                           local_neighbor_positions, large_sides,
                                           orientations,
                                           _u_upper, _u_lower)
end

# Return number of L2 mortars
@inline function nmpimortars(mpi_l2mortars::MPIL2MortarContainer2D)
    length(mpi_l2mortars.orientations)
end

# Create MPI mortar container and initialize MPI mortar data in `elements`.
function init_mpi_mortars(cell_ids, mesh::TreeMesh2D,
                          elements::ElementContainer2D,
                          ::LobattoLegendreMortarL2)
    # Initialize containers
    n_mpi_mortars = count_required_mpi_mortars(mesh, cell_ids)
    mpi_mortars = MPIL2MortarContainer2D{eltype(elements)}(n_mpi_mortars,
                                                           nvariables(elements),
                                                           nnodes(elements))

    # Connect elements with mortars
    init_mpi_mortars!(mpi_mortars, elements, mesh)
    return mpi_mortars
end

# Count the number of MPI mortars that need to be created
function count_required_mpi_mortars(mesh::TreeMesh2D, cell_ids)
    # No MPI mortars needed if MPI is not used
    if !mpi_isparallel()
        return 0
    end

    count = 0

    for cell_id in cell_ids
        for direction in eachdirection(mesh.tree)
            # If no neighbor exists, cell is small with large neighbor or at boundary
            if !has_neighbor(mesh.tree, cell_id, direction)
                # If no large neighbor exists, cell is at boundary -> do nothing
                if !has_coarse_neighbor(mesh.tree, cell_id, direction)
                    continue
                end

                # Skip if the large neighbor is on the same rank to prevent double counting
                parent_id = mesh.tree.parent_ids[cell_id]
                large_cell_id = mesh.tree.neighbor_ids[direction, parent_id]
                if is_own_cell(mesh.tree, large_cell_id)
                    continue
                end

                # Current cell is small with large neighbor on a different rank, find the other
                # small cell
                if direction == 1 # small cells right, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[1, parent_id]
                    upper_cell_id = mesh.tree.child_ids[3, parent_id]
                elseif direction == 2 # small cells left, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[2, parent_id]
                    upper_cell_id = mesh.tree.child_ids[4, parent_id]
                elseif direction == 3 # small cells right, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[1, parent_id]
                    upper_cell_id = mesh.tree.child_ids[2, parent_id]
                else # direction == 4, small cells left, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[3, parent_id]
                    upper_cell_id = mesh.tree.child_ids[4, parent_id]
                end

                if cell_id == lower_cell_id
                    sibling_id = upper_cell_id
                elseif cell_id == upper_cell_id
                    sibling_id = lower_cell_id
                else
                    error("should not happen")
                end

                # Skip if the other small cell is on the same rank and its id is smaller than the current
                # cell id to prevent double counting
                if is_own_cell(mesh.tree, sibling_id) && sibling_id < cell_id
                    continue
                end
            else # Cell has a neighbor
                # If neighbor has no children, this is a conforming interface -> do nothing
                neighbor_id = mesh.tree.neighbor_ids[direction, cell_id]
                if !has_children(mesh.tree, neighbor_id)
                    continue
                end

                # Skip if both small cells are on this rank -> create regular mortar instead
                if direction == 1 # small cells left, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[2, neighbor_id]
                    upper_cell_id = mesh.tree.child_ids[4, neighbor_id]
                elseif direction == 2 # small cells right, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[1, neighbor_id]
                    upper_cell_id = mesh.tree.child_ids[3, neighbor_id]
                elseif direction == 3 # small cells left, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[3, neighbor_id]
                    upper_cell_id = mesh.tree.child_ids[4, neighbor_id]
                else # direction == 4, small cells right, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[1, neighbor_id]
                    upper_cell_id = mesh.tree.child_ids[2, neighbor_id]
                end
                small_cell_ids = (lower_cell_id, upper_cell_id)
                if all(cell -> is_own_cell(mesh.tree, cell), small_cell_ids)
                    continue
                end
            end

            count += 1
        end
    end

    return count
end

# Initialize connectivity between elements and mortars
function init_mpi_mortars!(mpi_mortars, elements, mesh::TreeMesh2D)
    # Exit early if there are no MPI mortars to initialize
    if nmpimortars(mpi_mortars) == 0
        return nothing
    end

    # Construct cell -> element mapping for easier algorithm implementation
    tree = mesh.tree
    c2e = zeros(Int, length(tree))
    for element in eachelement(elements)
        c2e[elements.cell_ids[element]] = element
    end

    # Reset mortar count
    count = 0

    # Iterate over all elements to find neighbors and to connect via mortars
    for element in eachelement(elements)
        cell_id = elements.cell_ids[element]

        for direction in eachdirection(mesh.tree)
            # If no neighbor exists, cell is small with large neighbor or at boundary
            if !has_neighbor(mesh.tree, cell_id, direction)
                # If no large neighbor exists, cell is at boundary -> do nothing
                if !has_coarse_neighbor(mesh.tree, cell_id, direction)
                    continue
                end

                # Skip if the large neighbor is on the same rank -> will be handled in another iteration
                parent_cell_id = mesh.tree.parent_ids[cell_id]
                large_cell_id = mesh.tree.neighbor_ids[direction, parent_cell_id]
                if is_own_cell(mesh.tree, large_cell_id)
                    continue
                end

                # Current cell is small with large neighbor on a different rank, find the other
                # small cell
                if direction == 1 # small cells right, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[1, parent_cell_id]
                    upper_cell_id = mesh.tree.child_ids[3, parent_cell_id]
                elseif direction == 2 # small cells left, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[2, parent_cell_id]
                    upper_cell_id = mesh.tree.child_ids[4, parent_cell_id]
                elseif direction == 3 # small cells right, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[1, parent_cell_id]
                    upper_cell_id = mesh.tree.child_ids[2, parent_cell_id]
                else # direction == 4, small cells left, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[3, parent_cell_id]
                    upper_cell_id = mesh.tree.child_ids[4, parent_cell_id]
                end

                if cell_id == lower_cell_id
                    sibling_id = upper_cell_id
                elseif cell_id == upper_cell_id
                    sibling_id = lower_cell_id
                else
                    error("should not happen")
                end

                # Skip if the other small cell is on the same rank and its id is smaller than the current
                # cell id to prevent double counting
                if is_own_cell(mesh.tree, sibling_id) && sibling_id < cell_id
                    continue
                end
            else # Cell has a neighbor
                large_cell_id = cell_id # save explicitly for later processing

                # If neighbor has no children, this is a conforming interface -> do nothing
                neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
                if !has_children(mesh.tree, neighbor_cell_id)
                    continue
                end

                # Skip if both small cells are on this rank -> create regular mortar instead
                if direction == 1 # small cells left, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[2, neighbor_cell_id]
                    upper_cell_id = mesh.tree.child_ids[4, neighbor_cell_id]
                elseif direction == 2 # small cells right, mortar in x-direction
                    lower_cell_id = mesh.tree.child_ids[1, neighbor_cell_id]
                    upper_cell_id = mesh.tree.child_ids[3, neighbor_cell_id]
                elseif direction == 3 # small cells left, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[3, neighbor_cell_id]
                    upper_cell_id = mesh.tree.child_ids[4, neighbor_cell_id]
                else # direction == 4, small cells right, mortar in y-direction
                    lower_cell_id = mesh.tree.child_ids[1, neighbor_cell_id]
                    upper_cell_id = mesh.tree.child_ids[2, neighbor_cell_id]
                end
                small_cell_ids = (lower_cell_id, upper_cell_id)
                if all(cell -> is_own_cell(mesh.tree, cell), small_cell_ids)
                    continue
                end
            end

            # Create mortar between elements:
            # 1 -> small element in negative coordinate direction
            # 2 -> small element in positive coordinate direction
            # 3 -> large element
            count += 1

            # Note: `local_neighbor_ids` stores the MPI-local neighbors, 
            # but with globally valid index!
            local_neighbor_ids = Vector{Int}()
            local_neighbor_positions = Vector{Int}()
            if is_own_cell(mesh.tree, lower_cell_id)
                push!(local_neighbor_ids, c2e[lower_cell_id])
                push!(local_neighbor_positions, 1)
            end
            if is_own_cell(mesh.tree, upper_cell_id)
                push!(local_neighbor_ids, c2e[upper_cell_id])
                push!(local_neighbor_positions, 2)
            end
            if is_own_cell(mesh.tree, large_cell_id)
                push!(local_neighbor_ids, c2e[large_cell_id])
                push!(local_neighbor_positions, 3)
            end

            mpi_mortars.local_neighbor_ids[count] = local_neighbor_ids
            mpi_mortars.local_neighbor_positions[count] = local_neighbor_positions

            # Set large side, which denotes the direction (1 -> negative, 2 -> positive) of the large side
            # To prevent double counting, the mortars are always identified from the point of view of
            # a large cell, if it is on this rank. In that case, direction points towards the small cells.
            # If the large cell is not on this rank, the point of view of a small cell is taken instead,
            # hence direction points towards the large cell in this case.
            if iseven(direction)
                mpi_mortars.large_sides[count] = is_own_cell(mesh.tree, large_cell_id) ?
                                                 1 : 2
            else
                mpi_mortars.large_sides[count] = is_own_cell(mesh.tree, large_cell_id) ?
                                                 2 : 1
            end

            # Set orientation (1, 2 -> x; 3, 4 -> y)
            if direction in (1, 2)
                mpi_mortars.orientations[count] = 1
            else
                mpi_mortars.orientations[count] = 2
            end
        end
    end

    return nothing
end

# Container data structure (structure-of-arrays style) for FCT-type antidiffusive fluxes
#                            (i, j+1)
#                               |
#                          flux2(i, j+1)
#                               |
# (i-1, j) ---flux1(i, j)--- (i, j) ---flux1(i+1, j)--- (i+1, j)
#                               |
#                          flux2(i, j)
#                               |
#                            (i, j-1)
mutable struct ContainerAntidiffusiveFlux2D{uEltype <: Real}
    antidiffusive_flux1_L::Array{uEltype, 4} # [variables, i, j, elements]
    antidiffusive_flux1_R::Array{uEltype, 4} # [variables, i, j, elements]
    antidiffusive_flux2_L::Array{uEltype, 4} # [variables, i, j, elements]
    antidiffusive_flux2_R::Array{uEltype, 4} # [variables, i, j, elements]
    # internal `resize!`able storage
    _antidiffusive_flux1_L::Vector{uEltype}
    _antidiffusive_flux1_R::Vector{uEltype}
    _antidiffusive_flux2_L::Vector{uEltype}
    _antidiffusive_flux2_R::Vector{uEltype}
end

function ContainerAntidiffusiveFlux2D{uEltype}(capacity::Integer, n_variables,
                                               n_nodes) where {uEltype <: Real}
    nan_uEltype = convert(uEltype, NaN)

    # Initialize fields with defaults
    _antidiffusive_flux1_L = fill(nan_uEltype,
                                  n_variables * (n_nodes + 1) * n_nodes * capacity)
    antidiffusive_flux1_L = unsafe_wrap(Array, pointer(_antidiffusive_flux1_L),
                                        (n_variables, n_nodes + 1, n_nodes, capacity))
    _antidiffusive_flux1_R = fill(nan_uEltype,
                                  n_variables * (n_nodes + 1) * n_nodes * capacity)
    antidiffusive_flux1_R = unsafe_wrap(Array, pointer(_antidiffusive_flux1_R),
                                        (n_variables, n_nodes + 1, n_nodes, capacity))

    _antidiffusive_flux2_L = fill(nan_uEltype,
                                  n_variables * n_nodes * (n_nodes + 1) * capacity)
    antidiffusive_flux2_L = unsafe_wrap(Array, pointer(_antidiffusive_flux2_L),
                                        (n_variables, n_nodes, n_nodes + 1, capacity))
    _antidiffusive_flux2_R = fill(nan_uEltype,
                                  n_variables * n_nodes * (n_nodes + 1) * capacity)
    antidiffusive_flux2_R = unsafe_wrap(Array, pointer(_antidiffusive_flux2_R),
                                        (n_variables, n_nodes, n_nodes + 1, capacity))

    return ContainerAntidiffusiveFlux2D{uEltype}(antidiffusive_flux1_L,
                                                 antidiffusive_flux1_R,
                                                 antidiffusive_flux2_L,
                                                 antidiffusive_flux2_R,
                                                 _antidiffusive_flux1_L,
                                                 _antidiffusive_flux1_R,
                                                 _antidiffusive_flux2_L,
                                                 _antidiffusive_flux2_R)
end

nvariables(fluxes::ContainerAntidiffusiveFlux2D) = size(fluxes.antidiffusive_flux1_L, 1)
nnodes(fluxes::ContainerAntidiffusiveFlux2D) = size(fluxes.antidiffusive_flux1_L, 3)

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(fluxes::ContainerAntidiffusiveFlux2D, capacity)
    n_nodes = nnodes(fluxes)
    n_variables = nvariables(fluxes)

    @unpack _antidiffusive_flux1_L, _antidiffusive_flux2_L, _antidiffusive_flux1_R, _antidiffusive_flux2_R = fluxes

    resize!(_antidiffusive_flux1_L, n_variables * (n_nodes + 1) * n_nodes * capacity)
    fluxes.antidiffusive_flux1_L = unsafe_wrap(Array, pointer(_antidiffusive_flux1_L),
                                               (n_variables, n_nodes + 1, n_nodes,
                                                capacity))
    resize!(_antidiffusive_flux1_R, n_variables * (n_nodes + 1) * n_nodes * capacity)
    fluxes.antidiffusive_flux1_R = unsafe_wrap(Array, pointer(_antidiffusive_flux1_R),
                                               (n_variables, n_nodes + 1, n_nodes,
                                                capacity))
    resize!(_antidiffusive_flux2_L, n_variables * n_nodes * (n_nodes + 1) * capacity)
    fluxes.antidiffusive_flux2_L = unsafe_wrap(Array, pointer(_antidiffusive_flux2_L),
                                               (n_variables, n_nodes, n_nodes + 1,
                                                capacity))
    resize!(_antidiffusive_flux2_R, n_variables * n_nodes * (n_nodes + 1) * capacity)
    fluxes.antidiffusive_flux2_R = unsafe_wrap(Array, pointer(_antidiffusive_flux2_R),
                                               (n_variables, n_nodes, n_nodes + 1,
                                                capacity))

    return nothing
end

# Container data structure (structure-of-arrays style) for variables used for IDP limiting
mutable struct ContainerSubcellLimiterIDP2D{uEltype <: Real}
    alpha::Array{uEltype, 3}                  # [i, j, element]
    alpha1::Array{uEltype, 3}
    alpha2::Array{uEltype, 3}
    variable_bounds::Dict{Symbol, Array{uEltype, 3}}
    # internal `resize!`able storage
    _alpha::Vector{uEltype}
    _alpha1::Vector{uEltype}
    _alpha2::Vector{uEltype}
    _variable_bounds::Dict{Symbol, Vector{uEltype}}
end

function ContainerSubcellLimiterIDP2D{uEltype}(capacity::Integer, n_nodes,
                                               bound_keys) where {uEltype <: Real}
    nan_uEltype = convert(uEltype, NaN)

    # Initialize fields with defaults
    _alpha = fill(nan_uEltype, n_nodes * n_nodes * capacity)
    alpha = unsafe_wrap(Array, pointer(_alpha), (n_nodes, n_nodes, capacity))
    _alpha1 = fill(nan_uEltype, (n_nodes + 1) * n_nodes * capacity)
    alpha1 = unsafe_wrap(Array, pointer(_alpha1), (n_nodes + 1, n_nodes, capacity))
    _alpha2 = fill(nan_uEltype, n_nodes * (n_nodes + 1) * capacity)
    alpha2 = unsafe_wrap(Array, pointer(_alpha2), (n_nodes, n_nodes + 1, capacity))

    _variable_bounds = Dict{Symbol, Vector{uEltype}}()
    variable_bounds = Dict{Symbol, Array{uEltype, 3}}()
    for key in bound_keys
        _variable_bounds[key] = fill(nan_uEltype, n_nodes * n_nodes * capacity)
        variable_bounds[key] = unsafe_wrap(Array, pointer(_variable_bounds[key]),
                                           (n_nodes, n_nodes, capacity))
    end

    return ContainerSubcellLimiterIDP2D{uEltype}(alpha, alpha1, alpha2,
                                                 variable_bounds,
                                                 _alpha, _alpha1, _alpha2,
                                                 _variable_bounds)
end

nnodes(container::ContainerSubcellLimiterIDP2D) = size(container.alpha, 1)

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(container::ContainerSubcellLimiterIDP2D, capacity)
    n_nodes = nnodes(container)

    (; _alpha, _alpha1, _alpha2) = container
    resize!(_alpha, n_nodes * n_nodes * capacity)
    container.alpha = unsafe_wrap(Array, pointer(_alpha), (n_nodes, n_nodes, capacity))
    container.alpha .= convert(eltype(container.alpha), NaN)
    resize!(_alpha1, (n_nodes + 1) * n_nodes * capacity)
    container.alpha1 = unsafe_wrap(Array, pointer(_alpha1),
                                   (n_nodes + 1, n_nodes, capacity))
    resize!(_alpha2, n_nodes * (n_nodes + 1) * capacity)
    container.alpha2 = unsafe_wrap(Array, pointer(_alpha2),
                                   (n_nodes, n_nodes + 1, capacity))

    (; _variable_bounds) = container
    for (key, _) in _variable_bounds
        resize!(_variable_bounds[key], n_nodes * n_nodes * capacity)
        container.variable_bounds[key] = unsafe_wrap(Array,
                                                     pointer(_variable_bounds[key]),
                                                     (n_nodes, n_nodes, capacity))
    end

    return nothing
end
end # @muladd
