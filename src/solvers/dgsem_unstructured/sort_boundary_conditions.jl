# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    UnstructuredSortedBoundaryTypes

General struct to sort the boundary conditions by type and name for some unstructured meshes/solvers.
It stores a set of global indices for each boundary condition type and name to expedite computation
during the call to `calc_boundary_flux!`. The original `NamedTuple` of the boundary conditions
set by the user in the elixir file is also stored for printing.
"""
mutable struct UnstructuredSortedBoundaryTypes{N, BCs <: NTuple{N, Any},
                                               Vec <: AbstractVector{<:Integer},
                                               BoundaryConditions <: NamedTuple}
    const boundary_condition_types::BCs # specific boundary condition type(s), e.g. BoundaryConditionDirichlet
    boundary_indices::NTuple{N, Vec} # integer vectors containing global boundary indices
    const boundary_conditions::BoundaryConditions # boundary conditions as set by the user in the elixir file
    boundary_symbol_indices::Dict{Symbol, Vector{Int}} # integer vectors containing global boundary indices per boundary identifier
end

# constructor that "eats" the original boundary condition NamedTuple and sorts the information
# from the `UnstructuredBoundaryContainer2D` in cache.boundaries according to the boundary types
# and stores the associated global boundary indexing in NTuple
function UnstructuredSortedBoundaryTypes(boundary_conditions::NamedTuple, cache)
    # extract the unique boundary function routines from the NamedTuple
    BoundaryConditions = typeof(boundary_conditions)
    boundary_condition_types = Tuple(unique(values(boundary_conditions)))
    n_boundary_types = length(boundary_condition_types)

    validate_boundary_conditions(boundary_conditions, cache)

    boundary_indices, boundary_symbol_indices = initialize_boundary_data(boundary_conditions,
                                                                         boundary_condition_types,
                                                                         cache)

    return UnstructuredSortedBoundaryTypes{n_boundary_types,
                                           typeof(boundary_condition_types),
                                           Vector{Int},
                                           BoundaryConditions}(boundary_condition_types,
                                                               boundary_indices,
                                                               boundary_conditions,
                                                               boundary_symbol_indices)
end

# Check that supplied boundary conditions are valid, i.e., 
# - that the keys of the `boundary_conditions` match the names of the boundaries identified by the mesh (`cache.boundaries.name`), and
# - that each boundary has a boundary condition specified
function validate_boundary_conditions(boundary_conditions::NamedTuple, cache)
    unique_names = unique(cache.boundaries.name) # boundaries identified by the mesh

    # Verify that the names of the user-given boundaries match the ones identified by the mesh
    if mpi_isparallel()
        # Exchange of boundaries names
        send_buffer = Vector{UInt8}(join(unique_names, "\0"))
        push!(send_buffer, 0)
        if mpi_isroot()
            recv_buffer_length = MPI.Gather(length(send_buffer), mpi_root(), mpi_comm())
            recv_buffer = Vector{UInt8}(undef, sum(recv_buffer_length))
            MPI.Gatherv!(send_buffer, MPI.VBuffer(recv_buffer, recv_buffer_length),
                         mpi_root(), mpi_comm())
            all_names = unique(Symbol.(split(String(recv_buffer), "\0";
                                             keepempty = false)))
            for key in keys(boundary_conditions)
                if !(key in all_names)
                    println(stderr,
                            "ERROR: Key $(repr(key)) is not a valid boundary name. " *
                            "Valid names are $all_names.")
                    MPI.Abort(mpi_comm(), 1)
                end
            end
        else
            MPI.Gather(length(send_buffer), mpi_root(), mpi_comm())
            MPI.Gatherv!(send_buffer, nothing, mpi_root(), mpi_comm())
        end
    else
        for key in keys(boundary_conditions)
            if !(key in unique_names)
                error("Key $(repr(key)) is not a valid boundary name. " *
                      "Valid names are $unique_names.")
            end
        end
    end

    # Verify that each boundary (determined from connectivity) is equipped with a boundary condition
    for (index, boundary_name) in enumerate(unique_names)
        neighbor_element = get_boundary_element(cache.boundaries, index)
        if boundary_name == Symbol("---")
            @warn "Mesh connectivity identified boundary $index (neighbor element $neighbor_element) as boundary element/non-connected - check your mesh!"
        elseif !(boundary_name in keys(boundary_conditions))
            @warn "Boundary condition for boundary type $(repr(boundary_name)) of boundary $index (neighbor element $neighbor_element) not found in boundary conditions!"
        end
    end

    return nothing
end

function initialize_boundary_data(boundary_conditions::NamedTuple,
                                  boundary_condition_types,
                                  cache)
    boundary_names = cache.boundaries.name

    # pull and sort the indexing for each boundary type
    _boundary_indices = Vector{Vector{Int}}(undef, length(boundary_condition_types))
    for j in eachindex(boundary_condition_types)
        indices_for_current_type = Int[]
        for (test_name, test_condition) in pairs(boundary_conditions)
            temp_indices = findall(x -> x === test_name, boundary_names)
            if test_condition === boundary_condition_types[j]
                indices_for_current_type = vcat(indices_for_current_type, temp_indices)
            end
        end
        _boundary_indices[j] = sort!(indices_for_current_type)
    end
    boundary_indices = Tuple(_boundary_indices)

    boundary_symbol_indices = Dict{Symbol, Vector{Int}}()
    for (symbol, _) in pairs(boundary_conditions)
        indices = findall(x -> x === symbol, boundary_names)
        # Store the indices in `boundary_symbol_indices` dictionary
        boundary_symbol_indices[symbol] = sort!(indices)
    end

    return boundary_indices, boundary_symbol_indices
end

# This is called after AMR, i.e., when the mesh has changed and the boundary indices need to be re-initialized.
# Note that at this point no validation of the boundary condition names is necessary.
function reinitialize!(boundary_types_container::UnstructuredSortedBoundaryTypes,
                       cache)
    @unpack boundary_conditions, boundary_condition_types = boundary_types_container
    boundary_indices, boundary_symbol_indices = initialize_boundary_data(boundary_conditions,
                                                                         boundary_condition_types,
                                                                         cache)

    boundary_types_container.boundary_indices = boundary_indices
    boundary_types_container.boundary_symbol_indices = boundary_symbol_indices

    return boundary_types_container
end

# @eval due to @muladd
@eval Adapt.@adapt_structure(UnstructuredSortedBoundaryTypes)
end # @muladd
