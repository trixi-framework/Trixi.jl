# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    UnstructuredSortedBoundaryTypes

General container to sort the boundary conditions by type and name for some unstructured meshes/solvers.
It stores a set of global indices for each boundary condition type and name to expedite computation
during the call to `calc_boundary_flux!`. The original dictionary form of the boundary conditions
set by the user in the elixir file is also stored for printing.
"""
mutable struct UnstructuredSortedBoundaryTypes{N, BCs <: NTuple{N, Any},
                                               Vec <: AbstractVector{<:Integer}}
    boundary_condition_types::BCs # specific boundary condition type(s), e.g. BoundaryConditionDirichlet
    boundary_indices::NTuple{N, Vec} # integer vectors containing global boundary indices
    boundary_dictionary::Dict{Symbol, Any} # boundary conditions as set by the user in the elixir file
    boundary_symbol_indices::Dict{Symbol, Vector{Int}} # integer vectors containing global boundary indices per boundary identifier
end

# constructor that "eats" the original boundary condition dictionary and sorts the information
# from the `UnstructuredBoundaryContainer2D` in cache.boundaries according to the boundary types
# and stores the associated global boundary indexing in NTuple
function UnstructuredSortedBoundaryTypes(boundary_conditions::Dict, cache)
    # extract the unique boundary function routines from the dictionary
    boundary_condition_types = Tuple(unique(collect(values(boundary_conditions))))
    n_boundary_types = length(boundary_condition_types)
    boundary_indices = ntuple(_ -> [], n_boundary_types)

    # Initialize `boundary_symbol_indices` as an empty dictionary, filled later in `initialize!`
    boundary_symbol_indices = Dict{Symbol, Vector{Int}}()

    container = UnstructuredSortedBoundaryTypes{n_boundary_types,
                                                typeof(boundary_condition_types),
                                                Vector{Int}}(boundary_condition_types,
                                                             boundary_indices,
                                                             boundary_conditions,
                                                             boundary_symbol_indices)

    initialize!(container, cache)
end

function initialize!(boundary_types_container::UnstructuredSortedBoundaryTypes{N},
                     cache) where {N}
    @unpack boundary_dictionary, boundary_condition_types = boundary_types_container

    unique_names = unique(cache.boundaries.name)

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
            for key in keys(boundary_dictionary)
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
        for key in keys(boundary_dictionary)
            if !(key in unique_names)
                error("Key $(repr(key)) is not a valid boundary name. " *
                      "Valid names are $unique_names.")
            end
        end
    end

    # Verify that each boundary has a boundary condition
    for name in unique_names
        if name !== Symbol("---") && !haskey(boundary_dictionary, name)
            error("No boundary condition specified for boundary $(repr(name))")
        end
    end

    # pull and sort the indexing for each boundary type
    _boundary_indices = Vector{Any}(nothing, N)
    for j in 1:N
        indices_for_current_type = Int[]
        for (test_name, test_condition) in boundary_dictionary
            temp_indices = findall(x -> x === test_name, cache.boundaries.name)
            if test_condition === boundary_condition_types[j]
                indices_for_current_type = vcat(indices_for_current_type, temp_indices)
            end
        end
        _boundary_indices[j] = sort!(indices_for_current_type)
    end

    # Check if all boundaries (determined from connectivity) are equipped with a boundary condition
    for (index, boundary_name) in enumerate(cache.boundaries.name)
        if !(boundary_name in keys(boundary_dictionary))
            neighbor_element = cache.boundaries.neighbor_ids[index]
            @warn "Boundary condition for boundary type $(repr(boundary_name)) of boundary $(index) (neighbor element $neighbor_element) not found in boundary dictionary!"
        end
    end

    # convert the work array with the boundary indices into a tuple
    boundary_types_container.boundary_indices = Tuple(_boundary_indices)

    # Store boundary indices per symbol (required for force computations, for instance)
    for (symbol, _) in boundary_dictionary
        indices = findall(x -> x === symbol, cache.boundaries.name)
        # Store the indices in `boundary_symbol_indices` dictionary
        boundary_types_container.boundary_symbol_indices[symbol] = sort!(indices)
    end

    return boundary_types_container
end

# @eval due to @muladd
@eval Adapt.@adapt_structure(UnstructuredSortedBoundaryTypes)


"""
    boundary_condition_default_p4est_2D(boundary_condition)

Create a default boundary condition dictionary for p4est meshes in 2D
that use the standard boundary naming convention.

This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary

# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries

# Returns
- `Dict{Symbol, Any}`: Dictionary mapping boundary names to the boundary condition
"""
function boundary_condition_default_p4est_2D(boundary_condition)

     return Dict(:x_neg => boundary_condition,
                 :y_neg => boundary_condition,
                 :y_pos => boundary_condition,
                 :x_pos => boundary_condition)
end 

"""
    boundary_condition_default_3D

Create a default boundary condition dictionary for p4est meshes in 3D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary
- `:z_neg`: negative z-direction boundary
- `:z_pos`: positive z-direction boundary
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- `Dict{Symbol, Any}`: Dictionary mapping boundary names to the boundary condition
"""

function boundary_condition_default_p4est_3D(boundary_condition)

     return Dict(:x_neg => boundary_condition,
                  :x_pos => boundary_condition,
                  :y_neg => boundary_condition,
                  :y_pos => boundary_condition,
                  :z_neg => boundary_condition,
                  :z_pos => boundary_condition)
end


"""
    boundary_condition_default_structured_1D
Create a default boundary condition dictionary for structured meshes in 1D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary   
# Arguments
- `boundarycondition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function boundary_condition_default_structured_1D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition)
end
"""
    boundary_condition_default_structured_2D
Create a default boundary condition dictionary for structured meshes in 2D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary   
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary
# Arguments
- `boundarycondition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function boundary_condition_default_structured_2D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition)
end

"""
    boundary_condition_default_structured_3D(boundary_condition)

Create a default boundary condition dictionary for structured meshes in 3D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary  
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary
- `:z_neg`: negative z-direction boundary
- `:z_pos`: positive z-direction boundary 
# Arguments
- `boundarycondition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function boundary_condition_default_structured_3D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition,
            z_neg = boundary_condition,
            z_pos = boundary_condition)

end 


"""
    boundary_condition_default_tree_1D(boundary_condition)

Create a default boundary condition dictionary for tree meshes in 1D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition

"""
function boundary_condition_default_tree_1D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition)
end

"""
    boundary_condition_default_tree_2D(boundary_condition)

Create a default boundary condition dictionary for tree meshes in 2D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary    
- `:x_pos`: positive x-direction boundary
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function boundary_condition_default_tree_2D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition)
end

"""
    boundary_condition_default_tree_3D
Create a default boundary condition dictionary for tree meshes in 3D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary
- `:z_neg`: negative z-direction boundary
- `:z_pos`: positive z-direction boundary
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""

function boundary_condition_default_tree_3D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition,
            z_neg = boundary_condition,
            z_pos = boundary_condition)
end
end#@muladd
