"""
    SortedUnstructuredQuadBoundaryTypes{N, BCs<:NTuple{N, Any}}

General container to sort the boundary conditions by type for the unstructured quadrilateral solver.
It stores a set of global indices for each boundary condition type to expedite computation
during the call to `calc_boundary_flux!`. The original dictionary form of the boundary conditions
set by the user in the elixir file is also stored for printing.

!!! warning "Experimental code"
    This boundary condition container is experimental and can change any time.
"""
struct SortedUnstructuredQuadBoundaryTypes{N, BCs<:NTuple{N, Any}}
  boundary_condition_types::BCs # specific boundary condition type(s), e.g. BoundaryConditionWall
  boundary_indices::NTuple{N, Vector{Int}} # integer vectors containing global boundary indices
  boundary_dictionary::Dict{Symbol, Any} # boundary conditions as set by the user in the elixir file
end


# constructor that "eats" the original boundary condition dictionary and sorts the information
# from the `UnstructuredBoundaryContainer2D` in cache.boundaries according to the boundary types
# and stores the associated global boundary indexing in NTuple
function SortedUnstructuredQuadBoundaryTypes(boundary_conditions::Dict, cache)

  # extract the unique boundary function routines from the dictionary
  boundary_condition_types = Tuple(unique(collect(values(boundary_conditions))))
  n_boundary_types = length(boundary_condition_types)

  # pull and sort the indexing for each boundary type
  _boundary_indices = Vector{Any}(nothing, n_boundary_types)
  for j in 1:n_boundary_types
    indices_for_current_type = Int[]
    for (test_name, test_condition) in boundary_conditions
      temp_indices = findall(x->x===test_name, cache.boundaries.name)
      if test_condition === boundary_condition_types[j]
        indices_for_current_type = vcat(indices_for_current_type, temp_indices)
      end
    end
    _boundary_indices[j] = sort!(indices_for_current_type)
  end

  # convert the work array with the boundary indices into a tuple
  boundary_indices = Tuple(_boundary_indices)

  boundary_dictionary = boundary_conditions

  return SortedUnstructuredQuadBoundaryTypes{n_boundary_types, typeof(boundary_condition_types)}(
                                                                         boundary_condition_types,
                                                                         boundary_indices,
                                                                         boundary_dictionary)
end
