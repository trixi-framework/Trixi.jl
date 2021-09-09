# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    UnstructuredSortedBoundaryTypes

General container to sort the boundary conditions by type for some unstructured meshes/solvers.
It stores a set of global indices for each boundary condition type to expedite computation
during the call to `calc_boundary_flux!`. The original dictionary form of the boundary conditions
set by the user in the elixir file is also stored for printing.
"""
mutable struct UnstructuredSortedBoundaryTypes{N, BCs<:NTuple{N, Any}}
  boundary_condition_types::BCs # specific boundary condition type(s), e.g. BoundaryConditionDirichlet
  boundary_indices::NTuple{N, Vector{Int}} # integer vectors containing global boundary indices
  boundary_dictionary::Dict{Symbol, Any} # boundary conditions as set by the user in the elixir file
end


# constructor that "eats" the original boundary condition dictionary and sorts the information
# from the `UnstructuredBoundaryContainer2D` in cache.boundaries according to the boundary types
# and stores the associated global boundary indexing in NTuple
function UnstructuredSortedBoundaryTypes(boundary_conditions::Dict, cache)
  # extract the unique boundary function routines from the dictionary
  boundary_condition_types = Tuple(unique(collect(values(boundary_conditions))))
  n_boundary_types = length(boundary_condition_types)
  boundary_indices = ntuple(_ -> [], n_boundary_types)

  container = UnstructuredSortedBoundaryTypes{n_boundary_types, typeof(boundary_condition_types)}(
    boundary_condition_types, boundary_indices, boundary_conditions)

  initialize!(container, cache)
end


function initialize!(boundary_types_container::UnstructuredSortedBoundaryTypes{N}, cache) where N
  @unpack boundary_dictionary, boundary_condition_types = boundary_types_container

  unique_names = unique(cache.boundaries.name)

  # Verify that each Dict key is a valid boundary name
  for key in keys(boundary_dictionary)
    if !(key in unique_names)
      error("Key $(repr(key)) is not a valid boundary name")
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
      temp_indices = findall(x->x===test_name, cache.boundaries.name)
      if test_condition === boundary_condition_types[j]
        indices_for_current_type = vcat(indices_for_current_type, temp_indices)
      end
    end
    _boundary_indices[j] = sort!(indices_for_current_type)
  end

  # convert the work array with the boundary indices into a tuple
  boundary_types_container.boundary_indices = Tuple(_boundary_indices)

  return boundary_types_container
end


end # @muladd
