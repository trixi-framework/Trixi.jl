
"""
    AMRCallback(semi, indicator [,adaptor=AdaptorAMR(semi)];
                interval=5,
                adapt_initial_conditions=true,
                adapt_initial_conditions_only_refine=true)

Performs adaptive mesh refinement (AMR) every `interval` time steps
for a given semidiscretization `semi` using the chosen `indicator`.
"""
struct AMRCallback{Indicator, Adaptor}
  indicator::Indicator
  interval::Int
  adapt_initial_conditions::Bool
  adapt_initial_conditions_only_refine::Bool
  adaptor::Adaptor
end


function AMRCallback(semi, indicator, adaptor; interval=nothing,
                                               adapt_initial_conditions=true,
                                               adapt_initial_conditions_only_refine=true)
  # check arguments
  if !(interval isa Integer && interval >= 0)
    throw(ArgumentError("`interval` must be a non-negative integer (provided `interval = $interval`)"))
  end

  # AMR every `interval` time steps
  if interval > 0
    condition = (u, t, integrator) -> integrator.iter % interval == 0
  else # disable the AMR callback except possibly for initial refinement during initialization
    condition = (u, t, integrator) -> false
  end

  amr_callback = AMRCallback{typeof(indicator), typeof(adaptor)}(
                  indicator, interval, adapt_initial_conditions,
                  adapt_initial_conditions_only_refine, adaptor)

  DiscreteCallback(condition, amr_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end

function AMRCallback(semi, indicator; kwargs...)
  adaptor = AdaptorAMR(semi)
  AMRCallback(semi, indicator, adaptor; kwargs...)
end

function AdaptorAMR(semi; kwargs...)
  mesh, _, solver, _ = mesh_equations_solver_cache(semi)
  AdaptorAMR(mesh, solver; kwargs...)
end


# TODO: Taal bikeshedding, implement a method with less information and the signature
# function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AMRCallback}
#   amr_callback = cb.affect!
#   print(io, "AMRCallback")
# end
function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AMRCallback}
  amr_callback = cb.affect!
  println(io, "AMRCallback with")
  println(io, "- indicator: ", amr_callback.indicator)
  println(io, "- interval: ", amr_callback.interval)
  println(io, "- adapt_initial_conditions: ", amr_callback.adapt_initial_conditions)
  print(io,   "- adapt_initial_conditions_only_refine: ", amr_callback.adapt_initial_conditions_only_refine)
end


function get_element_variables!(element_variables, u, mesh, equations, solver, cache, amr_callback::AMRCallback)
  get_element_variables!(element_variables, u, mesh, equations, solver, cache, amr_callback.indicator, amr_callback)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:AMRCallback}
  amr_callback = cb.affect!
  semi = integrator.p

  @timeit_debug timer() "initial condition AMR" if amr_callback.adapt_initial_conditions
    # iterate until mesh does not change anymore
    has_changed = true
    while has_changed
      has_changed = amr_callback(integrator,
                                 only_refine=amr_callback.adapt_initial_conditions_only_refine)
      compute_coefficients!(integrator.u, t, semi)
      u_modified!(integrator, true)
    end
  end

  return nothing
end


# TODO: Taal remove?
# function (cb::DiscreteCallback{Condition,Affect!})(ode::ODEProblem) where {Condition, Affect!<:AMRCallback}
#   amr_callback = cb.affect!
#   semi = ode.p

#   @timeit_debug timer() "initial condition AMR" if amr_callback.adapt_initial_conditions
#     # iterate until mesh does not change anymore
#     has_changed = true
#     while has_changed
#       has_changed = amr_callback(ode.u0, semi,
#                                  only_refine=amr_callback.adapt_initial_conditions_only_refine)
#       compute_coefficients!(ode.u0, ode.tspan[1], semi)
#     end
#   end

#   return nothing
# end


function (amr_callback::AMRCallback)(integrator; kwargs...)
  u_ode = integrator.u
  semi = integrator.p

  @timeit_debug timer() "AMR" begin
    has_changed = amr_callback(u_ode, semi; kwargs...)
    if has_changed
      resize!(integrator, length(u_ode))
      u_modified!(integrator, true)
    end
  end

  return has_changed
end


@inline function (amr_callback::AMRCallback)(u_ode::AbstractVector, semi::SemidiscretizationHyperbolic; kwargs...)
  amr_callback(u_ode, mesh_equations_solver_cache(semi)...; kwargs...)
end


"""
    IndicatorThreeLevel(semi, indicator; base_level=1,
                                         med_level=base_level, med_threshold=0.0,
                                         max_level=base_level, max_threshold=1.0)

An AMR indicator based on three levels (in decending order of precedence):
- set the target level to `max_level` if `indicator > max_threshold`
- set the target level to `med_level` if `indicator > med_threshold`;
  if `med_level < 0`, set the target level to the current level
- set the target level to `base_level` otherwise
"""
struct IndicatorThreeLevel{RealT<:Real, Indicator, Cache}
  base_level::Int
  med_level ::Int
  max_level ::Int
  med_threshold::RealT
  max_threshold::RealT
  indicator::Indicator
  cache::Cache
end

function IndicatorThreeLevel(semi, indicator; base_level=1,
                                              med_level=base_level, med_threshold=0.0,
                                              max_level=base_level, max_threshold=1.0)
  med_threshold, max_threshold = promote(med_threshold, max_threshold)
  cache = create_cache(IndicatorThreeLevel, semi)
  IndicatorThreeLevel{typeof(max_threshold), typeof(indicator), typeof(cache)}(
    base_level, med_level, max_level, med_threshold, max_threshold, indicator, cache)
end

create_cache(indicator_type::Type{IndicatorThreeLevel}, semi) = create_cache(indicator_type, mesh_equations_solver_cache(semi)...)


function Base.show(io::IO, indicator::IndicatorThreeLevel)
  print(io, "IndicatorThreeLevel(")
  print(io, indicator.indicator)
  print(io, ", base_level=", indicator.base_level)
  print(io, ", med_level=",  indicator.med_level)
  print(io, ", max_level=",  indicator.max_level)
  print(io, ", med_threshold=", indicator.med_threshold)
  print(io, ", max_threshold=", indicator.max_threshold)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorThreeLevel)
  println(io, "IndicatorThreeLevel with")
  println(io, "- ", indicator.indicator)
  println(io, "- base_level: ", indicator.base_level)
  println(io, "- med_level:  ", indicator.med_level)
  println(io, "- max_level:  ", indicator.max_level)
  println(io, "- med_threshold: ", indicator.med_threshold)
  print(io,   "- max_threshold: ", indicator.max_threshold)
end


function get_element_variables!(element_variables, u, mesh, equations, solver, cache, indicator::IndicatorThreeLevel, amr_callback::AMRCallback)
  # call the indicator to get up-to-date values for IO
  indicator.indicator(u, equations, solver, cache)
  get_element_variables!(element_variables, indicator.indicator, amr_callback)
end

function get_element_variables!(element_variables, indicator::AbstractIndicator, ::AMRCallback)
  element_variables[:indicator_amr] = indicator.cache.alpha
  return nothing
end



# `passive_args` is currently used for Euler with self-gravity to adapt the gravity solver
# passively without querying its indicator, based on the assumption that both solvers use
# the same mesh. That's a hack and should be improved in the future once we have more examples
# and a better understandin of such a coupling.
# `passive_args` is expected to be an iterable of `Tuple`s of the form
# `(p_u_ode, p_mesh, p_equations, p_dg, p_cache)`.
function (amr_callback::AMRCallback)(u_ode::AbstractVector, mesh::TreeMesh,
                                     equations, dg::DG, cache;
                                     only_refine=false, only_coarsen=false,
                                     passive_args=())
  @unpack indicator, adaptor = amr_callback

  u = wrap_array(u_ode, mesh, equations, dg, cache)
  lambda = @timeit_debug timer() "indicator" indicator(u, mesh, equations, dg, cache)

  leaf_cell_ids = leaf_cells(mesh.tree)
  @boundscheck begin
   @assert axes(lambda) == axes(leaf_cell_ids) ("Indicator (axes = $(axes(lambda))) and leaf cell (axes = $(axes(leaf_cell_ids))) arrays have different axes")
  end

  to_refine  = Int[]
  to_coarsen = Int[]
  for element in eachelement(dg, cache)
    indicator_value = lambda[element]
    if indicator_value > 0
      push!(to_refine, leaf_cell_ids[element])
    elseif indicator_value < 0
      push!(to_coarsen, leaf_cell_ids[element])
    end
  end


  @timeit_debug timer() "refine" if !only_coarsen && !isempty(to_refine)
    # refine mesh
    refined_original_cells = @timeit_debug timer() "mesh" refine!(mesh.tree, to_refine)

    # refine solver
    @timeit_debug timer() "solver" refine!(u_ode, adaptor, mesh, equations, dg, cache, refined_original_cells)
    for (p_u_ode, p_mesh, p_equations, p_dg, p_cache) in passive_args
      @timeit_debug timer() "passive solver" refine!(p_u_ode, adaptor, p_mesh, p_equations, p_dg, p_cache, refined_original_cells)
    end
  else
    # If there is nothing to refine, create empty array for later use
    refined_original_cells = Int[]
  end


  @timeit_debug timer() "coarsen" if !only_refine && !isempty(to_coarsen)
    # Since the cells may have been shifted due to refinement, first we need to
    # translate the old cell ids to the new cell ids
    if !isempty(to_coarsen)
      to_coarsen = original2refined(to_coarsen, refined_original_cells, mesh)
    end

    # Next, determine the parent cells from which the fine cells are to be
    # removed, since these are needed for the coarsen! function. However, since
    # we only want to coarsen if *all* child cells are marked for coarsening,
    # we count the coarsening indicators for each parent cell and only coarsen
    # if all children are marked as such (i.e., where the count is 2^ndims). At
    # the same time, check if a cell is marked for coarsening even though it is
    # *not* a leaf cell -> this can only happen if it was refined due to 2:1
    # smoothing during the preceding refinement operation.
    parents_to_coarsen = zeros(Int, length(mesh.tree))
    for cell_id in to_coarsen
      # If cell has no parent, it cannot be coarsened
      if !has_parent(mesh.tree, cell_id)
        continue
      end

      # If cell is not leaf (anymore), it cannot be coarsened
      if !is_leaf(mesh.tree, cell_id)
        continue
      end

      # Increase count for parent cell
      parent_id = mesh.tree.parent_ids[cell_id]
      parents_to_coarsen[parent_id] += 1
    end

    # Extract only those parent cells for which all children should be coarsened
    to_coarsen = collect(1:length(parents_to_coarsen))[parents_to_coarsen .== 2^ndims(mesh)]

    # Finally, coarsen mesh
    coarsened_original_cells = @timeit_debug timer() "mesh" coarsen!(mesh.tree, to_coarsen)

    # Convert coarsened parent cell ids to the list of child cell ids that have
    # been removed, since this is the information that is expected by the solver
    removed_child_cells = zeros(Int, n_children_per_cell(mesh.tree) * length(coarsened_original_cells))
    for (index, coarse_cell_id) in enumerate(coarsened_original_cells)
      for child in 1:n_children_per_cell(mesh.tree)
        removed_child_cells[n_children_per_cell(mesh.tree) * (index-1) + child] = coarse_cell_id + child
      end
    end

    # coarsen solver
    @timeit_debug timer() "solver" coarsen!(u_ode, adaptor, mesh, equations, dg, cache, removed_child_cells)
    for (p_u_ode, p_mesh, p_equations, p_dg, p_cache) in passive_args
      @timeit_debug timer() "passive solver" coarsen!(p_u_ode, adaptor, p_mesh, p_equations, p_dg, p_cache, removed_child_cells)
    end
  else
    # If there is nothing to coarsen, create empty array for later use
    coarsened_original_cells = Int[]
  end

 # Return true if there were any cells coarsened or refined, otherwise false
 has_changed = !isempty(refined_original_cells) || !isempty(coarsened_original_cells)
 if has_changed # TODO: Taal decide, where shall we set this?
  # don't set it to has_changed since there can be changes from earlier calls
  mesh.unsaved_changes = true
 end

 return has_changed
end

# function original2refined(original_cell_ids, refined_original_cells, mesh) in src/amr/amr.jl


# TODO: Taal dimension agnostic
# TODO: Taal refactor, merge the two loops of IndicatorThreeLevel and IndicatorLöhner etc.?
#       But that would remove the simplest possibility to write that stuff to a file...
#       We could of course implement some additional logic and workarounds, but is it worth the effort?
function (indicator::IndicatorThreeLevel)(u::AbstractArray{<:Any},
                                          mesh::TreeMesh, equations, dg::DG, cache)

  @unpack indicator_value = indicator.cache
  resize!(indicator_value, nelements(dg, cache))

  alpha = indicator.indicator(u, equations, dg, cache)

  Threads.@threads for element in eachelement(dg, cache)
    cell_id = cache.elements.cell_ids[element]
    current_level = mesh.tree.levels[cell_id]

    # set target level
    target_level = current_level
    if alpha[element] > indicator.max_threshold
      target_level = indicator.max_level
    elseif alpha[element] > indicator.med_threshold
      if indicator.med_level > 0
        target_level = indicator.med_level
        # otherwise, target_level = current_level
        # set med_level = -1 to implicitly use med_level = current_level
      end
    else
      target_level = indicator.base_level
    end

    # compare target level with actual level to set indicator
    if current_level < target_level
      indicator_value[element] = 1 # refine!
    elseif current_level > target_level
      indicator_value[element] = -1 # coarsen!
    else
      indicator_value[element] = 0 # we're good
    end
  end

  return indicator_value
end


include("amr_dg1d.jl")
include("amr_dg2d.jl")
