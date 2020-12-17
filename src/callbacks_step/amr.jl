
"""
    AMRCallback(semi, controller [,adaptor=AdaptorAMR(semi)];
                interval,
                adapt_initial_condition=true,
                adapt_initial_condition_only_refine=true)

Performs adaptive mesh refinement (AMR) every `interval` time steps
for a given semidiscretization `semi` using the chosen `controller`.
"""
struct AMRCallback{Controller, Adaptor, Cache}
  controller::Controller
  interval::Int
  adapt_initial_condition::Bool
  adapt_initial_condition_only_refine::Bool
  adaptor::Adaptor
  amr_cache::Cache
end


function AMRCallback(semi, controller, adaptor;
                     interval,
                     adapt_initial_condition=true,
                     adapt_initial_condition_only_refine=true)
  # check arguments
  if !(interval isa Integer && interval >= 0)
    throw(ArgumentError("`interval` must be a non-negative integer (provided `interval = $interval`)"))
  end

  # AMR every `interval` time steps, but not after the final step
  if interval > 0
    condition = (u, t, integrator) -> integrator.iter % interval == 0 && !isfinished(integrator)
  else # disable the AMR callback except possibly for initial refinement during initialization
    condition = (u, t, integrator) -> false
  end

  to_refine  = Int[]
  to_coarsen = Int[]
  amr_cache = (; to_refine, to_coarsen)

  amr_callback = AMRCallback{typeof(controller), typeof(adaptor), typeof(amr_cache)}(
    controller, interval, adapt_initial_condition,
    adapt_initial_condition_only_refine, adaptor, amr_cache)

  DiscreteCallback(condition, amr_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end

function AMRCallback(semi, controller; kwargs...)
  adaptor = AdaptorAMR(semi)
  AMRCallback(semi, controller, adaptor; kwargs...)
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
function Base.show(io::IO, mime::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AMRCallback}
  if get(io, :compact, false)
    show(io, cb)
  else
    amr_callback = cb.affect!

    summary_header(io, "AMRCallback")
    summary_line(io, "controller", typeof(amr_callback.controller).name)
    show(increment_indent(io), mime, amr_callback.controller)
    summary_line(io, "interval", amr_callback.interval)
    summary_line(io, "adapt IC", amr_callback.adapt_initial_condition ? "yes" : "no",)
    if amr_callback.adapt_initial_condition
      summary_line(io, "│ only refine", amr_callback.adapt_initial_condition_only_refine ? "yes" : "no")
    end
    summary_footer(io)
  end
end


# The function below is used to control the output depending on whether or not AMR is enabled.
"""
    uses_amr(callback)

Checks whether the provided callback or `CallbackSet` is an [`AMRCallback`](@ref)
or contains one.
"""
uses_amr(cb) = false
uses_amr(cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AMRCallback} = true
uses_amr(callbacks::CallbackSet) = mapreduce(uses_amr, |, callbacks.discrete_callbacks)


function get_element_variables!(element_variables, u, mesh, equations, solver, cache,
                                amr_callback::AMRCallback; kwargs...)
  get_element_variables!(element_variables, u, mesh, equations, solver, cache,
                         amr_callback.controller, amr_callback; kwargs...)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:AMRCallback}
  amr_callback = cb.affect!
  semi = integrator.p

  @timeit_debug timer() "initial condition AMR" if amr_callback.adapt_initial_condition
    # iterate until mesh does not change anymore
    has_changed = amr_callback(integrator,
                               only_refine=amr_callback.adapt_initial_condition_only_refine)
    while has_changed
      compute_coefficients!(integrator.u, t, semi)
      u_modified!(integrator, true)
      has_changed = amr_callback(integrator,
                                 only_refine=amr_callback.adapt_initial_condition_only_refine)
    end
  end

  return nothing
end


# TODO: Taal remove?
# function (cb::DiscreteCallback{Condition,Affect!})(ode::ODEProblem) where {Condition, Affect!<:AMRCallback}
#   amr_callback = cb.affect!
#   semi = ode.p

#   @timeit_debug timer() "initial condition AMR" if amr_callback.adapt_initial_condition
#     # iterate until mesh does not change anymore
#     has_changed = true
#     while has_changed
#       has_changed = amr_callback(ode.u0, semi,
#                                  only_refine=amr_callback.adapt_initial_condition_only_refine)
#       compute_coefficients!(ode.u0, ode.tspan[1], semi)
#     end
#   end

#   return nothing
# end


function (amr_callback::AMRCallback)(integrator; kwargs...)
  u_ode = integrator.u
  semi = integrator.p

  @timeit_debug timer() "AMR" begin
    has_changed = amr_callback(u_ode, semi,
                               integrator.t, integrator.iter; kwargs...)
    if has_changed
      resize!(integrator, length(u_ode))
      u_modified!(integrator, true)
    end
  end

  return has_changed
end


@inline function (amr_callback::AMRCallback)(u_ode::AbstractVector,
                                             semi::SemidiscretizationHyperbolic,
                                             t, iter;
                                             kwargs...)
  amr_callback(u_ode, mesh_equations_solver_cache(semi)..., t, iter; kwargs...)
end



# `passive_args` is currently used for Euler with self-gravity to adapt the gravity solver
# passively without querying its indicator, based on the assumption that both solvers use
# the same mesh. That's a hack and should be improved in the future once we have more examples
# and a better understandin of such a coupling.
# `passive_args` is expected to be an iterable of `Tuple`s of the form
# `(p_u_ode, p_mesh, p_equations, p_dg, p_cache)`.
function (amr_callback::AMRCallback)(u_ode::AbstractVector, mesh::TreeMesh,
                                     equations, dg::DG, cache,
                                     t, iter;
                                     only_refine=false, only_coarsen=false,
                                     passive_args=())
  @unpack controller, adaptor = amr_callback

  u = wrap_array(u_ode, mesh, equations, dg, cache)
  lambda = @timeit_debug timer() "indicator" controller(u, mesh, equations, dg, cache,
                                                        t=t, iter=iter)

  if mpi_isparallel()
    # Collect lambda for all elements
    lambda_global = Vector{eltype(lambda)}(undef, nelementsglobal(dg, cache))
    # Use parent because n_elements_by_rank is an OffsetArray
    recvbuf = MPI.VBuffer(lambda_global, parent(cache.mpi_cache.n_elements_by_rank))
    MPI.Allgatherv!(lambda, recvbuf, mpi_comm())
    lambda = lambda_global
  end

  leaf_cell_ids = leaf_cells(mesh.tree)
  @boundscheck begin
   @assert axes(lambda) == axes(leaf_cell_ids) ("Indicator (axes = $(axes(lambda))) and leaf cell (axes = $(axes(leaf_cell_ids))) arrays have different axes")
  end

  @unpack to_refine, to_coarsen = amr_callback.amr_cache
  empty!(to_refine)
  empty!(to_coarsen)
  for element in 1:length(lambda)
    controller_value = lambda[element]
    if controller_value > 0
      push!(to_refine, leaf_cell_ids[element])
    elseif controller_value < 0
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


# After refining cells, shift original cell ids to match new locations
# Note: Assumes sorted lists of original and refined cell ids!
# Note: `mesh` is only required to extract ndims
function original2refined(original_cell_ids, refined_original_cells, mesh)
  # Sanity check
  @assert issorted(original_cell_ids) "`original_cell_ids` not sorted"
  @assert issorted(refined_original_cells) "`refined_cell_ids` not sorted"

  # Create array with original cell ids (not yet shifted)
  shifted_cell_ids = collect(1:original_cell_ids[end])

  # Loop over refined original cells and apply shift for all following cells
  for cell_id in refined_original_cells
    # Only calculate shifts for cell ids that are relevant
    if cell_id > length(shifted_cell_ids)
      break
    end

    # Shift all subsequent cells by 2^ndims ids
    shifted_cell_ids[(cell_id + 1):end] .+= 2^ndims(mesh)
  end

  # Convert original cell ids to their shifted values
  return shifted_cell_ids[original_cell_ids]
end



"""
    ControllerThreeLevel(semi, indicator; base_level=1,
                                          med_level=base_level, med_threshold=0.0,
                                          max_level=base_level, max_threshold=1.0)

An AMR controller based on three levels (in decending order of precedence):
- set the target level to `max_level` if `indicator > max_threshold`
- set the target level to `med_level` if `indicator > med_threshold`;
  if `med_level < 0`, set the target level to the current level
- set the target level to `base_level` otherwise
"""
struct ControllerThreeLevel{RealT<:Real, Indicator, Cache}
  base_level::Int
  med_level ::Int
  max_level ::Int
  med_threshold::RealT
  max_threshold::RealT
  indicator::Indicator
  cache::Cache
end

function ControllerThreeLevel(semi, indicator; base_level=1,
                                               med_level=base_level, med_threshold=0.0,
                                               max_level=base_level, max_threshold=1.0)
  med_threshold, max_threshold = promote(med_threshold, max_threshold)
  cache = create_cache(ControllerThreeLevel, semi)
  ControllerThreeLevel{typeof(max_threshold), typeof(indicator), typeof(cache)}(
    base_level, med_level, max_level, med_threshold, max_threshold, indicator, cache)
end

create_cache(indicator_type::Type{ControllerThreeLevel}, semi) = create_cache(indicator_type, mesh_equations_solver_cache(semi)...)


function Base.show(io::IO, controller::ControllerThreeLevel)
  print(io, "ControllerThreeLevel(")
  print(io, controller.indicator)
  print(io, ", base_level=", controller.base_level)
  print(io, ", med_level=",  controller.med_level)
  print(io, ", max_level=",  controller.max_level)
  print(io, ", med_threshold=", controller.med_threshold)
  print(io, ", max_threshold=", controller.max_threshold)
  print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", controller::ControllerThreeLevel)
  if get(io, :compact, false)
    show(io, controller)
  else
    summary_header(io, "ControllerThreeLevel")
    summary_line(io, "indicator", typeof(controller.indicator).name)
    show(increment_indent(io), mime, controller.indicator)
    summary_line(io, "base_level", controller.base_level)
    summary_line(io, "med_level", controller.med_level)
    summary_line(io, "max_level", controller.max_level)
    summary_line(io, "med_threshold", controller.med_threshold)
    summary_line(io, "max_threshold", controller.max_threshold)
    summary_footer(io)
  end
end


function get_element_variables!(element_variables, u, mesh, equations, solver, cache,
                                controller::ControllerThreeLevel, amr_callback::AMRCallback;
                                kwargs...)
  # call the indicator to get up-to-date values for IO
  controller.indicator(u, equations, solver, cache; kwargs...)
  get_element_variables!(element_variables, controller.indicator, amr_callback)
end

function get_element_variables!(element_variables, indicator::AbstractIndicator, ::AMRCallback)
  element_variables[:indicator_amr] = indicator.cache.alpha
  return nothing
end


# TODO: Taal refactor, merge the two loops of ControllerThreeLevel and IndicatorLöhner etc.?
#       But that would remove the simplest possibility to write that stuff to a file...
#       We could of course implement some additional logic and workarounds, but is it worth the effort?
function (controller::ControllerThreeLevel)(u::AbstractArray{<:Any},
                                            mesh::TreeMesh, equations, dg::DG, cache;
                                            kwargs...)

  @unpack controller_value = controller.cache
  resize!(controller_value, nelements(dg, cache))

  alpha = controller.indicator(u, equations, dg, cache; kwargs...)

  Threads.@threads for element in eachelement(dg, cache)
    cell_id = cache.elements.cell_ids[element]
    current_level = mesh.tree.levels[cell_id]

    # set target level
    target_level = current_level
    if alpha[element] > controller.max_threshold
      target_level = controller.max_level
    elseif alpha[element] > controller.med_threshold
      if controller.med_level > 0
        target_level = controller.med_level
        # otherwise, target_level = current_level
        # set med_level = -1 to implicitly use med_level = current_level
      end
    else
      target_level = controller.base_level
    end

    # compare target level with actual level to set controller
    if current_level < target_level
      controller_value[element] = 1 # refine!
    elseif current_level > target_level
      controller_value[element] = -1 # coarsen!
    else
      controller_value[element] = 0 # we're good
    end
  end

  return controller_value
end



"""
    ControllerThreeLevelCombined(semi, indicator_primary, indicator_secondary;
                                 base_level=1,
                                 med_level=base_level, med_threshold=0.0,
                                 max_level=base_level, max_threshold=1.0,
                                 max_threshold_secondary=1.0)

An AMR controller based on three levels (in decending order of precedence):
- set the target level to `max_level` if `indicator_primary > max_threshold`
- set the target level to `med_level` if `indicator_primary > med_threshold`;
  if `med_level < 0`, set the target level to the current level
- set the target level to `base_level` otherwise
If `indicator_secondary >= max_threshold_secondary`,
set the target level to `max_level`.
"""
struct ControllerThreeLevelCombined{RealT<:Real, IndicatorPrimary, IndicatorSecondary, Cache}
  base_level::Int
  med_level ::Int
  max_level ::Int
  med_threshold::RealT
  max_threshold::RealT
  max_threshold_secondary::RealT
  indicator_primary::IndicatorPrimary
  indicator_secondary::IndicatorSecondary
  cache::Cache
end

function ControllerThreeLevelCombined(semi, indicator_primary, indicator_secondary;
                                      base_level=1,
                                      med_level=base_level, med_threshold=0.0,
                                      max_level=base_level, max_threshold=1.0,
                                      max_threshold_secondary=1.0)
  med_threshold, max_threshold, max_threshold_secondary = promote(med_threshold, max_threshold, max_threshold_secondary)
  cache = create_cache(ControllerThreeLevelCombined, semi)
  ControllerThreeLevelCombined{typeof(max_threshold), typeof(indicator_primary), typeof(indicator_secondary), typeof(cache)}(
    base_level, med_level, max_level, med_threshold, max_threshold,
    max_threshold_secondary, indicator_primary, indicator_secondary, cache)
end

create_cache(indicator_type::Type{ControllerThreeLevelCombined}, semi) = create_cache(indicator_type, mesh_equations_solver_cache(semi)...)


function Base.show(io::IO, controller::ControllerThreeLevelCombined)
  print(io, "ControllerThreeLevelCombined(")
  print(io, controller.indicator_primary)
  print(io, ", ", controller.indicator_secondary)
  print(io, ", base_level=", controller.base_level)
  print(io, ", med_level=",  controller.med_level)
  print(io, ", max_level=",  controller.max_level)
  print(io, ", med_threshold=", controller.med_threshold)
  print(io, ", max_threshold_secondary=", controller.max_threshold_secondary)
  print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", controller::ControllerThreeLevelCombined)
  if get(io, :compact, false)
    show(io, controller)
  else
    summary_header(io, "ControllerThreeLevelCombined")
    summary_line(io, "primary indicator", typeof(controller.indicator_primary).name)
    show(increment_indent(io), mime, controller.indicator_primary)
    summary_line(io, "secondary indicator", typeof(controller.indicator_secondary).name)
    show(increment_indent(io), mime, controller.indicator_secondary)
    summary_line(io, "base_level", controller.base_level)
    summary_line(io, "med_level", controller.med_level)
    summary_line(io, "max_level", controller.max_level)
    summary_line(io, "med_threshold", controller.med_threshold)
    summary_line(io, "max_threshold", controller.max_threshold)
    summary_line(io, "max_threshold_secondary", controller.max_threshold_secondary)
    summary_footer(io)
  end
end


function get_element_variables!(element_variables, u, mesh, equations, solver, cache,
                                controller::ControllerThreeLevelCombined, amr_callback::AMRCallback;
                                kwargs...)
  # call the indicator to get up-to-date values for IO
  controller.indicator_primary(u, equations, solver, cache; kwargs...)
  get_element_variables!(element_variables, controller.indicator_primary, amr_callback)
end


function (controller::ControllerThreeLevelCombined)(u::AbstractArray{<:Any},
                                                    mesh::TreeMesh, equations, dg::DG, cache;
                                                    kwargs...)

  @unpack controller_value = controller.cache
  resize!(controller_value, nelements(dg, cache))

  alpha = controller.indicator_primary(u, equations, dg, cache; kwargs...)
  alpha_secondary = controller.indicator_secondary(u, equations, dg, cache)

  Threads.@threads for element in eachelement(dg, cache)
    cell_id = cache.elements.cell_ids[element]
    current_level = mesh.tree.levels[cell_id]

    # set target level
    target_level = current_level
    if alpha[element] > controller.max_threshold
      target_level = controller.max_level
    elseif alpha[element] > controller.med_threshold
      if controller.med_level > 0
        target_level = controller.med_level
        # otherwise, target_level = current_level
        # set med_level = -1 to implicitly use med_level = current_level
      end
    else
      target_level = controller.base_level
    end

    if alpha_secondary[element] >= controller.max_threshold_secondary
      target_level = controller.max_level
    end

    # compare target level with actual level to set controller
    if current_level < target_level
      controller_value[element] = 1 # refine!
    elseif current_level > target_level
      controller_value[element] = -1 # coarsen!
    else
      controller_value[element] = 0 # we're good
    end
  end

  return controller_value
end


include("amr_dg1d.jl")
include("amr_dg2d.jl")
include("amr_dg3d.jl")
