# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    AMRCallback(semi, controller [,adaptor=AdaptorAMR(semi)];
                interval,
                adapt_initial_condition=true,
                adapt_initial_condition_only_refine=true,
                dynamic_load_balancing=true)

Performs adaptive mesh refinement (AMR) every `interval` time steps
for a given semidiscretization `semi` using the chosen `controller`.
"""
struct AMRCallback{Controller, Adaptor, Cache}
  controller::Controller
  interval::Int
  adapt_initial_condition::Bool
  adapt_initial_condition_only_refine::Bool
  dynamic_load_balancing::Bool
  adaptor::Adaptor
  amr_cache::Cache
end


function AMRCallback(semi, controller, adaptor;
                     interval,
                     adapt_initial_condition=true,
                     adapt_initial_condition_only_refine=true,
                     dynamic_load_balancing=true)
  # check arguments
  if !(interval isa Integer && interval >= 0)
    throw(ArgumentError("`interval` must be a non-negative integer (provided `interval = $interval`)"))
  end

  # AMR every `interval` time steps, but not after the final step
  # With error-based step size control, some steps can be rejected. Thus,
  #   `integrator.iter >= integrator.stats.naccept`
  #    (total #steps)       (#accepted steps)
  # We need to check the number of accepted steps since callbacks are not
  # activated after a rejected step.
  if interval > 0
    condition = (u, t, integrator) -> ( (integrator.stats.naccept % interval == 0) &&
                                        !(integrator.stats.naccept == 0 && integrator.iter > 0) &&
                                        !isfinished(integrator) )
  else # disable the AMR callback except possibly for initial refinement during initialization
    condition = (u, t, integrator) -> false
  end

  to_refine  = Int[]
  to_coarsen = Int[]
  amr_cache = (; to_refine, to_coarsen)

  amr_callback = AMRCallback{typeof(controller), typeof(adaptor), typeof(amr_cache)}(
    controller, interval, adapt_initial_condition, adapt_initial_condition_only_refine,
    dynamic_load_balancing, adaptor, amr_cache)

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
# function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:AMRCallback})
#   @nospecialize cb # reduce precompilation time
#
#   amr_callback = cb.affect!
#   print(io, "AMRCallback")
# end
function Base.show(io::IO, mime::MIME"text/plain", cb::DiscreteCallback{<:Any, <:AMRCallback})
  @nospecialize cb # reduce precompilation time

  if get(io, :compact, false)
    show(io, cb)
  else
    amr_callback = cb.affect!

    summary_header(io, "AMRCallback")
    summary_line(io, "controller", amr_callback.controller |> typeof |> nameof)
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

  @trixi_timeit timer() "initial condition AMR" if amr_callback.adapt_initial_condition
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

#   @trixi_timeit timer() "initial condition AMR" if amr_callback.adapt_initial_condition
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

  @trixi_timeit timer() "AMR" begin
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
  # Note that we don't `wrap_array` the vector `u_ode` to be able to `resize!`
  # it when doing AMR while still dispatching on the `mesh` etc.
  amr_callback(u_ode, mesh_equations_solver_cache(semi)..., semi, t, iter; kwargs...)
end



# `passive_args` is currently used for Euler with self-gravity to adapt the gravity solver
# passively without querying its indicator, based on the assumption that both solvers use
# the same mesh. That's a hack and should be improved in the future once we have more examples
# and a better understanding of such a coupling.
# `passive_args` is expected to be an iterable of `Tuple`s of the form
# `(p_u_ode, p_mesh, p_equations, p_dg, p_cache)`.
function (amr_callback::AMRCallback)(u_ode::AbstractVector, mesh::TreeMesh,
                                     equations, dg::DG, cache, semi,
                                     t, iter;
                                     only_refine=false, only_coarsen=false,
                                     passive_args=())
  @unpack controller, adaptor = amr_callback

  u = wrap_array(u_ode, mesh, equations, dg, cache)
  lambda = @trixi_timeit timer() "indicator" controller(u, mesh, equations, dg, cache,
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


  @trixi_timeit timer() "refine" if !only_coarsen && !isempty(to_refine)
    # refine mesh
    refined_original_cells = @trixi_timeit timer() "mesh" refine!(mesh.tree, to_refine)

    # Find all indices of elements whose cell ids are in refined_original_cells
    elements_to_refine = findall(in(refined_original_cells), cache.elements.cell_ids)

    # refine solver
    @trixi_timeit timer() "solver" refine!(u_ode, adaptor, mesh, equations, dg, cache, elements_to_refine)
    for (p_u_ode, p_mesh, p_equations, p_dg, p_cache) in passive_args
      @trixi_timeit timer() "passive solver" refine!(p_u_ode, adaptor, p_mesh, p_equations, p_dg, p_cache, elements_to_refine)
    end
  else
    # If there is nothing to refine, create empty array for later use
    refined_original_cells = Int[]
  end


  @trixi_timeit timer() "coarsen" if !only_refine && !isempty(to_coarsen)
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
    coarsened_original_cells = @trixi_timeit timer() "mesh" coarsen!(mesh.tree, to_coarsen)

    # Convert coarsened parent cell ids to the list of child cell ids that have
    # been removed, since this is the information that is expected by the solver
    removed_child_cells = zeros(Int, n_children_per_cell(mesh.tree) * length(coarsened_original_cells))
    for (index, coarse_cell_id) in enumerate(coarsened_original_cells)
      for child in 1:n_children_per_cell(mesh.tree)
        removed_child_cells[n_children_per_cell(mesh.tree) * (index-1) + child] = coarse_cell_id + child
      end
    end

    # Find all indices of elements whose cell ids are in removed_child_cells
    elements_to_remove = findall(in(removed_child_cells), cache.elements.cell_ids)

    # coarsen solver
    @trixi_timeit timer() "solver" coarsen!(u_ode, adaptor, mesh, equations, dg, cache, elements_to_remove)
    for (p_u_ode, p_mesh, p_equations, p_dg, p_cache) in passive_args
      @trixi_timeit timer() "passive solver" coarsen!(p_u_ode, adaptor, p_mesh, p_equations, p_dg, p_cache, elements_to_remove)
    end
  else
    # If there is nothing to coarsen, create empty array for later use
    coarsened_original_cells = Int[]
  end

  # Store whether there were any cells coarsened or refined
  has_changed = !isempty(refined_original_cells) || !isempty(coarsened_original_cells)
  if has_changed # TODO: Taal decide, where shall we set this?
    # don't set it to has_changed since there can be changes from earlier calls
    mesh.unsaved_changes = true
  end

  # Dynamically balance computational load by first repartitioning the mesh and then redistributing the cells/elements
  if has_changed && mpi_isparallel() && amr_callback.dynamic_load_balancing
    @trixi_timeit timer() "dynamic load balancing" begin
      old_mpi_ranks_per_cell = copy(mesh.tree.mpi_ranks)

      partition!(mesh)

      rebalance_solver!(u_ode, mesh, equations, dg, cache, old_mpi_ranks_per_cell)
    end
  end

  # Return true if there were any cells coarsened or refined, otherwise false
  return has_changed
end


# Copy controller values to quad user data storage, will be called below
function copy_to_quad_iter_volume(info, user_data)
  info_obj = unsafe_load(info)

  # Load tree from global trees array, one-based indexing
  tree = unsafe_load_tree(info_obj.p4est, info_obj.treeid + 1)
  # Quadrant numbering offset of this quadrant
  offset = tree.quadrants_offset
  # Global quad ID
  quad_id = offset + info_obj.quadid

  # Access user_data = lambda
  user_data_ptr = Ptr{Int}(user_data)
  # Load controller_value = lambda[quad_id + 1]
  controller_value = unsafe_load(user_data_ptr, quad_id + 1)

  # Access quadrant's user data ([global quad ID, controller_value])
  quad_data_ptr = Ptr{Int}(unsafe_load(info_obj.quad.p.user_data))
  # Save controller value to quadrant's user data.
  unsafe_store!(quad_data_ptr, controller_value, 2)

  return nothing
end

# 2D
cfunction(::typeof(copy_to_quad_iter_volume), ::Val{2}) = @cfunction(copy_to_quad_iter_volume, Cvoid, (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))
# 3D
cfunction(::typeof(copy_to_quad_iter_volume), ::Val{3}) = @cfunction(copy_to_quad_iter_volume, Cvoid, (Ptr{p8est_iter_volume_info_t}, Ptr{Cvoid}))

function (amr_callback::AMRCallback)(u_ode::AbstractVector, mesh::P4estMesh,
                                     equations, dg::DG, cache, semi,
                                     t, iter;
                                     only_refine=false, only_coarsen=false,
                                     passive_args=())
  @unpack controller, adaptor = amr_callback

  u = wrap_array(u_ode, mesh, equations, dg, cache)
  lambda = @trixi_timeit timer() "indicator" controller(u, mesh, equations, dg, cache,
                                                 t=t, iter=iter)

  @boundscheck begin
    @assert axes(lambda) == (Base.OneTo(ncells(mesh)),) (
      "Indicator array (axes = $(axes(lambda))) and mesh cells (axes = $(Base.OneTo(ncells(mesh)))) have different axes"
    )
  end

  # Copy controller value of each quad to the quad's user data storage
  iter_volume_c = cfunction(copy_to_quad_iter_volume, Val(ndims(mesh)))

  # The pointer to lambda will be interpreted as Ptr{Int} above
  @assert lambda isa Vector{Int}
  iterate_p4est(mesh.p4est, lambda; iter_volume_c=iter_volume_c)

  @trixi_timeit timer() "refine" if !only_coarsen
    # Refine mesh
    refined_original_cells = @trixi_timeit timer() "mesh" refine!(mesh)

    # Refine solver
    @trixi_timeit timer() "solver" refine!(u_ode, adaptor, mesh, equations, dg, cache,
                                    refined_original_cells)
    for (p_u_ode, p_mesh, p_equations, p_dg, p_cache) in passive_args
      @trixi_timeit timer() "passive solver" refine!(p_u_ode, adaptor, p_mesh, p_equations,
                                              p_dg, p_cache, refined_original_cells)
    end
  else
    # If there is nothing to refine, create empty array for later use
    refined_original_cells = Int[]
  end

  @trixi_timeit timer() "coarsen" if !only_refine
    # Coarsen mesh
    coarsened_original_cells = @trixi_timeit timer() "mesh" coarsen!(mesh)

    # coarsen solver
    @trixi_timeit timer() "solver" coarsen!(u_ode, adaptor, mesh, equations, dg, cache,
                                     coarsened_original_cells)
    for (p_u_ode, p_mesh, p_equations, p_dg, p_cache) in passive_args
      @trixi_timeit timer() "passive solver" coarsen!(p_u_ode, adaptor, p_mesh, p_equations,
                                               p_dg, p_cache, coarsened_original_cells)
    end
  else
    # If there is nothing to coarsen, create empty array for later use
    coarsened_original_cells = Int[]
  end

  # Store whether there were any cells coarsened or refined and perform load balancing
  has_changed = !isempty(refined_original_cells) || !isempty(coarsened_original_cells)
  # Check if mesh changed on other processes
  if mpi_isparallel()
    has_changed = MPI.Allreduce!(Ref(has_changed), |, mpi_comm())[]
  end

  if has_changed # TODO: Taal decide, where shall we set this?
    # don't set it to has_changed since there can be changes from earlier calls
    mesh.unsaved_changes = true

    if mpi_isparallel() && amr_callback.dynamic_load_balancing
      @trixi_timeit timer() "dynamic load balancing" begin
        global_first_quadrant = unsafe_wrap(Array, unsafe_load(mesh.p4est).global_first_quadrant, mpi_nranks() + 1)
        old_global_first_quadrant = copy(global_first_quadrant)
        partition!(mesh)
        rebalance_solver!(u_ode, mesh, equations, dg, cache, old_global_first_quadrant)
      end
    end

    reinitialize_boundaries!(semi.boundary_conditions, cache)
  end

  # Return true if there were any cells coarsened or refined, otherwise false
  return has_changed
end

function reinitialize_boundaries!(boundary_conditions::UnstructuredSortedBoundaryTypes, cache)
  # Reinitialize boundary types container because boundaries may have changed.
  initialize!(boundary_conditions, cache)
end

function reinitialize_boundaries!(boundary_conditions, cache)
  return boundary_conditions
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

An AMR controller based on three levels (in descending order of precedence):
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
  @nospecialize controller # reduce precompilation time

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
  @nospecialize controller # reduce precompilation time

  if get(io, :compact, false)
    show(io, controller)
  else
    summary_header(io, "ControllerThreeLevel")
    summary_line(io, "indicator", controller.indicator |> typeof |> nameof)
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
  controller.indicator(u, mesh, equations, solver, cache; kwargs...)
  get_element_variables!(element_variables, controller.indicator, amr_callback)
end

function get_element_variables!(element_variables, indicator::AbstractIndicator, ::AMRCallback)
  element_variables[:indicator_amr] = indicator.cache.alpha
  return nothing
end


function current_element_levels(mesh::TreeMesh, solver, cache)
  cell_ids = cache.elements.cell_ids[eachelement(solver, cache)]

  return mesh.tree.levels[cell_ids]
end


function extract_levels_iter_volume(info, user_data)
  info_obj = unsafe_load(info)

  # Load tree from global trees array, one-based indexing
  tree = unsafe_load_tree(info_obj.p4est, info_obj.treeid + 1)
  # Quadrant numbering offset of this quadrant
  offset = tree.quadrants_offset
  # Global quad ID
  quad_id = offset + info_obj.quadid
  # Julia element ID
  element_id = quad_id + 1

  current_level = unsafe_load(info_obj.quad.level)

  # Unpack user_data = current_levels and save current element level
  ptr = Ptr{Int}(user_data)
  unsafe_store!(ptr, current_level, element_id)

  return nothing
end

# 2D
cfunction(::typeof(extract_levels_iter_volume), ::Val{2}) = @cfunction(extract_levels_iter_volume, Cvoid, (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))
# 3D
cfunction(::typeof(extract_levels_iter_volume), ::Val{3}) = @cfunction(extract_levels_iter_volume, Cvoid, (Ptr{p8est_iter_volume_info_t}, Ptr{Cvoid}))

function current_element_levels(mesh::P4estMesh, solver, cache)
  current_levels = Vector{Int}(undef, nelements(solver, cache))

  iter_volume_c = cfunction(extract_levels_iter_volume, Val(ndims(mesh)))
  iterate_p4est(mesh.p4est, current_levels; iter_volume_c=iter_volume_c)

  return current_levels
end


# TODO: Taal refactor, merge the two loops of ControllerThreeLevel and IndicatorLöhner etc.?
#       But that would remove the simplest possibility to write that stuff to a file...
#       We could of course implement some additional logic and workarounds, but is it worth the effort?
function (controller::ControllerThreeLevel)(u::AbstractArray{<:Any},
                                            mesh, equations, dg::DG, cache;
                                            kwargs...)

  @unpack controller_value = controller.cache
  resize!(controller_value, nelements(dg, cache))

  alpha = controller.indicator(u, mesh, equations, dg, cache; kwargs...)
  current_levels = current_element_levels(mesh, dg, cache)

  @threaded for element in eachelement(dg, cache)
    current_level = current_levels[element]

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

An AMR controller based on three levels (in descending order of precedence):
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
  @nospecialize controller # reduce precompilation time

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
  @nospecialize controller # reduce precompilation time

  if get(io, :compact, false)
    show(io, controller)
  else
    summary_header(io, "ControllerThreeLevelCombined")
    summary_line(io, "primary indicator", controller.indicator_primary |> typeof |> nameof)
    show(increment_indent(io), mime, controller.indicator_primary)
    summary_line(io, "secondary indicator", controller.indicator_secondary |> typeof |> nameof)
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
  controller.indicator_primary(u, mesh, equations, solver, cache; kwargs...)
  get_element_variables!(element_variables, controller.indicator_primary, amr_callback)
end


function (controller::ControllerThreeLevelCombined)(u::AbstractArray{<:Any},
                                                    mesh, equations, dg::DG, cache;
                                                    kwargs...)

  @unpack controller_value = controller.cache
  resize!(controller_value, nelements(dg, cache))

  alpha = controller.indicator_primary(u, mesh, equations, dg, cache; kwargs...)
  alpha_secondary = controller.indicator_secondary(u, mesh, equations, dg, cache)

  current_levels = current_element_levels(mesh, dg, cache)

  @threaded for element in eachelement(dg, cache)
    current_level = current_levels[element]

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


include("amr_dg.jl")
include("amr_dg1d.jl")
include("amr_dg2d.jl")
include("amr_dg3d.jl")


end # @muladd
