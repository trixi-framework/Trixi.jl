# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    SaveSolutionCallback(; interval::Integer=0,
                           dt=nothing,
                           save_initial_solution=true,
                           save_final_solution=true,
                           output_directory="out",
                           solution_variables=cons2prim)

Save the current numerical solution in regular intervals. Either pass `interval` to save
every `interval` time steps or pass `dt` to save in intervals of `dt` in terms
of integration time by adding additional (shortened) time steps where necessary (note that this may change the solution).
`solution_variables` can be any callable that converts the conservative variables
at a single point to a set of solution variables. The first parameter passed
to `solution_variables` will be the set of conservative variables
and the second parameter is the equation struct.
"""
mutable struct SaveSolutionCallback{IntervalType, SolutionVariablesType}
  interval_or_dt::IntervalType
  save_initial_solution::Bool
  save_final_solution::Bool
  output_directory::String
  solution_variables::SolutionVariablesType
end


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SaveSolutionCallback})
  @nospecialize cb # reduce precompilation time

  save_solution_callback = cb.affect!
  print(io, "SaveSolutionCallback(interval=", save_solution_callback.interval_or_dt, ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any, <:PeriodicCallbackAffect{<:SaveSolutionCallback}})
  @nospecialize cb # reduce precompilation time

  save_solution_callback = cb.affect!.affect!
  print(io, "SaveSolutionCallback(dt=", save_solution_callback.interval_or_dt, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:SaveSolutionCallback})
  @nospecialize cb # reduce precompilation time

  if get(io, :compact, false)
    show(io, cb)
  else
    save_solution_callback = cb.affect!

    setup = [
             "interval" => save_solution_callback.interval_or_dt,
             "solution variables" => save_solution_callback.solution_variables,
             "save initial solution" => save_solution_callback.save_initial_solution ? "yes" : "no",
             "save final solution" => save_solution_callback.save_final_solution ? "yes" : "no",
             "output directory" => abspath(normpath(save_solution_callback.output_directory)),
            ]
    summary_box(io, "SaveSolutionCallback", setup)
  end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:PeriodicCallbackAffect{<:SaveSolutionCallback}})
  @nospecialize cb # reduce precompilation time

  if get(io, :compact, false)
    show(io, cb)
  else
    save_solution_callback = cb.affect!.affect!

    setup = [
             "dt" => save_solution_callback.interval_or_dt,
             "solution variables" => save_solution_callback.solution_variables,
             "save initial solution" => save_solution_callback.save_initial_solution ? "yes" : "no",
             "save final solution" => save_solution_callback.save_final_solution ? "yes" : "no",
             "output directory" => abspath(normpath(save_solution_callback.output_directory)),
            ]
    summary_box(io, "SaveSolutionCallback", setup)
  end
end


function SaveSolutionCallback(; interval::Integer=0,
                                dt=nothing,
                                save_initial_solution=true,
                                save_final_solution=true,
                                output_directory="out",
                                solution_variables=cons2prim)

  if !isnothing(dt) && interval > 0
    throw(ArgumentError("You can either set the number of steps between output (using `interval`) or the time between outputs (using `dt`) but not both simultaneously"))
  end

  # Expected most frequent behavior comes first
  if isnothing(dt)
    interval_or_dt = interval
  else # !isnothing(dt)
    interval_or_dt = dt
  end

  solution_callback = SaveSolutionCallback(interval_or_dt,
                                           save_initial_solution, save_final_solution,
                                           output_directory, solution_variables)

  # Expected most frequent behavior comes first
  if isnothing(dt)
    # Save every `interval` (accepted) time steps
    # The first one is the condition, the second the affect!
    return DiscreteCallback(solution_callback, solution_callback,
                            save_positions=(false,false),
                            initialize=initialize_save_cb!)
  else
    # Add a `tstop` every `dt`, and save the final solution.
    return PeriodicCallback(solution_callback, dt,
                            save_positions=(false, false),
                            initialize=initialize_save_cb!,
                            final_affect=save_final_solution)
  end
end


function initialize_save_cb!(cb, u, t, integrator)
  # The SaveSolutionCallback is either cb.affect! (with DiscreteCallback)
  # or cb.affect!.affect! (with PeriodicCallback).
  # Let recursive dispatch handle this.
  initialize_save_cb!(cb.affect!, u, t, integrator)
end

function initialize_save_cb!(solution_callback::SaveSolutionCallback, u, t, integrator)
  mpi_isroot() && mkpath(solution_callback.output_directory)

  semi = integrator.p
  mesh, _, _, _ = mesh_equations_solver_cache(semi)
  @trixi_timeit timer() "I/O" begin
    if mesh.unsaved_changes
      mesh.current_filename = save_mesh_file(mesh, solution_callback.output_directory)
      mesh.unsaved_changes = false
    end
  end

  if solution_callback.save_initial_solution
    solution_callback(integrator)
  end

  return nothing
end


# this method is called to determine whether the callback should be activated
function (solution_callback::SaveSolutionCallback)(u, t, integrator)
  @unpack interval_or_dt, save_final_solution = solution_callback

  # With error-based step size control, some steps can be rejected. Thus,
  #   `integrator.iter >= integrator.stats.naccept`
  #    (total #steps)       (#accepted steps)
  # We need to check the number of accepted steps since callbacks are not
  # activated after a rejected step.
  return interval_or_dt > 0 && (
    ((integrator.stats.naccept % interval_or_dt == 0) && !(integrator.stats.naccept == 0 && integrator.iter > 0)) ||
    (save_final_solution && isfinished(integrator)))
end


# this method is called when the callback is activated
function (solution_callback::SaveSolutionCallback)(integrator)
  u_ode = integrator.u
  @unpack t, dt = integrator
  iter = integrator.stats.naccept
  semi = integrator.p
  mesh, _, _, _ = mesh_equations_solver_cache(semi)

  @trixi_timeit timer() "I/O" begin
    @trixi_timeit timer() "save mesh" if mesh.unsaved_changes
      mesh.current_filename = save_mesh_file(mesh, solution_callback.output_directory, iter)
      mesh.unsaved_changes = false
    end

    element_variables = Dict{Symbol, Any}()
    @trixi_timeit timer() "get element variables" begin
      get_element_variables!(element_variables, u_ode, semi)
      callbacks = integrator.opts.callback
      if callbacks isa CallbackSet
        for cb in callbacks.continuous_callbacks
          get_element_variables!(element_variables, u_ode, semi, cb; t=integrator.t, iter=integrator.stats.naccept)
        end
        for cb in callbacks.discrete_callbacks
          get_element_variables!(element_variables, u_ode, semi, cb; t=integrator.t, iter=integrator.stats.naccept)
        end
      end
    end

    @trixi_timeit timer() "save solution" save_solution_file(u_ode, t, dt, iter, semi, solution_callback, element_variables)
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)
  return nothing
end


@inline function save_solution_file(u_ode, t, dt, iter,
                                    semi::AbstractSemidiscretization, solution_callback,
                                    element_variables=Dict{Symbol,Any}())
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  u = wrap_array_native(u_ode, mesh, equations, solver, cache)
  save_solution_file(u, t, dt, iter, mesh, equations, solver, cache, solution_callback, element_variables)
end


# TODO: Taal refactor, move save_mesh_file?
# function save_mesh_file(mesh::TreeMesh, output_directory, timestep=-1) in io/io.jl

include("save_solution_dg.jl")


end # @muladd
