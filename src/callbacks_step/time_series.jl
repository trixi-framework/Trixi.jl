# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    TimeSeriesCallback(semi, point_coordinates;
                       interval=1, solution_variables=cons2cons,
                       output_directory="out", filename="time_series.h5",
                       RealT=real(solver), uEltype=eltype(cache.elements))

Create a callback that records point-wise data at points given in `point_coordinates` every
`interval` time steps. The point coordinates are to be specified either as a vector of coordinate
tuples or as a two-dimensional array where the first dimension is the point number and the second
dimension is the coordinate dimension. By default, the conservative variables are recorded, but this
can be controlled by passing a different conversion function to `solution_variables`.

After the last time step, the results are stored in an HDF5 file `filename` in directory
`output_directory`.

The real data type `RealT` and data type for solution variables `uEltype` default to the respective
types used in the solver and the cache.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
mutable struct TimeSeriesCallback{RealT<:Real, uEltype<:Real, SolutionVariables, VariableNames, Cache}
  interval::Int
  solution_variables::SolutionVariables
  variable_names::VariableNames
  output_directory::String
  filename::String
  point_coordinates::Array{RealT, 2}
  # Point data is stored as a vector of vectors of the solution data type:
  # * the "outer" `Vector` contains one vector for each point at which a time_series is recorded
  # * the "inner" `Vector` contains the actual time series for a single point,
  #   with each record  adding "n_vars" entries
  # The reason for using this data structure is that the length of the inner vectors needs to be
  # increased for each record, which can only be realized in Julia using ordinary `Vector`s.
  point_data::Vector{Vector{uEltype}}
  time::Vector{RealT}
  step::Vector{Int}
  time_series_cache::Cache
end


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:TimeSeriesCallback})
  @nospecialize cb # reduce precompilation time

  time_series_callback = cb.affect!
  @unpack interval, solution_variables, output_directory, filename = time_series_callback
  print(io, "TimeSeriesCallback(",
            "interval=", interval, ", ",
            "solution_variables=", interval, ", ",
            "output_directory=", "\"output_directory\"", ", ",
            "filename=", "\"filename\"",
            ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:TimeSeriesCallback})
  @nospecialize cb # reduce precompilation time

  if get(io, :compact, false)
    show(io, cb)
  else
    time_series_callback = cb.affect!

    setup = [
             "#points" => size(time_series_callback.point_coordinates, 2),
             "interval" => time_series_callback.interval,
             "solution_variables" => time_series_callback.solution_variables,
             "output_directory" => time_series_callback.output_directory,
             "filename" => time_series_callback.filename,
            ]
    summary_box(io, "TimeSeriesCallback", setup)
  end
end


# Main constructor
function TimeSeriesCallback(mesh, equations, solver, cache, point_coordinates;
                            interval::Integer=1,
                            solution_variables=cons2cons,
                            output_directory="out",
                            filename="time_series.h5",
                            RealT=real(solver),
                            uEltype=eltype(cache.elements))
  # check arguments
  if !(interval isa Integer && interval >= 0)
    throw(ArgumentError("`interval` must be a non-negative integer (provided `interval = $interval`)"))
  end

  if ndims(point_coordinates) != 2 || size(point_coordinates, 2) != ndims(mesh)
    throw(ArgumentError("`point_coordinates` must be a matrix of size n_points × ndims"))
  end

  # Transpose point_coordinates to our usual format [ndims, n_points]
  # Note: They are accepted in a different format to allow direct input from `readdlm`
  point_coordinates_ = permutedims(point_coordinates)

  # Invoke callback every `interval` time steps or after final step (for storing the data on disk)
  if interval > 0
    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    condition = (u, t, integrator) -> ( (integrator.stats.naccept % interval == 0 &&
                                        !(integrator.stats.naccept == 0 && integrator.iter > 0)) ||
                                      isfinished(integrator))
  else # disable the callback for interval == 0
    condition = (u, t, integrator) -> false
  end

  # Create data structures that are to be filled by the callback
  variable_names = varnames(solution_variables, equations)
  n_points = size(point_coordinates_, 2)
  point_data = Vector{uEltype}[Vector{uEltype}() for _ in 1:n_points]
  time = Vector{RealT}()
  step = Vector{Int}()
  time_series_cache = create_cache_time_series(point_coordinates_, mesh, solver, cache)

  time_series_callback = TimeSeriesCallback(interval,
                                           solution_variables,
                                           variable_names,
                                           output_directory,
                                           filename,
                                           point_coordinates_,
                                           point_data,
                                           time,
                                           step,
                                           time_series_cache)

  return DiscreteCallback(condition, time_series_callback, save_positions=(false,false))
end


# Convenience constructor that unpacks the semidiscretization into mesh, equations, solver, cache
function TimeSeriesCallback(semi, point_coordinates; kwargs...)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  return TimeSeriesCallback(mesh, equations, solver, cache, point_coordinates; kwargs...)
end


# Convenience constructor that converts a vector of points into a Trixi.jl-style coordinate array
function TimeSeriesCallback(mesh, equations, solver, cache, point_coordinates::AbstractVector;
                            kwargs...)
  # Coordinates are usually stored in [ndims, n_points], but here as [n_points, ndims]
  n_points = length(point_coordinates)
  point_coordinates_ = Matrix{eltype(eltype(point_coordinates))}(undef, n_points, ndims(mesh))

  for p in 1:n_points
    for d in 1:ndims(mesh)
      point_coordinates_[p, d] = point_coordinates[p][d]
    end
  end

  return TimeSeriesCallback(mesh, equations, solver, cache, point_coordinates_; kwargs...)
end


# This method is called as callback during the time integration.
function (time_series_callback::TimeSeriesCallback)(integrator)
  # Ensure this is not accidentally used with AMR enabled
  if uses_amr(integrator.opts.callback)
    error("the TimeSeriesCallback does not work with AMR enabled")
  end

  @unpack interval = time_series_callback

  # Create record if in correct interval (needs to be checked since the callback is also called
  # after the final step for storing the data on disk, independent of the current interval)
  if integrator.stats.naccept % interval == 0
    @trixi_timeit timer() "time series" begin
      # Store time and step
      push!(time_series_callback.time, integrator.t)
      push!(time_series_callback.step, integrator.stats.naccept)

      # Unpack data
      u_ode = integrator.u
      semi = integrator.p
      mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
      u = wrap_array(u_ode, mesh, equations, solver, cache)

      @unpack (point_data, solution_variables,
              variable_names, time_series_cache) = time_series_callback

      # Record state at points (solver/mesh-dependent implementation)
      record_state_at_points!(point_data, u, solution_variables, length(variable_names), mesh,
                              equations, solver, time_series_cache)
    end
  end

  # Store time_series if this is the last time step
  if isfinished(integrator)
    semi = integrator.p
    mesh, equations, solver, _ = mesh_equations_solver_cache(semi)
    save_time_series_file(time_series_callback, mesh, equations, solver)
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)

  return nothing
end


include("time_series_dg.jl")
include("time_series_dg2d.jl")


end # @muladd
