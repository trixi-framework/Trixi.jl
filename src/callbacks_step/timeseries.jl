
"""
    TimeseriesCallback(semi, point_coordinates;
                       interval=1, solution_variables=cons2cons,
                       output_directory="out", filename="timeseries.h5",
                       RealT=real(solver), uEltype=eltype(cache.elements))

Create a callback that records point-wise data at points given in `point_coordinates` every
`interval` time steps. By default, the conservative variables are recorded, but this can be
controlled by passing a different conversion function to `solution_variables`.

After the last time step, the results are stored in an HDF5 file `filename` in directory
`output_directory`.

The real data type `RealT` and data type for solution variables `uEltype` default to the respective
types used in the solver and the cache.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
mutable struct TimeseriesCallback{RealT<:Real, uEltype<:Real, SolutionVariables, VariableNames, Cache}
  interval::Int
  solution_variables::SolutionVariables
  variable_names::VariableNames
  output_directory::String
  filename::String
  point_coordinates::Array{RealT, 2}
  # Point data is stored as a vector of vectors of the solution data type:
  # * the "outer" `Vector` contains one vector for each point at which a timeseries is recorded
  # * the "inner" `Vector` contains the actual time series for a single point,
  #   with each record  adding "n_vars" entries
  # The reason for using this data structure is that the length of the inner vectors needs to be
  # increased for each record, which can only be realized in Julia using ordinary `Vector`s.
  point_data::Vector{Vector{uEltype}}
  time::Vector{RealT}
  step::Vector{Int}
  timeseries_cache::Cache
end


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:TimeseriesCallback})
  @nospecialize cb # reduce precompilation time

  timeseries_callback = cb.affect!
  @unpack interval, solution_variables, output_directory, filename = timeseries_callback
  print(io, "TimeseriesCallback(",
            "interval=", interval, ", ",
            "solution_variables=", interval, ", ",
            "output_directory=", "\"output_directory\"", ", ",
            "filename=", "\"filename\"",
            ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:TimeseriesCallback})
  @nospecialize cb # reduce precompilation time

  if get(io, :compact, false)
    show(io, cb)
  else
    timeseries_callback = cb.affect!

    setup = [
             "#points" => size(timeseries_callback.point_coordinates, 2),
             "interval" => timeseries_callback.interval,
             "solution_variables" => timeseries_callback.solution_variables,
             "output_directory" => timeseries_callback.output_directory,
             "filename" => timeseries_callback.filename,
            ]
    summary_box(io, "TimeseriesCallback", setup)
  end
end


# Main constructor
function TimeseriesCallback(mesh, equations, solver, cache, point_coordinates;
                            interval::Integer=1,
                            solution_variables=cons2cons,
                            output_directory="out",
                            filename="timeseries.h5",
                            RealT=real(solver),
                            uEltype=eltype(cache.elements))

  # check arguments
  if !(interval isa Integer && interval >= 0)
    throw(ArgumentError("`interval` must be a non-negative integer (provided `interval = $interval`)"))
  end

  # Invoke callback every `interval` time steps or after final step (for storing the data on disk)
  if interval > 0
    condition = (u, t, integrator) -> (integrator.iter % interval == 0 || isfinished(integrator))
  else # disable the callback for interval == 0
    condition = (u, t, integrator) -> false
  end

  # Create data structures that are to be filled by the callback
  variable_names = varnames(solution_variables, equations)
  n_points = size(point_coordinates, 2)
  point_data = Vector{uEltype}[Vector{uEltype}() for _ in 1:n_points]
  time = Vector{RealT}()
  step = Vector{Int}()
  timeseries_cache = create_cache_timeseries(point_coordinates, mesh, solver, cache)

  timeseries_callback = TimeseriesCallback(interval,
                                           solution_variables,
                                           variable_names,
                                           output_directory,
                                           filename,
                                           point_coordinates,
                                           point_data,
                                           time,
                                           step,
                                           timeseries_cache)

  return DiscreteCallback(condition, timeseries_callback, save_positions=(false,false))
end


# Convenience constructor that unpacks the semidiscretization into mesh, equations, solver, cache
function TimeseriesCallback(semi, point_coordinates; kwargs...)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  return TimeseriesCallback(mesh, equations, solver, cache, point_coordinates; kwargs...)
end


# Convenience constructor that converts a vector of points into a Trixi-style coordinate array
function TimeseriesCallback(mesh, equations, solver, cache, point_coordinates::AbstractVector;
                            kwargs...)
  # Coordinates are always stored in [ndims, n_points]
  n_points = length(point_coordinates)
  point_coordinates_ = Matrix{eltype(eltype(point_coordinates))}(undef, ndims(mesh), n_points)

  for p in 1:n_points
    for d in 1:ndims(mesh)
      point_coordinates_[d, p] = point_coordinates[p][d]
    end
  end

  return TimeseriesCallback(mesh, equations, solver, cache, point_coordinates_; kwargs...)
end


# This method is called as callback during the time integration.
@inline function (timeseries_callback::TimeseriesCallback)(integrator)
  @unpack iter = integrator
  @unpack interval = timeseries_callback

  # Create record if in correct interval (needs to be checked since the callback is also called
  # after the final step for storing the data on disk, indepdendent of the current interval)
  if integrator.iter % interval == 0
    @timeit_debug timer() "time series" begin
      # Store time and step
      push!(timeseries_callback.time, integrator.t)
      push!(timeseries_callback.step, iter)

      # Unpack data
      u_ode = integrator.u
      semi = integrator.p
      mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
      u = wrap_array(u_ode, mesh, equations, solver, cache)

      @unpack (point_data, solution_variables,
              variable_names, timeseries_cache) = timeseries_callback

      # Record state at points (solver/mesh-dependent implementation)
      record_state_at_points!(point_data, u, solution_variables, length(variable_names), mesh,
                              equations, solver, timeseries_cache)
    end
  end

  # Store timeseries if this is the last time step
  if isfinished(integrator)
    @unpack (solution_variables, variable_names,
             output_directory, filename,
             point_coordinates, point_data,
             time, step, timeseries_cache) = timeseries_callback
    n_points = length(point_data)

    h5open(joinpath(output_directory, filename), "w") do file
      # Add context information as attributes
      n_variables = length(variable_names)
      attributes(file)["ndims"] = ndims(mesh)
      attributes(file)["equations"] = get_name(equations)
      attributes(file)["polydeg"] = polydeg(solver)
      attributes(file)["n_vars"] = n_variables
      attributes(file)["n_points"] = n_points
      attributes(file)["interval"] = interval
      attributes(file)["variable_names"] = collect(variable_names)

      file["time"] = time
      file["timestep"] = step
      file["point_coordinates"] = point_coordinates
      for p in 1:n_points
        # Store data as 2D array for convenience
        file["point_data_$p"] = reshape(point_data[p], n_variables, length(time))
      end
    end
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)

  return nothing
end


include("timeseries_dg2d.jl")
