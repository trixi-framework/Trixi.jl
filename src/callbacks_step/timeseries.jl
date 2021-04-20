
"""
    TimeseriesCallback(semi, point_coordinates;
                       interval=1, solution_variables=cons2cons,
                       output_directory="out", filename="timeseries.h5")

"""
mutable struct TimeseriesCallback{RealT<:Real, uEltype<:Real, SolutionVariables, VariableNames, Cache}
  interval::Int
  solution_variables::SolutionVariables
  variable_names::VariableNames
  output_directory::String
  filename::String
  point_coordinates::Matrix{RealT}
  point_data::Vector{Vector{uEltype}}
  time::Vector{RealT}
  step::Vector{Int}
  timeseries_cache::Cache
end


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:TimeseriesCallback})
  @nospecialize cb # reduce precompilation time

  timeseries_callback = cb.affect!
  @unpack interval = timeseries_callback
  print(io, "TimeseriesCallback(interval=", interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:TimeseriesCallback})
  @nospecialize cb # reduce precompilation time

  if get(io, :compact, false)
    show(io, cb)
  else
    timeseries_callback = cb.affect!

    setup = [
             "interval" => timeseries_callback.interval,
            ]
    summary_box(io, "TimeseriesCallback", setup)
  end
end


function TimeseriesCallback(semi, point_coordinates; kwargs...)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  return TimeseriesCallback(mesh, equations, solver, cache, point_coordinates; kwargs...)
end

function TimeseriesCallback(mesh, equations, solver, cache, point_coordinates::AbstractVector;
                            kwargs...)
  n_points = length(point_coordinates)
  point_coordinates_ = Matrix{eltype(eltype(point_coordinates))}(undef, ndims(mesh), n_points)

  for p in 1:n_points
    for d in 1:ndims(mesh)
      point_coordinates_[d, p] = point_coordinates[p][d]
    end
  end

  return TimeseriesCallback(mesh, equations, solver, cache, point_coordinates_; kwargs...)
end

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

  # Save point data every `interval` time steps
  if interval > 0
    condition = (u, t, integrator) -> (integrator.iter % interval == 0 || isfinished(integrator))
  else # disable the callback for interval == 0
    condition = (u, t, integrator) -> false
  end

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


# This method is called as callback during the time integration.
@inline function (timeseries_callback::TimeseriesCallback)(integrator)
  @unpack iter = integrator
  @unpack interval = timeseries_callback

  # Create record if in correct interval
  if integrator.iter % interval == 0
    @timeit_debug timer() "time series" begin
      # Store time and step
      push!(timeseries_callback.time, integrator.t)
      push!(timeseries_callback.step, iter)

      u_ode = integrator.u
      semi = integrator.p
      mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
      u = wrap_array(u_ode, mesh, equations, solver, cache)

      @unpack (point_data, solution_variables,
              variable_names, timeseries_cache) = timeseries_callback

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
      attributes(file)["ndims"] = ndims(mesh)
      attributes(file)["equations"] = get_name(equations)
      attributes(file)["polydeg"] = polydeg(solver)
      attributes(file)["n_vars"] = length(variable_names)
      attributes(file)["n_points"] = n_points
      attributes(file)["interval"] = interval
      attributes(file)["variable_names"] = collect(variable_names)

      file["time"] = time
      file["timestep"] = step
      file["point_coordinates"] = point_coordinates
      for p in 1:n_points
        file["point_data_$p"] = point_data[p]
      end
    end
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)

  return nothing
end


include("timeseries_dg2d.jl")
