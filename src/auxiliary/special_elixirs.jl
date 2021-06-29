

# Note: We can't call the method below `Trixi.include` since that is created automatically
# inside `module Trixi` to `include` source files and evaluate them within the global scope
# of `Trixi`. However, users will want to evaluate in the global scope of `Main` or something
# similar to manage dependencies on their own.
"""
    trixi_include([mod::Module=Main,] elixir::AbstractString; kwargs...)

`include` the file `elixir` and evaluate its content in the global scope of module `mod`.
You can override specific assignments in `elixir` by supplying keyword arguments.
It's basic purpose is to make it easier to modify some parameters while running Trixi from the
REPL. Additionally, this is used in tests to reduce the computational burden for CI while still
providing examples with sensible default values for users.

# Examples

```jldoctest
julia> redirect_stdout(devnull) do
         trixi_include(@__MODULE__, joinpath(examples_dir(), "tree_1d_dgsem", "elixir_advection_extended.jl"),
                       tspan=(0.0, 0.1))
         sol.t[end]
       end
0.1
```
"""
function trixi_include(mod::Module, elixir::AbstractString; kwargs...)
  Base.include(ex -> replace_assignments(ex; kwargs...), mod, elixir)
end

trixi_include(elixir::AbstractString; kwargs...) = trixi_include(Main, elixir; kwargs...)


"""
    convergence_test([mod::Module=Main,] elixir::AbstractString, iterations; kwargs...)

Run `iterations` Trixi simulations using the setup given in `elixir` and compute
the experimental order of convergence (EOC) in the ``L^2`` and ``L^\\infty`` norm.
In each iteration, the resolution of the respective mesh will be doubled.
Additional keyword arguments `kwargs...` and the optional module `mod` are passed directly
to [`trixi_include`](@ref).
"""
function convergence_test(mod::Module, elixir::AbstractString, iterations; kwargs...)
  @assert(iterations > 1, "Number of iterations must be bigger than 1 for a convergence analysis")

  # Types of errors to be calcuated
  errors = Dict(:l2 => Float64[], :linf => Float64[])

  initial_resolution = extract_initial_resolution(elixir, kwargs)

  # run simulations and extract errors
  for iter in 1:iterations
    println("Running convtest iteration ", iter, "/", iterations)

    include_refined(mod, elixir, initial_resolution, iter; kwargs)

    l2_error, linf_error = mod.analysis_callback(mod.sol)

    # collect errors as one vector to reshape later
    append!(errors[:l2],   l2_error)
    append!(errors[:linf], linf_error)

    println("\n\n")
    println("#"^100)
  end

  # number of variables
  _, equations, _, _ = mesh_equations_solver_cache(mod.semi)
  variablenames = varnames(cons2cons, equations)
  nvariables = length(variablenames)

  # Reshape errors to get a matrix where the i-th row represents the i-th iteration
  # and the j-th column represents the j-th variable
  errorsmatrix = Dict(kind => transpose(reshape(error, (nvariables, iterations))) for (kind, error) in errors)

  # Calculate EOCs where the columns represent the variables
  # As dx halves in every iteration the denominator needs to be log(1/2)
  eocs = Dict(kind => log.(error[2:end, :] ./ error[1:end-1, :]) ./ log(1 / 2) for (kind, error) in errorsmatrix)

  eoc_mean_values = Dict{Symbol,Any}()
  eoc_mean_values[:variables] = variablenames

  for (kind, error) in errorsmatrix
    println(kind)

    for v in variablenames
      @printf("%-20s", v)
    end
    println("")

    for k = 1:nvariables
      @printf("%-10s", "error")
      @printf("%-10s", "EOC")
    end
    println("")

    # Print errors for the first iteration
    for k = 1:nvariables
      @printf("%-10.2e", error[1, k])
      @printf("%-10s", "-")
    end
    println("")

    # For the following iterations print errors and EOCs
    for j = 2:iterations
      for k = 1:nvariables
        @printf("%-10.2e", error[j, k])
        @printf("%-10.2f", eocs[kind][j-1, k])
      end
      println("")
    end
    println("")

    # Print mean EOCs
    mean_values = zeros(nvariables)
    for v in 1:nvariables
      mean_values[v] = sum(eocs[kind][:, v]) ./ length(eocs[kind][:, v])
      @printf("%-10s", "mean")
      @printf("%-10.2f", mean_values[v])
    end
    eoc_mean_values[kind] = mean_values
    println("")
    println("-"^100)
  end

  return eoc_mean_values
end

convergence_test(elixir::AbstractString, iterations; kwargs...) = convergence_test(Main, elixir::AbstractString, iterations; kwargs...)



# Helper methods used in the functions defined above

# Apply the function `f` to `expr` and all sub-expressions recursively.
walkexpr(f, expr::Expr) = f(Expr(expr.head, (walkexpr(f, arg) for arg in expr.args)...))
walkexpr(f, x) = f(x)

# Replace assignments to `key` in `expr` by `key = val` for all `(key,val)` in `kwargs`.
function replace_assignments(expr; kwargs...)
  # replace explicit and keyword assignemnts
  expr = walkexpr(expr) do x
    if x isa Expr
      for (key,val) in kwargs
        if (x.head === Symbol("=") || x.head === :kw) && x.args[1] === Symbol(key)
          x.args[2] = :( $val )
          # dump(x)
        end
      end
    end
    return x
  end

  return expr
end

# find a (keyword or common) assignment to `destination` in `expr`
# and return the assigned value
function find_assignment(expr, destination)
  # declare result to be able to assign to it in the closure
  local result

  # find explicit and keyword assignemnts
  walkexpr(expr) do x
    if x isa Expr
      if (x.head === Symbol("=") || x.head === :kw) && x.args[1] === Symbol(destination)
        result = x.args[2]
        # dump(x)
      end
    end
    return x
  end

  result
end

# searches the parameter that specifies the mesh reslution in the elixir
function extract_initial_resolution(elixir, kwargs)
  code = read(elixir, String)
  expr = Meta.parse("begin $code end")

  try
    # get the initial_refinement_level from the elixir
    initial_refinement_level = find_assignment(expr, :initial_refinement_level)

    if haskey(kwargs, :initial_refinement_level)
      return kwargs[:initial_refinement_level]
    else
      return initial_refinement_level
    end
  catch e
    if isa(e, UndefVarError)
      # get cells_per_dimension from the elixir
      cells_per_dimension = eval(find_assignment(expr, :cells_per_dimension))

      if haskey(kwargs, :cells_per_dimension)
        return kwargs[:cells_per_dimension]
      else
        return cells_per_dimension
      end
    else
      throw(e)
    end
  end
end

# runs the specified elixir with a doubled resolution each time iter is increased by 1
# works for TreeMesh
function include_refined(mod, elixir, initial_refinement_level::Int, iter; kwargs)
  trixi_include(mod, elixir; kwargs..., initial_refinement_level=initial_refinement_level+iter-1)
end

# runs the specified elixir with a doubled resolution each time iter is increased by 1
# works for StructuredMesh
function include_refined(mod, elixir, cells_per_dimension::NTuple{NDIMS, Int}, iter; kwargs) where {NDIMS}
  new_cells_per_dimension = cells_per_dimension .* 2^(iter - 1)

  trixi_include(mod, elixir; kwargs..., cells_per_dimension=new_cells_per_dimension)
end
