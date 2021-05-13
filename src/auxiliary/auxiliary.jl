include("containers.jl")
include("math.jl")


# Enable debug timings `@timeit_debug timer() "name" stuff...`.
# This allows us to disable timings completely by executing
# `TimerOutputs.disable_debug_timings(Trixi)`
# and to enable them again by executing
# `TimerOutputs.enable_debug_timings(Trixi)`
timeit_debug_enabled() = true

# Store main timer for global timing of functions
const main_timer = TimerOutput()

# Always call timer() to hide implementation details
timer() = main_timer


"""
    PerformanceCounter

A `PerformanceCounter` be used to track the runtime performance of some calls.
Add a new runtime measurement via `put!(counter, runtime)` and get the averaged
runtime of all measurements added so far via `take!(counter)`, resetting the
`counter`.
"""
mutable struct PerformanceCounter
  ncalls_since_readout::Int
  runtime::Float64
end

PerformanceCounter() = PerformanceCounter(0, 0.0)

function Base.take!(counter::PerformanceCounter)
  time_per_call = counter.runtime / counter.ncalls_since_readout
  counter.ncalls_since_readout = 0
  counter.runtime = 0.0
  return time_per_call
end

function Base.put!(counter::PerformanceCounter, runtime::Real)
  counter.ncalls_since_readout += 1
  counter.runtime += runtime
end



"""
    examples_dir()

Return the directory where the example files provided with Trixi.jl are located. If Trixi is
installed as a regular package (with `]add Trixi`), these files are read-only and should *not* be
modified. To find out which files are available, use, e.g., `readdir`:

# Examples
```julia
julia> readdir(examples_dir())
5-element Array{String,1}:
 "1d"
 "2d"
 "3d"
 "README.md"
 "paper-self-gravitating-gas-dynamics"
```
"""
examples_dir() = joinpath(pathof(Trixi) |> dirname |> dirname, "examples")


"""
    get_examples()

Return a list of all example elixirs that are provided by Trixi.
"""
function get_examples()
  examples = String[]
  for (root, dirs, files) in walkdir(examples_dir())
    for f in files
      if startswith(f, "elixir_") && endswith(f, ".jl")
        push!(examples, joinpath(root, f))
      end
    end
  end

  return examples
end


"""
    default_example()

Return the path to an example elixir that can be used to quickly see Trixi in action on a
[`TreeMesh`]@(ref).
"""
default_example() = joinpath(examples_dir(), "2d", "elixir_advection_basic.jl")


"""
    default_example_unstructured()

Return the path to an example elixir that can be used to quickly see Trixi in action on an
[`UnstructuredQuadMesh`]@(ref). This simulation is run on the example curved, unstructured mesh
given in the Trixi documentation regarding unstructured meshes.
"""
default_example_unstructured() = joinpath(examples_dir(), "2d", "elixir_euler_unstructured_quad_basic.jl")

# Print informative message at startup
function print_startup_message()
  s = """

    ████████╗██████╗ ██╗██╗  ██╗██╗
    ╚══██╔══╝██╔══██╗██║╚██╗██╔╝██║
       ██║   ██████╔╝██║ ╚███╔╝ ██║
       ██║   ██╔══██╗██║ ██╔██╗ ██║
       ██║   ██║  ██║██║██╔╝ ██╗██║
       ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝
    """
  mpi_println(s)
end


"""
    get_name(x)

Returns a name of `x` ready for pretty printing.
By default, return `string(y)` if `x isa Val{y}` and return `string(x)` otherwise.

# Examples

```jldoctest
julia> Trixi.get_name("test")
"test"

julia> Trixi.get_name(Val(:test))
"test"
```
"""
get_name(x) = string(x)
get_name(::Val{x}) where x = string(x)



"""
    @threaded for ... end

Basically the same as `Threads.@threads` with an additional check whether only
one thread is used to reduce the overhead of serial execution.

Some discussion can be found at https://discourse.julialang.org/t/overhead-of-threads-threads/53964
"""
macro threaded(expr)
  # esc(quote ... end) as suggested in https://github.com/JuliaLang/julia/issues/23221
  return esc(quote
    let
      if Threads.nthreads() == 1
        $(expr)
      else
        Threads.@threads $(expr)
      end
    end
  end)
end
