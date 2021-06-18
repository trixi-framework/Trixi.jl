include("containers.jl")
include("math.jl")


# Enable debug timings `@trixi_timeit timer() "name" stuff...`.
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

Semantically the same as `Threads.@threads` when iterating over a `AbstractUnitRange`
but without guarantee that the underlying implementation uses `Threads.@threads`
or works for more general `for` loops.
In particular, there may be an additional check whether only one thread is used
to reduce the overhead of serial execution or the underlying threading capabilities
might be provided by other packages such as [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl).

!!! warn
    This macro does not necessarily work for general `for` loops. For example,
    it does not necessarily support general iterables such as `eachline(filename)`.

Some discussion can be found at https://discourse.julialang.org/t/overhead-of-threads-threads/53964
and https://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435.
"""
macro threaded(expr)
  # Use `esc(quote ... end)` for nested macro calls as suggested in
  # https://github.com/JuliaLang/julia/issues/23221
  #
  # The following code is a simple version using only `Threads.@threads` from the
  # standard library with an additional check whether only a single thread is used
  # to reduce some overhead (and allocations) for serial execution.
  #
  # return esc(quote
  #   let
  #     if Threads.nthreads() == 1
  #       $(expr)
  #     else
  #       Threads.@threads $(expr)
  #     end
  #   end
  # end)
  #
  # However, the code below using `@batch` from Polyester.jl is more efficient,
  # since this packages provides threads with less overhead. Since it is written
  # by Chris Elrod, the author of LoopVectorization.jl, we expect this package
  # to provide the most efficient and useful implementation of threads (as we use
  # them) available in Julia.
  # !!! danger "Heisenbug"
  #     Look at the comments for `wrap_array` when considering to change this macro.

  return esc(quote @batch $(expr) end)
end


#     @trixi_timeit timer() "some label" expression
#
# Basically the same as a special case of `@timeit_debug` from
# [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl),
# but without `try ... finally ... end` block. Thus, it's not exception-safe,
# but it also avoids some related performance problems. Since we do not use
# exception handling in Trixi, that's not really an issue.
macro trixi_timeit(timer_output, label, expr)
  timeit_block = quote
    if timeit_debug_enabled()
      local to = $(esc(timer_output))
      local enabled = to.enabled
      if enabled
        local accumulated_data = $(TimerOutputs.push!)(to, $(esc(label)))
      end
      local b₀ = $(TimerOutputs.gc_bytes)()
      local t₀ = $(TimerOutputs.time_ns)()
    end
    local val = $(esc(expr))
    if timeit_debug_enabled() && enabled
      $(TimerOutputs.do_accumulate!)(accumulated_data, t₀, b₀)
      $(TimerOutputs.pop!)(to)
    end
    val
  end
end


"""
    init_p4est()

Initialize p4est by calling `p4est_init` and setting the log level to `SC_LP_ERROR`.
This function will check if p4est is already initialized
and if yes, do nothing, thus it is safe to call it multiple times.
"""
function init_p4est()
  if p4est_package_id()[] >= 0
    return nothing
  end

  # Initialize p4est with log level ERROR to prevent a lot of output in AMR simulations
  p4est_init(C_NULL, SC_LP_ERROR)

  return nothing
end
