# The following statements below outside the `@muladd begin ... end` block, as otherwise
# Revise.jl might be broken

include("containers.jl")
include("math.jl")

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    PerformanceCounter()

A `PerformanceCounter` can be used to track the runtime performance of some calls.
Add a new runtime measurement via `put!(counter, runtime)` and get the averaged
runtime of all measurements added so far via `take!(counter)`, resetting the
`counter`.
"""
mutable struct PerformanceCounter
    ncalls_since_readout::Int
    runtime::Float64
end

PerformanceCounter() = PerformanceCounter(0, 0.0)

@inline function Base.take!(counter::PerformanceCounter)
    time_per_call = counter.runtime / counter.ncalls_since_readout
    counter.ncalls_since_readout = 0
    counter.runtime = 0.0
    return time_per_call
end

@inline function Base.put!(counter::PerformanceCounter, runtime::Real)
    counter.ncalls_since_readout += 1
    counter.runtime += runtime
end

@inline ncalls(counter::PerformanceCounter) = counter.ncalls_since_readout

"""
    PerformanceCounterList{N}()

A `PerformanceCounterList{N}` can be used to track the runtime performance of
calls to multiple functions, adding them up.
Add a new runtime measurement via `put!(counter.counters[i], runtime)` and get
the averaged runtime of all measurements added so far via `take!(counter)`,
resetting the `counter`.
"""
struct PerformanceCounterList{N}
    counters::NTuple{N, PerformanceCounter}
    check_ncalls_consistency::Bool
end

function PerformanceCounterList{N}(check_ncalls_consistency) where {N}
    counters = ntuple(_ -> PerformanceCounter(), Val{N}())
    return PerformanceCounterList{N}(counters, check_ncalls_consistency)
end
PerformanceCounterList{N}() where {N} = PerformanceCounterList{N}(true)

@inline function Base.take!(counter_list::PerformanceCounterList)
    time_per_call = 0.0
    for c in counter_list.counters
        time_per_call += take!(c)
    end
    return time_per_call
end

@inline function ncalls(counter_list::PerformanceCounterList)
    ncalls_first = ncalls(first(counter_list.counters))

    if counter_list.check_ncalls_consistency
        for c in counter_list.counters
            if ncalls_first != ncalls(c)
                error("Some counters have a different number of calls. Using `ncalls` on the counter list is undefined behavior.")
            end
        end
    end

    return ncalls_first
end

"""
    examples_dir()

Return the directory where the example files provided with Trixi.jl are located. If Trixi.jl is
installed as a regular package (with `]add Trixi`), these files are read-only and should *not* be
modified. To find out which files are available, use, e.g., `readdir`:

# Examples
```@example
readdir(examples_dir())
```
"""
examples_dir() = pkgdir(Trixi, "examples")

"""
    get_examples()

Return a list of all example elixirs that are provided by Trixi.jl. See also
[`examples_dir`](@ref) and [`default_example`](@ref).
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

Return the path to an example elixir that can be used to quickly see Trixi.jl in action on a
[`TreeMesh`](@ref). See also [`examples_dir`](@ref) and [`get_examples`](@ref).
"""
function default_example()
    joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl")
end

"""
    default_example_unstructured()

Return the path to an example elixir that can be used to quickly see Trixi.jl in action on an
[`UnstructuredMesh2D`](@ref). This simulation is run on the example curved, unstructured mesh
given in the Trixi.jl documentation regarding unstructured meshes.
"""
function default_example_unstructured()
    joinpath(examples_dir(), "unstructured_2d_dgsem", "elixir_euler_basic.jl")
end

"""
    ode_default_options()

Return the default options for OrdinaryDiffEq's `solve`. Pass `ode_default_options()...` to `solve`
to only return the solution at the final time and enable **MPI aware** error-based step size control,
whenever MPI is used.
For example, use `solve(ode, alg; ode_default_options()...)`.
"""
function ode_default_options()
    if mpi_isparallel()
        return (; save_everystep = false, internalnorm = ode_norm,
                unstable_check = ode_unstable_check)
    else
        return (; save_everystep = false)
    end
end

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
get_name(::Val{x}) where {x} = string(x)

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

Some discussion can be found at [https://discourse.julialang.org/t/overhead-of-threads-threads/53964](https://discourse.julialang.org/t/overhead-of-threads-threads/53964)
and [https://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435](https://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435).
"""
macro threaded(expr)
    # !!! danger "Heisenbug"
    #     Look at the comments for `wrap_array` when considering to change this macro.
    expr = if _PREFERENCE_POLYESTER
        # Currently using `@batch` from Polyester.jl is more efficient,
        # bypasses the Julia task scheduler and provides parallelization with less overhead.
        quote
            $Trixi.@batch $(expr)
        end
    else
        # The following code is a simple version using only `Threads.@threads` from the
        # standard library with an additional check whether only a single thread is used
        # to reduce some overhead (and allocations) for serial execution.
        quote
            let
                if $Threads.nthreads() == 1
                    $(expr)
                else
                    $Threads.@threads :static $(expr)
                end
            end
        end
    end
    # Use `esc(quote ... end)` for nested macro calls as suggested in
    # https://github.com/JuliaLang/julia/issues/23221
    return esc(expr)
end

"""
    @autoinfiltrate
    @autoinfiltrate condition::Bool

Invoke the `@infiltrate` macro of the package Infiltrator.jl to create a breakpoint for ad-hoc
interactive debugging in the REPL. If the optional argument `condition` is given, the breakpoint is
only enabled if `condition` evaluates to `true`.

As opposed to using `Infiltrator.@infiltrate` directly, this macro does not require Infiltrator.jl
to be added as a dependency to Trixi.jl. As a bonus, the macro will also attempt to load the
Infiltrator module if it has not yet been loaded manually.

Note: For this macro to work, the Infiltrator.jl package needs to be installed in your current Julia
environment stack.

See also: [Infiltrator.jl](https://github.com/JuliaDebug/Infiltrator.jl)

!!! warning "Internal use only"
    Please note that this macro is intended for internal use only. It is *not* part of the public
    API of Trixi.jl, and it thus can altered (or be removed) at any time without it being considered
    a breaking change.
"""
macro autoinfiltrate(condition = true)
    pkgid = Base.PkgId(Base.UUID("5903a43b-9cc3-4c30-8d17-598619ec4e9b"), "Infiltrator")
    if !haskey(Base.loaded_modules, pkgid)
        try
            Base.eval(Main, :(using Infiltrator))
        catch err
            @error "Cannot load Infiltrator.jl. Make sure it is included in your environment stack."
        end
    end
    i = get(Base.loaded_modules, pkgid, nothing)
    lnn = LineNumberNode(__source__.line, __source__.file)

    if i === nothing
        return Expr(:macrocall,
                    Symbol("@warn"),
                    lnn,
                    "Could not load Infiltrator.")
    end

    return Expr(:macrocall,
                Expr(:., i, QuoteNode(Symbol("@infiltrate"))),
                lnn,
                esc(condition))
end

# Use the *experimental* feature in `Base` to add error hints for specific errors. We use it to
# warn users in case they try to execute functions that are extended in package extensions which
# have not yet been loaded.
#
# Reference: https://docs.julialang.org/en/v1/base/base/#Base.Experimental.register_error_hint
function register_error_hints()
    # We follow the advice in the docs and gracefully exit without doing anything if the experimental
    # features gets silently removed.
    if !isdefined(Base.Experimental, :register_error_hint)
        return nothing
    end

    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f in [iplot, iplot!] && isempty(methods(exc.f))
            print(io,
                  "\n$(exc.f) has no methods yet. It is part of a plotting extension of Trixi.jl " *
                  "that relies on Makie being loaded.\n" *
                  "To activate the extension, execute `using Makie`, `using CairoMakie`, " *
                  "`using GLMakie`, or load any other package that also uses Makie.")
        end
    end

    return nothing
end

"""
    Trixi.download(src_url, file_path)

Download a file from given `src_url` to given `file_path` if
`file_path` is not already a file. This function just returns
`file_path`.
This is a small wrapper of `Downloads.download(src_url, file_path)`
that avoids race conditions when multiple MPI ranks are used.
"""
function download(src_url, file_path)
    # Note that `mpi_isroot()` is also `true` if running
    # in serial (without MPI).
    if mpi_isroot()
        isfile(file_path) || Downloads.download(src_url, file_path)
    end

    if mpi_isparallel()
        MPI.Barrier(mpi_comm())
    end

    return file_path
end
end # @muladd
