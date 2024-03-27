# Disable formatting this file since it contains highly unusual formatting for better
# readability
#! format: off

using BenchmarkTools
using Pkg
Pkg.activate(@__DIR__)

const SUITE = BenchmarkGroup()

elixir = joinpath(@__DIR__, "elixir_3d_euler_source_terms_tree.jl")

benchname = basename(elixir) * "_threads"
SUITE[benchname] = BenchmarkGroup()

command(julia, elixir, threads) = `$julia --project=$(@__DIR__)
                                          --threads $threads
                                          benchmark_scaling_threaded.jl $elixir`

for thread_exp in 0:2
    threads = 2^thread_exp
    run(command(Base.julia_cmd(), elixir, threads))

    # load benchmark results
    SUITE[benchname]["t$threads"] =
        BenchmarkTools.load("$(basename(elixir))_t$threads.json")

    # pretty print
    show(stdout, MIME"text/plain"(), SUITE[benchname]["t$threads"][1])
    println()
end
