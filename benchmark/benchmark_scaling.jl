# Disable formatting this file since it contains highly unusual formatting for better
# readability
#! format: off

using BenchmarkTools
using Printf
using Pkg
Pkg.activate(@__DIR__)

const SUITE = BenchmarkGroup()

elixir = joinpath(@__DIR__, "elixir_3d_euler_source_terms_tree.jl")

benchname = basename(elixir) * "_threads"
SUITE[benchname] = BenchmarkGroup()

command(julia, elixir, threads) = `$julia --project=$(@__DIR__)
                                          --threads $threads
                                          benchmark_scaling_threaded.jl $elixir`

for thread_exp in 0:1
    threads = 2^thread_exp
    run(command(Base.julia_cmd(), elixir, threads))

    # load benchmark results
    SUITE[benchname]["t$threads"] =
        BenchmarkTools.load("$(basename(elixir))_t$threads.json")

    # pretty print
    show(stdout, MIME"text/plain"(), SUITE[benchname]["t$threads"][1])
    println()
end

println()
@info "Summary"

trial_keys = collect(keys(SUITE[benchname]))
sort!(trial_keys)

println("#Threads | Median time")
for key in trial_keys
    median_time = median(SUITE[benchname][key][1]).time
    @printf("%8s | %s\n", key, BenchmarkTools.prettytime(median_time))
end
