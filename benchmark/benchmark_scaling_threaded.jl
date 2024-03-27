using BenchmarkTools
using Trixi

threads = Threads.nthreads()

println()
@info "Running benchmark $(ARGS[1]) with $threads threads."

@info "Warm up"
trixi_include(ARGS[1], tspan=(0.0, 1.0e-10))

@info "Benchmark"
trial = @benchmark begin
    Trixi.rhs!($(similar(sol.u[end])), $(copy(sol.u[end])), $(semi), $(first(tspan)))
end

t_median = BenchmarkTools.prettytime(median(trial).time)
@info "Done! Median time: $t_median"

BenchmarkTools.save("$(basename(ARGS[1]))_t$threads.json", trial)
