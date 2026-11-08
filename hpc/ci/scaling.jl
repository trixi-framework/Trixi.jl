# Combines the PIDs of one scaling series, written by benchmark/run.jl as `pid-<count>.txt` into the results directory.
#
#   julia hpc/ci/scaling.jl <name>
#
# benchmark.json holds the PID of each count in seconds, smaller is better.
# efficiency.json holds the parallel efficiency relative to the smallest count, larger is better.
# Ideal scaling keeps the PID constant, so the efficiency is the PID ratio to the smallest count.

const results = ENV["RESULTS"]
name = only(ARGS)

counts = sort!([parse(Int, m[1])
                for m in filter!(!isnothing, match.(r"^pid-(\d+)\.txt$", readdir(results)))])
@assert !isempty(counts)
pids = [parse(Float64, read(joinpath(results, "pid-$count.txt"), String))
        for count in counts]
labels = ["$name=$count" for count in counts]

entry(label, unit, value) = """{"name": "$label", "unit": "$unit", "value": $value}"""
write(joinpath(results, "benchmark.json"),
      "[" * join(entry.(labels, "s", pids), ", ") * "]\n")
write(joinpath(results, "efficiency.json"),
      "[" * join(entry.(labels[2:end], "ratio", first(pids) ./ pids[2:end]), ", ") * "]\n")
foreach(println, entry.(labels, "s", pids))
