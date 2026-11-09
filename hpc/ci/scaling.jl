# Combines the PIDs of one scaling series in the results directory.
# benchmark/run.jl writes the PID of each count to `pid-<count>.txt`.
#
#   julia hpc/ci/scaling.jl <name>
#
# benchmark.json holds the PID of each count in seconds, smaller is better.
# efficiency.json holds the parallel efficiency to the smallest count, larger is better.
# Ideal scaling keeps the PID constant, so the efficiency is the ratio of the PIDs.

const results = ENV["RESULTS"]
name = only(ARGS)

matches = filter!(!isnothing, match.(r"^pid-(\d+)\.txt$", readdir(results)))
counts = sort!([parse(Int, m[1]) for m in matches])
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
