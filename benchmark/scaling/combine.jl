# Combines the PIDs of one scaling series, written by `run.jl` to `pid-<count>.txt`.
#
#   julia benchmark/scaling/combine.jl <name> <directory> <baseline>
#
# benchmark.json holds the PID of each count in seconds, smaller is better.
# efficiency.json holds the parallel efficiency of the counts above the baseline count,
# larger is better.
# Ideal scaling keeps the PID constant, so the efficiency is the ratio of the PIDs.

name, results, baseline = ARGS
baseline = parse(Int, baseline)

matches = filter!(!isnothing, match.(r"^pid-(\d+)\.txt$", readdir(results)))
counts = sort!([parse(Int, m[1]) for m in matches])
@assert !isempty(counts)
pids = [parse(Float64, read(joinpath(results, "pid-$count.txt"), String))
        for count in counts]
labels = ["$name=$count" for count in counts]
reference = pids[only(findall(==(baseline), counts))]
above = counts .> baseline

entry(label, unit, value) = """{"name": "$label", "unit": "$unit", "value": $value}"""
write(joinpath(results, "benchmark.json"),
      "[" * join(entry.(labels, "s", pids), ", ") * "]\n")
write(joinpath(results, "efficiency.json"),
      "[" * join(entry.(labels[above], "ratio", reference ./ pids[above]), ", ") * "]\n")
foreach(println, entry.(labels, "s", pids))
