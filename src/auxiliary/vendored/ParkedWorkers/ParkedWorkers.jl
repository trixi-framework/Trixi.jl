# Vendored from ParkedWorkers.jl by Valentin Churavy
# (https://github.com/vchuravy/ParkedWorkers.jl, not yet registered),
# commit a08912a6cc2f35a18c1bdf87cd5c2b50a846b9e6,
# file `src/ParkedWorkers.jl`, formatted with the JuliaFormatter settings of Trixi.jl and otherwise
# unchanged. Do not edit here; change the package and copy it again. Replace this copy by
# a package dependency once ParkedWorkers.jl is registered. See `LICENSE` in this
# directory.
#
# MIT License. Copyright (c) 2026 Valentin Churavy.

"""
    ParkedWorkers

A small pool of persistent ("parked") worker tasks for low-overhead parallel `for` loops
on top of Base threads.

There is one sticky worker task per thread of the `:default` thread pool. A parallel
loop splits its range into one contiguous chunk per thread, runs chunk 1 on the calling
task and hands the other chunks to workers. After finishing a chunk, a worker spins for a
short while (see [`set_spin_time!`](@ref)) before it goes to sleep, so that back-to-back
loops do not pay for task creation, scheduling and wake-up.

Public API: [`@pfor`](@ref), [`pforeach`](@ref), [`pmapreduce`](@ref), [`preduce`](@ref),
[`ParkedWorkers.nworkers`](@ref), [`ParkedWorkers.set_spin_time!`](@ref),
[`ParkedWorkers.spin_time`](@ref), [`ParkedWorkers.shutdown!`](@ref),
[`ParkedWorkers.neutral`](@ref).
"""
module ParkedWorkers

export @pfor, pforeach, pmapreduce, preduce

include("pool.jl")
include("loops.jl")

end # module
