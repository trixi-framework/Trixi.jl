# Vendored from ParkedWorkers.jl by Valentin Churavy
# (https://github.com/vchuravy/ParkedWorkers.jl, not yet registered),
# commit a08912a6cc2f35a18c1bdf87cd5c2b50a846b9e6,
# file `src/pool.jl`, formatted with the JuliaFormatter settings of Trixi.jl and otherwise
# unchanged. Do not edit here; change the package and copy it again. Replace this copy by
# a package dependency once ParkedWorkers.jl is registered. See `LICENSE` in this
# directory.
#
# MIT License. Copyright (c) 2026 Valentin Churavy.

# The worker pool.
#
# There is one sticky worker task per thread of the `:default` thread pool. Worker `k`
# runs on thread `tids[k]` and waits for commands in its own `WorkerState` (padded to
# avoid false sharing). A command is the index of the chunk to run, `0` means idle,
# `RUNNING` that the worker claimed its chunk and `STOP` asks the worker to exit.
#
# A parallel region is started by `dispatch!`:
#
# 1. The chunk function `f::F` is stored in a per-type `Base.RefValue{F}` slot owned by
#    the pool. Workers receive the slot, which is already heap-allocated, so the hand-off
#    does not allocate (no boxing of `f`).
# 2. Workers get chunks `2:n` by storing the chunk index into their command field. The
#    worker running on the caller's thread (if any) is skipped, because the caller
#    occupies that thread. Sleeping workers are woken up with their `Event`.
# 3. The caller runs chunk 1 itself. Then it "steals" every chunk that its worker has not
#    claimed yet (a compare-and-swap of the command from `chunk` to `0`; the worker claims
#    it with a swap to `RUNNING`) and runs it itself. This keeps the region from waiting
#    for a worker that is asleep or whose thread the OS does not schedule (oversubscribed
#    cores). Finally it waits until every worker it used is idle again, also when a chunk
#    threw.
#
# Only one parallel region can be active at a time. The `busy` flag is taken with a
# compare-and-swap; a caller that does not get it (a nested loop, or a loop started
# concurrently by another task) runs its loop serially instead of waiting.

const STOP = -1
const RUNNING = -2

mutable struct WorkerState
    @atomic cmd::Int
    @atomic sleeping::Bool
    # Padding: keep the fields of different workers on different cache lines
    _pad::NTuple{14, Int}
end
WorkerState() = WorkerState(0, false, ntuple(_ -> 0, Val(14)))

mutable struct Pool
    # `Base.RefValue{F}` holding the chunk function of the current parallel region
    @atomic job::Any
    const workers::Vector{WorkerState}
    const events::Vector{Base.Event}
    const tasks::Vector{Task}
    # `errs[k]`: `CapturedException` thrown by the chunk worker `k` ran, or `nothing`
    const errs::Vector{Any}
    # `tid2worker[tid]`: index of the worker running on thread `tid`, or `0`
    const tid2worker::Vector{Int}
    # One `Base.RefValue{F}` per chunk function type `F`, and one partials buffer per
    # result type of a reduction. Only accessed while holding `POOL.busy`.
    const slots::IdDict{Type, Any}
    const buffers::IdDict{Type, Any}
end

# Global state. `busy` is held by the task running a parallel region (and by `shutdown!`);
# `pool` is only created, replaced or used while holding `busy`.
mutable struct PoolHolder
    @atomic busy::Bool
    # Task that holds `busy` (to detect `shutdown!` calls from inside a region)
    @atomic owner::Any
    @atomic pool::Union{Nothing, Pool}
end
const POOL = PoolHolder(false, nothing, nothing)

# Spin time of an idle worker before it goes to sleep, in nanoseconds
const SPIN_NS = Threads.Atomic{Int}(100_000)

"""
    ParkedWorkers.set_spin_time!(seconds::Real)

Set how long an idle worker busy-waits for the next parallel loop before it goes to
sleep. The default is `100e-6` (100 µs). Takes effect the next time a worker starts
waiting.

Longer spinning makes parallel loops that follow each other with short serial gaps cheaper
(waking up a sleeping worker costs about 10 µs), but idle workers keep their CPU busy for
that long after every loop. `0` makes workers go to sleep right away (lowest CPU use,
highest latency).
"""
function set_spin_time!(seconds::Real)
    seconds >= 0 || throw(ArgumentError("spin time must be non-negative"))
    SPIN_NS[] = round(Int, seconds * 1e9)
    return seconds
end

"""
    ParkedWorkers.spin_time() -> Float64

Current spin time of idle workers in seconds, see [`set_spin_time!`](@ref).
"""
spin_time() = SPIN_NS[] / 1e9

"""
    ParkedWorkers.nworkers() -> Int

Maximum number of chunks a parallel loop is split into: the number of threads of the
`:default` thread pool (`Threads.nthreads(:default)`).
"""
nworkers() = Threads.threadpoolsize(:default)

@inline cpu_pause() = ccall(:jl_cpu_pause, Cvoid, ())

@inline generating_output() = ccall(:jl_generating_output, Cint, ()) != 0

# Called by the workers through `invokelatest`, so that chunk functions whose types were
# defined after the workers were started can be called.
function run_job(slot::Base.RefValue{F}, chunk::Int) where {F}
    slot[](chunk)
    return nothing
end

# Wait until `ws.cmd` is non-zero: spin for `SPIN_NS` nanoseconds, yielding to other tasks
# on this thread now and then, then sleep on `ev`.
function wait_for_command(ws::WorkerState, ev::Base.Event)
    spins = 0
    t0 = time_ns()
    while true
        cmd = @atomic :acquire ws.cmd
        if cmd > 0
            # Claim the chunk; fails if the dispatcher stole it in the meantime
            (@atomicreplace ws.cmd cmd=>RUNNING).success && return cmd
        elseif cmd == STOP
            return cmd
        end
        spins += 1
        if spins & 255 == 0
            if time_ns() - t0 > SPIN_NS[]
                @atomic ws.sleeping = true
                # Re-check after announcing that we sleep: either we see the command
                # here, or the dispatcher sees `sleeping` and notifies `ev`.
                if (@atomic ws.cmd) <= 0
                    wait(ev)
                end
                @atomic ws.sleeping = false
                spins = 0
                t0 = time_ns()
                continue
            elseif spins & 1023 == 0
                yield()
            end
        end
        cpu_pause()
        GC.safepoint()
    end
end

function worker_loop(pool::Pool, k::Int)
    ws = pool.workers[k]
    ev = pool.events[k]
    while true
        cmd = wait_for_command(ws, ev)
        cmd == STOP && break
        try
            Base.invokelatest(run_job, @atomic(:acquire, pool.job), cmd)
        catch err
            pool.errs[k] = CapturedException(err, catch_backtrace())
        end
        @atomic :release ws.cmd=0
    end
    @atomic :release ws.cmd=0
    return nothing
end

function Pool()
    tids = Threads.threadpooltids(:default)
    nw = length(tids)
    tid2worker = zeros(Int, maximum(tids))
    for (k, tid) in enumerate(tids)
        tid2worker[tid] = k
    end
    pool = Pool(nothing, [WorkerState() for _ in 1:nw],
                [Base.Event(true) for _ in 1:nw], Task[], Any[nothing for _ in 1:nw],
                tid2worker, IdDict{Type, Any}(), IdDict{Type, Any}())
    for (k, tid) in enumerate(tids)
        task = Task(() -> worker_loop(pool, k))
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, tid - 1)
        push!(pool.tasks, task)
        schedule(task)
    end
    return pool
end

# Is a parallel region active (in this task or any other)?
#
# This check is inlined into every parallel loop in front of the serial fallback loop, so
# it must only use relaxed (`:monotonic`) loads. LLVM's MemoryDependenceAnalysis treats any
# acquire (or stronger) load as a clobber of all memory, even for `!invariant.load` queries
# ("FIXME: This is overly conservative" in MemoryDependenceAnalysis.cpp). An acquire load
# here stops GVN from merging loads after it with loads before it (e.g., array sizes
# checked by the caller), so the serial loop would lose constant strides and other facts
# (Trixi.jl: +4 to +8% time in several kernels on Julia 1.13). A stale value is harmless:
# it only makes us try the pool (and fall back to serial in `acquire_pool`) or run one loop
# serially. Data is only published with acquire/release in the non-inlined `run_chunks`,
# `run_fold` and `dispatch!`.
@inline pool_busy() = @atomic :monotonic POOL.busy

# Take the pool for a parallel region (starting it if needed); `nothing` if the loop has to
# run serially. Only called from non-inlined functions.
@inline function acquire_pool()
    generating_output() && return nothing
    (@atomicreplace POOL.busy false=>true).success || return nothing
    @atomic :monotonic POOL.owner=current_task()
    pool = @atomic :acquire POOL.pool
    if pool === nothing
        pool = try
            new_pool()
        catch
            release_pool()
            rethrow()
        end
    end
    return pool
end

@noinline function new_pool()
    pool = Pool()
    @atomic :release POOL.pool=pool
    return pool
end

@inline function release_pool()
    @atomic :monotonic POOL.owner=nothing
    @atomic :release POOL.busy=false
    return nothing
end

@noinline function job_slot(pool::Pool, ::Type{F}) where {F}
    return get!(() -> Base.RefValue{F}(), pool.slots, F)::Base.RefValue{F}
end

@noinline function partials_buffer(pool::Pool, ::Type{T}, n::Int) where {T}
    buf = get!(() -> Vector{T}(undef, n), pool.buffers, T)::Vector{T}
    length(buf) < n && resize!(buf, n)
    return buf
end

# Spin until worker `k` is idle. Yields now and then, so that a worker (or a chunk that
# yields) sharing the thread of the waiting task can make progress.
@inline function wait_idle(pool::Pool, k::Int)
    ws = @inbounds pool.workers[k]
    spins = 0
    while (@atomic :acquire ws.cmd) != 0
        spins += 1
        if spins & 1023 == 0
            # Defensive: wake the worker if it went to sleep with a pending command
            @atomic(ws.sleeping) && notify(@inbounds pool.events[k])
            yield()
        else
            cpu_pause()
            GC.safepoint()
        end
    end
    return nothing
end

# Index of the worker that runs chunk `chunk` (≥ 2) when worker `skip` is not used
@inline worker_for_chunk(chunk, skip) = (k = chunk - 1;
                                         (skip != 0 && k >= skip) ? k + 1 :
                                         k)

# Run `f(chunk)` for `chunk in 1:n` in parallel; the caller holds `POOL.busy`.
function dispatch!(pool::Pool, f::F, n::Int) where {F}
    slot = job_slot(pool, F)
    slot[] = f
    @atomic :release pool.job=slot
    tid = Threads.threadid()
    skip = tid <= length(pool.tid2worker) ? @inbounds(pool.tid2worker[tid]) : 0
    for chunk in 2:n
        k = worker_for_chunk(chunk, skip)
        # A worker can still be busy if a previous region was interrupted
        wait_idle(pool, k)
        @inbounds pool.errs[k] = nothing
        ws = @inbounds pool.workers[k]
        # Sequentially consistent store and load (Dekker-style with the worker, which
        # stores `sleeping` and then loads `cmd`), so that either the worker sees the
        # command before it sleeps, or we see that it sleeps and wake it up.
        @atomic ws.cmd = chunk
        if @atomic ws.sleeping
            notify(@inbounds pool.events[k])
        end
    end
    try
        f(1)
    catch
        # Cancel the chunks that were not started and wait for the others: never leave
        # workers running when returning from the region
        for chunk in 2:n
            k = worker_for_chunk(chunk, skip)
            @atomicreplace (@inbounds pool.workers[k]).cmd chunk=>0
        end
        wait_workers(pool, n, skip)
        rethrow()
    end
    # Run the chunks not claimed by their worker yet
    for chunk in 2:n
        k = worker_for_chunk(chunk, skip)
        if (@atomicreplace (@inbounds pool.workers[k]).cmd chunk=>0).success
            try
                f(chunk)
            catch err
                @inbounds pool.errs[k] = CapturedException(err, catch_backtrace())
            end
        end
    end
    wait_workers(pool, n, skip)
    for chunk in 2:n
        k = worker_for_chunk(chunk, skip)
        err = @inbounds pool.errs[k]
        if err !== nothing
            throw_worker_error(err::CapturedException)
        end
    end
    return nothing
end

@inline function wait_workers(pool::Pool, n, skip)
    for chunk in 2:n
        wait_idle(pool, worker_for_chunk(chunk, skip))
    end
    return nothing
end

# Rethrow the original exception of a worker chunk. The worker's backtrace is kept on the
# exception stack (`current_exceptions()`), so it is shown as "caused by: ...".
@noinline function throw_worker_error(err::CapturedException)
    try
        throw(err)
    catch
        throw(err.ex)
    end
end

# Run `f(chunk)` for `chunk in 1:n`, in parallel if the pool is free, else serially.
# `n` must not exceed `nworkers()`.
# Not inlined: the atomics in here must stay out of the caller (see `pool_busy`).
@noinline function run_chunks(f::F, n::Int) where {F}
    pool = acquire_pool()
    if pool === nothing
        for chunk in 1:n
            f(chunk)
        end
        return nothing
    end
    try
        dispatch!(pool, f, n)
    finally
        release_pool()
    end
    return nothing
end

struct StoreChunk{F, T}
    f::F
    buf::Vector{T}
end
@inline (s::StoreChunk)(chunk) = (@inbounds s.buf[chunk] = s.f(chunk); nothing)

# Compute `op(op(f(1), f(2)), ...)` over `1:n` with the `f(chunk)` evaluated in parallel
# (`n ≥ 2`). The partial results are combined in chunk order, so the result is
# deterministic for a given `n`.
# Not inlined: the atomics in here must stay out of the caller (see `pool_busy`).
@noinline function run_fold(f::F, op::Op, n::Int) where {F, Op}
    pool = acquire_pool()
    if pool === nothing
        acc = f(1)
        for chunk in 2:n
            acc = op(acc, f(chunk))
        end
        return acc
    end
    try
        T = Base.promote_op(f, Int)
        T === Union{} && (T = Any)
        buf = partials_buffer(pool, T, n)
        dispatch!(pool, StoreChunk{F, T}(f, buf), n)
        acc = @inbounds buf[1]
        for chunk in 2:n
            acc = op(acc, @inbounds buf[chunk])
        end
        return acc
    finally
        release_pool()
    end
end

"""
    ParkedWorkers.shutdown!()

Stop all worker tasks and drop the pool, including the references it keeps to the last
chunk function of each type (and thereby to the arrays captured by it). Waits for a
parallel region running in another task to finish first. The next parallel loop starts
a new pool.

Throws an error when called from inside a parallel loop.
"""
function shutdown!()
    while !(@atomicreplace POOL.busy false=>true).success
        # The region can not end while we wait in one of its chunks
        if @atomic(POOL.owner) === current_task() || in_worker()
            error("ParkedWorkers.shutdown! called from inside a parallel loop")
        end
        yield()
    end
    @atomic :monotonic POOL.owner=current_task()
    try
        pool = @atomic POOL.pool
        pool === nothing && return nothing
        @atomic POOL.pool = nothing
        for (k, ws) in enumerate(pool.workers)
            wait_idle(pool, k)
            @atomic ws.cmd = STOP
            notify(pool.events[k])
        end
        foreach(wait, pool.tasks)
        empty!(pool.slots)
        empty!(pool.buffers)
        @atomic pool.job = nothing
    finally
        release_pool()
    end
    return nothing
end

function in_worker()
    pool = @atomic POOL.pool
    return pool !== nothing && any(t -> t === current_task(), pool.tasks)
end
