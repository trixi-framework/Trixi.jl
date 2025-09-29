using Trixi
using CUDA
using TimerOutputs
using JSON

function main(elixir_path)

    # setup
    maxiters = 50
    initial_refinement_level = 3
    storage_type = CuArray
    real_type = Float64

    println("Warming up...")

    # start simulation with tiny final time to trigger compilation
    duration_compile = @elapsed begin
        trixi_include(elixir_path,
                      tspan = (0.0, 1e-14),
                      storage_type = storage_type,
                      real_type = real_type)
        trixi_include(elixir_path,
                      tspan = (0.0, 1e-14),
                      storage_type = storage_type,
                      real_type = Float32)
    end

    println("Finished warm-up in $duration_compile seconds\n")
    println("Starting simulation...")

    # start the real simulation
    duration_elixir = @elapsed trixi_include(elixir_path,
                                             maxiters = maxiters,
                                             initial_refinement_level = initial_refinement_level,
                                             storage_type = storage_type,
                                             real_type = real_type)

    # store metrics (on every rank!)
    metrics = Dict{String, Float64}("elapsed time" => duration_elixir)

    # read TimerOutputs timings
    timer = Trixi.timer()
    metrics["total time"] = 1.0e-9 * TimerOutputs.tottime(timer)
    metrics["rhs! time"] = 1.0e-9 * TimerOutputs.time(timer["rhs!"])

    # compute performance index
    nrhscalls = Trixi.ncalls(semi.performance_counter)
    walltime = 1.0e-9 * take!(semi.performance_counter)
    metrics["PID"] = walltime * Trixi.mpi_nranks() / (Trixi.ndofsglobal(semi) * nrhscalls)

    # write json file
    open("metrics.out", "w") do f
        indent = 2
        JSON.print(f, metrics, indent)
    end

    # run profiler
    maxiters = 5
    initial_refinement_level = 2

    println("Running profiler (Float64)...")
    trixi_include(elixir_path,
                  maxiters = maxiters,
                  initial_refinement_level = initial_refinement_level,
                  storage_type = storage_type,
                  real_type = Float64,
                  run_profiler = true)

    open("profile_float64.txt", "w") do io
        show(io, prof_result)
    end

    println("Running profiler (Float32)...")
    trixi_include(elixir_path,
                  maxiters = maxiters,
                  initial_refinement_level = initial_refinement_level,
                  storage_type = storage_type,
                  real_type = Float32,
                  run_profiler = true)

    open("profile_float32.txt", "w") do io
        show(io, prof_result)
    end
end

# hardcoded elixir
elixir_path = joinpath(@__DIR__(), "elixir_euler_taylor_green_vortex.jl")

main(elixir_path)
