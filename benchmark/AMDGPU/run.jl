using Trixi
using AMDGPU
using TimerOutputs
using JSON

function main(elixir_path)

    # setup
    maxiters = 50
    initial_refinement_level = 3
    storage_type = ROCArray
    real_type = Float64

    println("Warming up...")

    # start simulation with tiny final time to trigger compilation
    duration_compile = @elapsed begin
        trixi_include(elixir_path,
                      tspan = (0.0, 1e-14),
                      storage_type = storage_type,
                      real_type = real_type)
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
    latest_semi = @invokelatest (@__MODULE__).semi
    nrhscalls = Trixi.ncalls(latest_semi.performance_counter)
    walltime = 1.0e-9 * take!(latest_semi.performance_counter)
    metrics["PID"] = walltime * Trixi.mpi_nranks() /
                     (Trixi.ndofsglobal(latest_semi) * nrhscalls)

    # write json file
    open("metrics.out", "w") do f
        indent = 2
        JSON.print(f, metrics, indent)
    end
end

# hardcoded elixir
elixir_path = joinpath(@__DIR__(), "elixir_euler_taylor_green_vortex.jl")

main(elixir_path)
