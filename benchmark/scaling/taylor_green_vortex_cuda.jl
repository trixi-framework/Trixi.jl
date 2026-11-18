# Taylor-Green vortex of the CUDA benchmark with one GPU per rank.

using CUDA

CUDA.device!(Trixi.mpi_rank())
elixir = joinpath(@__DIR__, "..", "CUDA", "elixir_euler_taylor_green_vortex.jl")
overrides = (; initial_refinement_level = 3, maxiters = 500, storage_type = CuArray,
             real_type = Float64, analysis_callback = TrivialCallback())
