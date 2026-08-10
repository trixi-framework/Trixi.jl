
using Pkg
Pkg.activate("C:/Users/Shahu Karale/Trixi_Fork")

using Trixi
using CUDA
using KernelAbstractions

#using CUDA, KernelAbstractions, Adapt, Trixi

Trixi.storage_type(::Type{<:CuArray}) = CuArray

# 4. Load and run the P4est AMR simulation
println("Starting simulation...")
trixi_include("C:/Users/Shahu Karale/Trixi_Fork/examples/p4est_3d_dgsem/elixir_advection_nonconforming.jl")

# 1. Setup the simulation (Replace with your specific 3D elixir if needed)
#trixi_include("examples/p4est_3d_dgsem/elixir_advection_amr.jl")

# 2. Extract and Adapt to GPU
semi_gpu = Trixi.trixi_adapt(CuArray, Float64, semi)
cache_gpu = semi_gpu.cache
mortar_gpu = semi_gpu.solver.mortar
surface_integral_gpu = semi_gpu.solver.surface_integral
solver_gpu = semi_gpu.solver
mesh = semi_gpu.mesh
equations = semi_gpu.equations

surface_flux_values_gpu = cache_gpu.elements.surface_flux_values
nonconservative_terms = Trixi.have_nonconservative_terms(equations)
backend = CUDABackend()

# 3. Warm-up: Compile everything WITHOUT profiling
println("Warming up the JIT compiler...")
Trixi.calc_mortar_flux!(backend, surface_flux_values_gpu, mesh, nonconservative_terms, 
                        equations, mortar_gpu, surface_integral_gpu, solver_gpu, cache_gpu)
#KernelAbstractions.synchronize(backend)
CUDA.synchronize()

# 4. The Targeted Profile
# println("Starting hardware profiling...")
# CUDA.@profile begin
#     Trixi.calc_mortar_flux!(backend, surface_flux_values_gpu, mesh, nonconservative_terms, 
#                             equations, mortar_gpu, surface_integral_gpu, solver_gpu, cache_gpu)
#     #KernelAbstractions.synchronize(backend)
#     CUDA.synchronize()
# end
# 4. The Targeted Profile
println("Starting hardware profiling...")

# Force the C-API signal, bypassing Julia's macro guardrails
CUDA.Profile.start()

Trixi.calc_mortar_flux!(backend, surface_flux_values_gpu, mesh, nonconservative_terms, 
                        equations, mortar_gpu, surface_integral_gpu, solver_gpu, cache_gpu)
KernelAbstractions.synchronize(backend)

# Explicitly stop the profiler
CUDA.Profile.stop()

println("Profiling complete.")

println("Profiling complete.")