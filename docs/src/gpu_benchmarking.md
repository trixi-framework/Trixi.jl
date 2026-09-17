# GPU benchmarking

When benchmarking Trixi.jl on a GPU, make sure that the data used by the benchmark and the semidiscretization stored in the `ODEProblem` are on the GPU before measuring the kernel execution time. In particular, create the `ODEProblem` first and then adapt its arrays. Using `semi` directly in place of `ode.p` in the `rhs!` call does not benchmark the same object that is stored in the ODE problem.

For NVIDIA GPUs, the following example uses `CuArray` from [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). For AMD GPUs, replace `CUDA`/`CuArray` with the corresponding [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) types.

## Manual GPU benchmarking

A simple way to benchmark a GPU RHS evaluation is:

```julia
using BenchmarkTools
using CUDA
using SciMLBase: remake
using Trixi

CUDA.allowscalar(false)

trixi_include("examples/p4est_2d_dgsem/elixir_advection_basic.jl",
              tspan = (0.0, 1.0))

# Create the ODE problem on the CPU first.
# The ODE problem stores the semidiscretization in `ode.p`.
ode_gpu = remake(ode,
                  u0 = Trixi.trixi_adapt(CuArray, Float32, ode.u0),
                  p = Trixi.trixi_adapt(CuArray, Float32, ode.p))

du_gpu = similar(ode_gpu.u0)

# Warm up the GPU and trigger compilation before measuring.
Trixi.rhs!(du_gpu, ode_gpu.u0, ode_gpu.p, first(ode_gpu.tspan))
CUDA.@sync Trixi.rhs!(du_gpu, ode_gpu.u0, ode_gpu.p, first(ode_gpu.tspan))

@benchmark CUDA.@sync Trixi.rhs!($du_gpu,
                                  $(ode_gpu.u0),
                                  $(ode_gpu.p),
                                  $(first(ode_gpu.tspan)))
```

The important details are:

1. Construct `ode` before adapting the arrays to the GPU.
2. Adapt both `ode.u0` and `ode.p` to the GPU. `ode.p` contains the semidiscretization used by the ODE problem.
3. Pass `ode.p`, not the original `semi`, to `Trixi.rhs!`.
4. Warm up the GPU before collecting measurements so compilation is not included in the benchmark.
5. Use `CUDA.@sync` around the operation being measured. GPU kernel launches are asynchronous, so synchronizing is necessary to measure execution time rather than only kernel launch overhead.
6. Use interpolation (`$`) for benchmark inputs to prevent `BenchmarkTools.jl` from including their lookup in the measured expression.

For a full GPU simulation, the same principle applies: configure the `ODEProblem` with GPU-resident data before calling the ODE solver. Trixi.jl also supports passing `storage_type = CuArray` and `real_type = Float32` directly to `semidiscretize`; see the [heterogeneous computing documentation](https://trixi-framework.github.io/TrixiDocumentation/stable/heterogeneous/) for the user-facing GPU interface.

!!! warning "GPU synchronization"
    Do not benchmark a CUDA kernel without synchronization. For example, `@benchmark Trixi.rhs!(...)`
    can report mostly kernel-launch overhead because the GPU work may still be running when the
    benchmark expression returns. Use `CUDA.@sync` for CUDA benchmarks.

!!! note "Benchmark the same workload"
    Keep the mesh, equations, polynomial degree, numerical flux, and other relevant setup parameters
    identical when comparing CPU and GPU implementations. Exclude one-time compilation and data-transfer
    costs unless those costs are specifically part of the performance question you are investigating.
