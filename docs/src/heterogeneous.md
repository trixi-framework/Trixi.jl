# Heterogeneous computing

Support for heterogeneous computing is currently being worked on.


## User-facing interface

GPU support in [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) is controlled via
the keyword arguments of [`trixi_adapt`](@ref), which adapts a semidiscretization:

- `storage_type`: the array type used for all internal data structures.
  Set to `CuArray` (from [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)) for NVIDIA GPUs,
  `ROCArray` (from [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)) for AMD GPUs,
  or `MtlArray` (from [Metal.jl](https://github.com/JuliaGPU/Metal.jl), experimental)
  for Apple GPUs.
  Defaults to `nothing`, which keeps the current array type (usually the standard
  CPU `Array`).
- `real_type`: the floating-point element type. GPU workflows typically benefit from
  setting this to `Float32`. Defaults to `nothing`, which keeps the current real type
  (usually `Float64`).
- `tuning`: performance settings of the GPU kernels as a [`Tuning`](@ref). They select
  between different implementations of the same discretization, which are equivalent in
  exact arithmetic but can differ in performance, in the resources required on a GPU,
  and in their results at the level of floating-point round-off errors. For example,
  `tuning = Tuning(flux_differencing_kernel = FullSweep())` selects the GPU kernel used
  for [`VolumeIntegralFluxDifferencing`](@ref), one of [`HalfSweep()`](@ref) (default on
  GPUs), [`FullSweep()`](@ref), or [`FullSweepGlobal()`](@ref). Settings that are
  `nothing` keep their current value, and settings that were never set use the defaults
  of the backend. Defaults to `nothing`, which keeps all current settings.

All arguments can be used independently. A typical GPU setup looks like:

```julia
using CUDA   # or using AMDGPU, using Metal
semi = trixi_adapt(semi; real_type = Float32, storage_type = CuArray)
ode = semidiscretize(semi, tspan)
```

!!! warning "Experimental support for Apple GPUs via Metal.jl"
    Support for Apple GPUs via `storage_type = MtlArray` is experimental. Metal
    does not support `Float64`, so `real_type = Float32` is required. Because of
    upstream bugs in Metal.jl (see [Metal.jl#1017](https://github.com/JuliaGPU/Metal.jl/issues/1017)
    and [Metal.jl#1018](https://github.com/JuliaGPU/Metal.jl/issues/1018)), some setups
    currently crash, hang, or silently produce wrong results on Apple M1/M2 GPUs,
    in particular for 3D meshes.

The adapted semidiscretization replaces the original one, which may share data with
the adapted one and must not be used anymore. Thus, create the callbacks after calling
`trixi_adapt` so that they use the adapted semidiscretization. Otherwise, the rest of
the elixir (callbacks, ODE solver call) remains unchanged. [`semidiscretize`](@ref)
converts the time span `tspan` to the real type of the semidiscretization, so you can
keep, e.g., `tspan = (0.0, 1.0)`. See, e.g.,
`examples/p4est_2d_dgsem/elixir_euler_source_terms.jl` for a concrete example.

Mixed hyperbolic-parabolic problems, i.e., a [`SemidiscretizationHyperbolicParabolic`](@ref),
are supported on conforming (non-adaptive) `P4estMesh{2}`s and `P4estMesh{3}`s as well,
see, e.g., `examples/p4est_2d_dgsem/elixir_navierstokes_lid_driven_cavity.jl` and
`examples/p4est_3d_dgsem/elixir_navierstokes_convergence.jl`.

Hyperbolic problems on conforming `P4estMesh`es in 2D can also be run on multiple GPUs
with MPI, using one MPI rank per GPU. The setup is the same as above, and the elixir is
started with MPI as described in [Parallelization](@ref parallel_system_MPI).
This requires a local CUDA/ROCm-aware MPI installation, since the MPI buffers are stored
on the GPU and passed directly to MPI. Mixed hyperbolic-parabolic problems, 3D meshes,
and non-conforming meshes (mortars, AMR) are not supported with multiple GPUs yet.
See the section on
[GPU-aware MPI support](https://juliaparallel.org/MPI.jl/stable/usage/#GPU-aware-MPI-support)
in the MPI.jl documentation for details, including example scripts such as
`alltoall_test_cuda.jl` and `alltoall_test_rocm.jl` that check whether your MPI
installation is GPU-aware. The functions `MPI.has_cuda()` and `MPI.has_rocm()` can query
GPU support only from Open MPI; for other MPI implementations, they usually return `false`
(see their docstrings for details).

If a node has multiple GPUs, each MPI rank on this node must use a different GPU.
As described in the MPI.jl documentation on
[multiple GPUs per node](https://juliaparallel.org/MPI.jl/stable/usage/#Multiple-GPUs-per-node),
there are two ways to achieve this:

1. Select the GPU in the elixir based on the rank within the node, which is obtained
   from a node-local MPI communicator. Do this before creating any GPU arrays, i.e.,
   before calling `trixi_adapt`. The function `KernelAbstractions.device!` of
   [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) selects
   the device in a backend-agnostic way. Using CUDA.jl, this reads
   ```julia
   using Trixi, MPI, CUDA
   backend = CUDABackend() # or `ROCBackend()` with `using AMDGPU`
   comm_node = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED,
                                   MPI.Comm_rank(MPI.COMM_WORLD))
   KernelAbstractions.device!(backend, MPI.Comm_rank(comm_node) + 1)
   ```
   If there are more MPI ranks than GPUs on a node, `KernelAbstractions.device!`
   throws an `ArgumentError`.
2. Let the job scheduler or the environment variables `CUDA_VISIBLE_DEVICES` or
   `ROCR_VISIBLE_DEVICES` expose a single GPU to each MPI rank
   (e.g., `srun --gpus-per-task=1` with SLURM). Then, each rank can use its default
   device. We strongly advice against this approach since we have experienced
   several issues with it.

!!! note "Single-precision computations using `Float32`"
    To use `Float32` consistently, make sure to write all equations, initial conditions,
    boundary conditions, and source terms in a type-stable manner — avoid hard-coded
    `Float64` literals and instead use, e.g., the `f0` suffix (`0.5f0`) for exact values
    or `convert(RealT, ...)` for non-exact ones, where `RealT = eltype(u)`.
    When using a [`StepsizeCallback`](@ref) for CFL-based step size control, also pass
    `dt = 1` (an integer) rather than `dt = 1.0` to `solve`. Otherwise, the time
    integrator promotes the time to `Float64`, and the [`SummaryCallback`](@ref)
    warns about it.
    See [Numeric types and type stability](@ref numeric-types) for detailed guidelines.


## Internal use of Adapt.jl

[Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) is a package in the
[JuliaGPU](https://github.com/JuliaGPU) family that allows for
the translation of nested data structures. The primary goal is to allow the substitution of `Array`
at the storage level with a GPU array like `CuArray` from [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

To facilitate this, data structures must be parameterized, so instead of:

```julia
struct Container <: Trixi.AbstractContainer
   data::Array{Float64, 2}
end
```

They must be written as:

```jldoctest adapt; output = false, setup=:(import Trixi)
struct Container{D<:AbstractArray} <: Trixi.AbstractContainer
   data::D
end

# output

```

furthermore, we need to define a function that allows for the conversion of storage
of our types:

```jldoctest adapt; output = false
using Adapt

function Adapt.adapt_structure(to, C::Container)
    return Container(adapt(to, C.data))
end

# output

```

or simply

```julia
Adapt.@adapt_structure(Container)
```

additionally, we must define `Adapt.parent_type`.

```jldoctest adapt; output = false
function Adapt.parent_type(::Type{<:Container{D}}) where D
    return D
end

# output

```

All together we can use this machinery to perform conversions of a container.

```jldoctest adapt
julia> C = Container(zeros(3))
Container{Vector{Float64}}([0.0, 0.0, 0.0])

julia> Trixi.storage_type(C)
Array
```


```julia-repl
julia> using CUDA

julia> GPU_C = adapt(CuArray, C)
Container{CuArray{Float64, 1, CUDA.DeviceMemory}}([0.0, 0.0, 0.0])

julia> Trixi.storage_type(C)
CuArray
```

## Element-type conversion with `trixi_adapt`

We can use [`trixi_adapt`](@ref) to perform both an element-type and a storage-type adaptation:

```jldoctest adapt
julia> C = Container(zeros(3))
Container{Vector{Float64}}([0.0, 0.0, 0.0])

julia> Trixi.trixi_adapt(Array, Float32, C)
Container{Vector{Float32}}(Float32[0.0, 0.0, 0.0])
```

```julia-repl
julia> Trixi.trixi_adapt(CuArray, Float32, C)
Container{CuArray{Float32, 1, CUDA.DeviceMemory}}(Float32[0.0, 0.0, 0.0])
```

!!! note
    `adapt(Array{Float32}, C)` is tempting, but it will do the wrong thing
    in the presence of `SVector`s and similar arrays from StaticArrays.jl.


## Writing GPU kernels

Offloading computations to the GPU is done with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
allowing for vendor-agnostic GPU code.

### Example

Given the following Trixi.jl code, which would typically be called from within `rhs_hyperbolic!`:

```julia
function trixi_rhs_fct(mesh, equations, solver, cache, args)
    @threaded for element in eachelement(solver, cache)
        # code
    end
end
```

1. Move the inner code into a new inlined function `rhs_fct_per_element`.
   ```julia
   Base.@propagate_inbounds function rhs_fct_per_element(..., element, ...)
       ...
   end
   ```
   Besides the index `element`, pass all required fields as arguments, but make sure to
   `@unpack` them from their structs in advance.
   `Base.@propagate_inbounds` inlines the function and propagates `@inbounds`
   from the caller, so that bounds checks inside the function are removed when it is
   called from an `@inbounds` context such as the kernel below (see
   [Enabling bounds checking](@ref enabling-bounds-checking)).
2. Where `trixi_rhs_fct` is called, get the backend, i.e., the hardware we are currently
   running on via `trixi_backend(x)`.
   This will, e.g., work with `u_ode`. Internally, KernelAbstractions.jl's `get_backend`
   will be called, i.e., KernelAbstractions.jl has to know the type of `x`.
   ```julia
   backend = trixi_backend(u_ode)
   ```
3. Add a new argument `backend` to `trixi_rhs_fct` used for dispatch.
   When `backend` is `nothing`, the legacy implementation should be used:
   ```julia
   function trixi_rhs_fct(backend::Nothing, mesh, equations, solver, cache, args)
       @unpack unpacked_args = cache
       # Explicit bounds check, which allows us to assume inbounds access below
       @boundscheck begin
           check_axes(unpacked_args, mesh, equations, solver, cache)
       end
       @threaded for element in eachelement(solver, cache)
           @inbounds rhs_fct_per_element(element, unpacked_args, args)
       end
   end
   ```
4. When `backend` is a `Backend` (a type defined by KernelAbstractions.jl), write a
   KernelAbstractions.jl kernel:
   ```julia
   function trixi_rhs_fct(backend::Backend, mesh, equations, solver, cache, args)
       nelements(solver, cache) == 0 && return nothing  # return early when there are no elements
       @unpack unpacked_args = cache
       # Explicit bounds check, which allows us to assume inbounds access in the kernel
       @boundscheck begin
           check_axes(unpacked_args, mesh, equations, solver, cache)
       end
       kernel! = rhs_fct_kernel!(backend)
       kernel!(unpacked_args, args,
               ndrange = nelements(solver, cache))
       return nothing
   end

   @kernel inbounds=true function rhs_fct_kernel!(unpacked_args, args)
       element = @index(Global)
       rhs_fct_per_element(element, unpacked_args, args)
   end
   ```
   The kernel uses `inbounds=true`, i.e., it assumes that all accesses are in bounds.
   Thus, verify the axes of all arrays accessed in the kernel (directly or in functions
   called there) once in an explicit `@boundscheck` block in the function launching the
   kernel, as described in [Enabling bounds checking](@ref enabling-bounds-checking).
   Perform these checks on the host before launching the kernel, not inside the kernel.
   When Julia is started with `--check-bounds=yes`, the `@boundscheck` block is always
   executed and `inbounds=true` is ignored, so that out-of-bounds accesses are caught.
