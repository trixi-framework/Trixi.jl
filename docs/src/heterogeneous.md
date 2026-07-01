# Heterogeneous computing

Support for heterogeneous computing is currently being worked on.


## User-facing interface

GPU support in [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) is controlled via two keyword arguments to [`semidiscretize`](@ref):

- `storage_type`: the array type used for all internal data structures.
  Set to `CuArray` (from [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)) for NVIDIA GPUs
  or `ROCArray` (from [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)) for AMD GPUs.
  Defaults to `nothing`, which keeps the standard CPU `Array`.
- `real_type`: the floating-point element type. GPU workflows typically benefit from
  setting this to `Float32`. Defaults to `nothing`, which retains the type used when
  building the semidiscretization (usually `Float64`).

Both arguments can be used independently. A typical GPU setup looks like:

```julia
using CUDA   # or using AMDGPU
ode = semidiscretize(semi, tspan; real_type = Float32, storage_type = CuArray)
```

The rest of the elixir (callbacks, ODE solver call) remains unchanged. See, e.g.,
`examples/p4est_2d_dgsem/elixir_euler_source_terms.jl` for a concrete example.

!!! note "Single-precision computations using `Float32`"
    To use `Float32` consistently, make sure to write all equations, initial conditions,
    boundary conditions, and source terms in a type-stable manner — avoid hard-coded
    `Float64` literals and instead use, e.g., the `f0` suffix (`0.5f0`) for exact values
    or `convert(RealT, ...)` for non-exact ones, where `RealT = eltype(u)`.
    When using a [`StepsizeCallback`](@ref) for CFL-based step size control, also pass
    `dt = 1` (an integer) rather than `dt = 1.0` to `solve`.
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

## Element-type conversion with `Trixi.trixi_adapt`.

We can use [`Trixi.trixi_adapt`](@ref) to perform both an element-type and a storage-type adoption:

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

Given the following Trixi.jl code, which would typically be called from within `rhs!`:

```julia
function trixi_rhs_fct(mesh, equations, solver, cache, args)
    @threaded for element in eachelement(solver, cache)
        # code
    end
end
```

1. Move the inner code into a new inlined function `rhs_fct_per_element`.
   ```julia
   @inline function rhs_fct_per_element(..., element, ...)
       ...
   end
   ```
   Besides the index `element`, pass all required fields as arguments, but make sure to
   `@unpack` them from their structs in advance.
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
       @threaded for element in eachelement(solver, cache)
           rhs_fct_per_element(element, unpacked_args, args)
       end
   end
   ```
4. When `backend` is a `Backend` (a type defined by KernelAbstractions.jl), write a
   KernelAbstractions.jl kernel:
   ```julia
   function trixi_rhs_fct(backend::Backend, mesh, equations, solver, cache, args)
       nelements(solver, cache) == 0 && return nothing  # return early when there are no elements
       @unpack unpacked_args = cache
       kernel! = rhs_fct_kernel!(backend)
       kernel!(unpacked_args, args,
               ndrange = nelements(solver, cache))
       return nothing
   end

   @kernel function rhs_fct_kernel!(unpacked_args, args)
       element = @index(Global)
       rhs_fct_per_element(element, unpacked_args, args)
   end
   ```
