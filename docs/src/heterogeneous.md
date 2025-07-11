# Heterogeneous computing

Support for heterogeneous computing is currently being worked on.

## The use of Adapt.jl

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

We can use [`Trixi.trixi_adapt`](@ref) to perform both an element-type and a storage-type adoption

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
    `adapt(Array{Float32}, C)` is tempting but will do the wrong thing in the presence of `StaticArrays`.


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

1.  Put the inner code in a new function `rhs_fct_per_element`. Besides the index
    `element`, pass all required fields as arguments, but make sure to `@unpack` them from
    their structs in advance.

2.  Where `trixi_rhs_fct` is called, get the backend, i.e. the hardware we are currently
    running on via `trixi_backend(x)`.
    This will, e.g., work with `u_ode`. Internally, `KernelAbstractions.jl`'s `get_backend`
    will be called, i.e. `KernelAbstractions.jl` has to know the type of `x`.

    ```julia
    backend = trixi_backend(u_ode)
    ```

3.  Add a new argument `backend` to `trixi_rhs_fct` used for dispatch.
    When `backend` is `nothing`, the legacy implementation should be used:
    ```julia
    function trixi_rhs_fct(backend::Nothing, mesh, equations, solver, cache, args)
        @unpack unpacked_args = cache
        @threaded for element in eachelement(solver, cache)
            rhs_fct_per_element(element, unpacked_args, args)
        end
    end
    ```

4.  When `backend` is a `Backend` (a type defined by `KernelAbstractions.jl`), write a
    `KernelAbstractions.jl` kernel:
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