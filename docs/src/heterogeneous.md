# Heterogeneous computing

Support for heterogeneous computing is currently being worked on.

## The use of Adapt.jl

[Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) is a package in the
[JuliaGPU](https://github.com/JuliaGPU) family that allows for
the translation of nested data structures. The primary goal is to allow the substitution of `Array` 
at the storage leaves with a GPU array like `CuArray` from [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

To facilitate this data structures must be parameterized, so instead of:

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