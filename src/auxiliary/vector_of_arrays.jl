# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Wraps a Vector of Arrays, forwards `getindex` to the underlying Vector.
# Implements `Adapt.adapt_structure` to allow offloading to the GPU which is
# not possible for a plain Vector of Arrays.
struct VecOfArrays{T <: AbstractArray}
    arrays::Vector{T}
end
Base.getindex(v::VecOfArrays, i::Int) = Base.getindex(v.arrays, i)
Base.IndexStyle(v::VecOfArrays) = Base.IndexStyle(v.arrays)
Base.size(v::VecOfArrays) = Base.size(v.arrays)
Base.length(v::VecOfArrays) = Base.length(v.arrays)
Base.eltype(v::VecOfArrays{T}) where {T} = T
function Adapt.adapt_structure(to, v::VecOfArrays)
    return VecOfArrays([Adapt.adapt(to, arr) for arr in v.arrays])
end
function Adapt.parent_type(::Type{<:VecOfArrays{T}}) where {T}
    return T
end
function Adapt.unwrap_type(A::Type{<:VecOfArrays})
    Adapt.unwrap_type(Adapt.parent_type(A))
end
function Base.convert(::Type{<:VecOfArrays}, v::Vector{<:AbstractArray})
    VecOfArrays(v)
end
end # @muladd
