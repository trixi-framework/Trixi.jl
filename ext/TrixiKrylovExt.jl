module TrixiKrylovExt

using Krylov: Krylov
import Krylov: FloatOrComplex

using LinearAlgebra: dot, norm

using Trixi: TrixiStateVector

# kdot and knorm delegate to the MPI-reduced LinearAlgebra overloads on TrixiStateVector.
# Krylov.jl calls these instead of the standard LA names for its internal workspace vectors.

function Krylov.kdot(n::Integer, x::TrixiStateVector{T},
                     y::TrixiStateVector{T}) where {T <: FloatOrComplex}
    return dot(x, y)
end

function Krylov.knorm(n::Integer,
                      x::TrixiStateVector{T}) where {T <: FloatOrComplex}
    return norm(x)
end

# In-place kernels bypass the TrixiStateVector broadcast path to avoid threading
# overhead for these simple per-element operations.

function Krylov.kscal!(n::Integer, s::T,
                       x::TrixiStateVector{T}) where {T <: FloatOrComplex}
    x.data .*= s
    return x
end

function Krylov.kdiv!(n::Integer, x::TrixiStateVector{T},
                      s::T) where {T <: FloatOrComplex}
    x.data ./= s
    return x
end

function Krylov.kaxpy!(n::Integer, s::T, x::TrixiStateVector{T},
                       y::TrixiStateVector{T}) where {T <: FloatOrComplex}
    y.data .+= s .* x.data
    return y
end

function Krylov.kaxpby!(n::Integer, s::T, x::TrixiStateVector{T}, t::T,
                        y::TrixiStateVector{T}) where {T <: FloatOrComplex}
    y.data .= s .* x.data .+ t .* y.data
    return y
end

function Krylov.kcopy!(n::Integer, y::TrixiStateVector{T},
                       x::TrixiStateVector{T}) where {T <: FloatOrComplex}
    y.data .= x.data
    return y
end

function Krylov.kscalcopy!(n::Integer, y::TrixiStateVector{T}, s::T,
                           x::TrixiStateVector{T}) where {T <: FloatOrComplex}
    y.data .= s .* x.data
    return y
end

function Krylov.kdivcopy!(n::Integer, y::TrixiStateVector{T},
                          x::TrixiStateVector{T},
                          s::T) where {T <: FloatOrComplex}
    y.data .= x.data ./ s
    return y
end

function Krylov.kfill!(x::TrixiStateVector{T},
                       val::T) where {T <: FloatOrComplex}
    fill!(x.data, val)
    return x
end

end # module TrixiKrylovExt
