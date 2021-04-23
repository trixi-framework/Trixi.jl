using OrdinaryDiffEq
using StaticArrays
using StructArrays
using RecursiveArrayTools

RecursiveArrayTools.recursivecopy(u::StructArray) = copy(u)
function rhs!(du::StructArray,u::StructArray,p,t)
    du .= u
    return nothing
end
struct Foo{T} <: FieldVector{2,T}
    u::T
    v::T
end
u = StructArray([Foo(1.,2.) for i = 1:5])
du = similar(u)
prob = ODEProblem(rhs!,u,(0.0,.1))
sol = solve(prob,Tsit5())

# tuple_to_SoA(x::NTuple{N,AbstractArray{T}}) = StructArray{SVector{N,T}}(x...)
# rhs!(du::VectorOfArray,u::VectorOfArray,p,t) = rhs!(tuple_to_SoA(du.u),tuple_to_SoA(u.u),p,t)

function RecursiveArrayTools.recursivecopy(a::AbstractArray{T,N}) where {T<:AbstractArray,N}
    if ArrayInterface.ismutable(a)
      b = similar(a)
      map!(recursivecopy,b,a)
    else
      ArrayInterface.restructure(a,map(recursivecopy,a))
    end
end
ArrayInterface.ismutable(x::StructArray) = true