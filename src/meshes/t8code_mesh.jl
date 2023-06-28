# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent


"""
    T8codeMesh{NDIMS} <: AbstractMesh{NDIMS}

An unstructured curved mesh based on trees that uses the library `t8code`
to manage trees and mesh refinement.
"""
mutable struct T8codeMesh{NDIMS, RealT <: Real, Forest, Elements} <: AbstractMesh{NDIMS}
    forest::Forest
    elements::Elements

    function T8codeMesh{NDIMS}(forest, elements) where {NDIMS}
        @assert NDIMS == 2
        mesh = new{NDIMS, eltype(elements[1].volume), typeof(forest), typeof(elements)}(forest, elements)

        # Clean-up t8code mesh
        T8code.t8_forest_unref(Ref(forest))

        return mesh
    end
end



@inline Base.ndims(::T8codeMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::T8codeMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

# ndims(mesh::T8codeMesh) = length(mesh.elements[1].midpoint)
# real(mesh::T8codeMesh) = eltype(mesh.elements[1].volume)

function Base.show(io::IO, mesh::T8codeMesh)
    print(io, "T8codeMesh{", ndims(mesh), ", ", real(mesh), "}")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::T8codeMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_box(io,
                    "T8codeMesh{" * string(ndims(mesh)) * ", " * string(real(mesh)) * "}")
    end
end
end # @muladd
