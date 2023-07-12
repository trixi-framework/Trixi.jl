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
mutable struct T8codeMesh{NDIMS, RealT <: Real, Forest} <: AbstractMesh{NDIMS}
    forest::Forest
    number_trees::Int
    # number_elements::Int
    # tree shapes
    max_number_faces::Int

    function T8codeMesh{NDIMS}(forest, max_number_faces) where {NDIMS}
        @assert NDIMS == 2
        number_trees = t8_forest_get_num_local_trees(forest)

        mesh = new{NDIMS, Cdouble, typeof(forest)}(forest, number_trees, max_number_faces)

        return mesh
    end
end

@inline Base.ndims(::T8codeMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::T8codeMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline nelements_global(mesh::T8codeMesh, solver, cache) = t8_forest_get_global_num_elements(mesh.forest)

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

function create_cache(mesh::T8codeMesh, equations,
                      solver, RealT, uEltype)
    elements = init_elements(mesh, RealT, uEltype)

    cache = (; elements)

    return cache
end
end # @muladd
