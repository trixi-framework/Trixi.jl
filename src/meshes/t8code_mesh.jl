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

function output_data_to_vtu(mesh, semi, prefix)
    @unpack element_data = semi.cache.elements
    n_elements = length(element_data)

    # We need to allocate a new array to store the data on their own.
    # These arrays have one entry per local element.
    u = Vector{Cdouble}(undef, n_elements)

    # Copy the elment's volumes from our data array to the output array.
    for ielem = 1:n_elements
      u[ielem] = semi.initial_condition(element_data[ielem].midpoint, 0.0, semi.equations)
    end

    vtk_data = [
      t8_vtk_data_field_t(
        T8_VTK_SCALAR,
        NTuple{8192, Cchar}(rpad("scalar\0", 8192, ' ')),
        pointer(u),
      ),
    ]

    # The number of user defined data fields to write.
    num_data = length(vtk_data)

    # Write user defined data to vtu file.
    write_treeid = 1
    write_mpirank = 1
    write_level = 1
    write_element_id = 1
    write_ghosts = 0
    t8_forest_write_vtk_ext(mesh.forest, prefix, write_treeid, write_mpirank,
                             write_level, write_element_id, write_ghosts,
                             0, 0, num_data, pointer(vtk_data))
  end
end # @muladd
