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
    max_number_faces::Int
    number_elements::Int
    # number_ghost_elements
    # tree shapes
    current_filename::String
    unsaved_changes::Bool

    function T8codeMesh{NDIMS}(forest, max_number_faces; current_filename = "",
                               unsaved_changes = true) where {NDIMS}
        @assert NDIMS == 2
        number_trees = t8_forest_get_num_local_trees(forest)

        number_elements = t8_forest_get_local_num_elements(forest)

        mesh = new{NDIMS, Cdouble, typeof(forest)}(forest, number_trees,
                                                   max_number_faces, number_elements,
                                                   current_filename, unsaved_changes)

        return mesh
    end
end

@inline Base.ndims(::T8codeMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::T8codeMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline nelements(mesh::T8codeMesh, solver, cache) = nelementsglobal(mesh, solver, cache)

@inline function eachelement(mesh::T8codeMesh, solver, cache)
    eachelement(mesh, solver)
end

@inline function eachelement(mesh::T8codeMesh, solver)
    Base.OneTo(mesh.number_elements)
end

@inline function nelementsglobal(mesh::T8codeMesh, solver, cache)
    nelementsglobal(mesh)
end

@inline function nelementsglobal(mesh::T8codeMesh)
    t8_forest_get_global_num_elements(mesh.forest)
end

function Base.show(io::IO, mesh::T8codeMesh)
    print(io, "T8codeMesh{", ndims(mesh), ", ", real(mesh), "}(")
    print(io, "#trees: ", mesh.number_trees)
    print(io, ", #elements: ", nelementsglobal(mesh))
end

function Base.show(io::IO, ::MIME"text/plain", mesh::T8codeMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_header(io,
                       "T8codeMesh{" * string(ndims(mesh)) * ", " * string(real(mesh)) *
                       "}")
        summary_line(io, "#trees", mesh.number_trees)
        summary_line(io, "#elements", nelementsglobal(mesh))
        summary_footer(io)
    end
end

function create_cache(mesh::T8codeMesh, equations,
                      solver, RealT, uEltype)
    elements = init_elements(mesh, RealT, uEltype)

    u_ = init_solution!(mesh, equations)

    cache = (; elements, u_)

    return cache
end

# Write the forest as vtu and also write the element's volumes in the file.
#
# t8code supports writing element based data to vtu as long as its stored
# as doubles. Each of the data fields to write has to be provided in its own
# array of length num_local_elements.
# We support two types: T8_VTK_SCALAR - One double per element.
#                  and  T8_VTK_VECTOR - Three doubles per element.
function output_data_to_vtu(mesh::T8codeMesh, equations, solver, u, out)
    vars = varnames(cons2cons, equations)
    vtk_data = [t8_vtk_data_field_t(T8_VTK_SCALAR,
                                    NTuple{8192, Cchar}(rpad("$(vars[v])\0", 8192, ' ')),
                                    pointer([u[element].u[v]
                                             for element in eachelement(mesh, solver)]))
                for v in eachvariable(equations)]

    # The number of user defined data fields to write.
    num_data = length(vtk_data)

    # Write user defined data to vtu file.
    write_treeid = 1
    write_mpirank = 1
    write_level = 1
    write_element_id = 1
    write_ghosts = 0
    t8_forest_write_vtk_ext(mesh.forest, out, write_treeid, write_mpirank,
                            write_level, write_element_id, write_ghosts,
                            0, 0, num_data, pointer(vtk_data))
end
end # @muladd
