# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    T8codeFVMesh{NDIMS} <: AbstractMesh{NDIMS}

An unstructured curved mesh based on trees that uses the library `t8code`
to manage trees and mesh refinement.
"""
mutable struct T8codeFVMesh{NDIMS, RealT <: Real, Forest} <: AbstractMesh{NDIMS}
    forest::Forest
    number_trees_global::Int
    number_trees_local::Int
    max_number_faces::Int
    number_elements::Int
    # number_ghost_elements
    # tree shapes
    current_filename::String
    unsaved_changes::Bool

    function T8codeFVMesh{NDIMS}(mesh_function, initial_refinement_level;
                               current_filename = "",
                               unsaved_changes = true) where {NDIMS}
        @assert NDIMS == 2
        comm = MPI.COMM_WORLD

        cmesh = mesh_function(comm, NDIMS)
        scheme = t8_scheme_new_default_cxx()

        do_face_ghost = 1
        forest = t8_forest_new_uniform(cmesh, scheme, initial_refinement_level,
                                       do_face_ghost, comm)

        number_trees_global = t8_forest_get_num_global_trees(forest)
        number_trees_local = t8_forest_get_num_local_trees(forest)

        number_elements = t8_forest_get_local_num_elements(forest)

        # Very ugly way to get the maximum number of faces automatically.
        max_number_faces = 0
        for itree in 0:(number_trees_local - 1)
            tree_class = t8_forest_get_tree_class(forest, itree)
            eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

            # Get the number of elements of this tree.
            num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

            # Loop over all local elements in the tree.
            for ielement in 0:(num_elements_in_tree - 1)
                element = t8_forest_get_element_in_tree(forest, itree, ielement)
                num_faces = t8_element_num_faces(eclass_scheme, element)
                max_number_faces = max(max_number_faces, num_faces)
            end
        end
        if mpi_isparallel()
            max_number_faces = MPI.Allreduce!(Ref(max_number_faces), max, mpi_comm())[]
        end
        mesh = new{NDIMS, Cdouble, typeof(forest)}(forest, number_trees_global,
                                                   number_trees_local,
                                                   max_number_faces, number_elements,
                                                   current_filename, unsaved_changes)

        return mesh
    end
end

@inline Base.ndims(::T8codeFVMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::T8codeFVMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline function nelements(mesh::T8codeFVMesh, solver, cache)
    nelementsglobal(mesh, solver, cache)
end

@inline function eachelement(mesh::T8codeFVMesh, solver, cache)
    eachelement(mesh, solver)
end

@inline function eachelement(mesh::T8codeFVMesh, solver)
    Base.OneTo(mesh.number_elements)
end

@inline function nelementsglobal(mesh::T8codeFVMesh, solver, cache)
    nelementsglobal(mesh)
end

@inline function nelementsglobal(mesh::T8codeFVMesh)
    t8_forest_get_global_num_elements(mesh.forest)
end

function Base.show(io::IO, mesh::T8codeFVMesh)
    print(io, "T8codeFVMesh{", ndims(mesh), ", ", real(mesh), "}(")
    print(io, "#trees: ", mesh.number_trees_global)
    print(io, ", #elements: ", nelementsglobal(mesh))
end

function Base.show(io::IO, ::MIME"text/plain", mesh::T8codeFVMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_header(io,
                       "T8codeFVMesh{" * string(ndims(mesh)) * ", " * string(real(mesh)) *
                       "}")
        summary_line(io, "#trees", mesh.number_trees_global)
        summary_line(io, "#elements", nelementsglobal(mesh))
        summary_footer(io)
    end
end

function create_cache(mesh::T8codeFVMesh, equations,
                      solver, RealT, uEltype)
    elements = init_elements(mesh, RealT, uEltype)

    interfaces = init_interfaces(mesh, equations, elements)

    u_ = init_solution!(mesh, equations)

    cache = (; elements, interfaces, u_)

    return cache
end

# Write the forest as vtu and also write the element's volumes in the file.
#
# t8code supports writing element based data to vtu as long as its stored
# as doubles. Each of the data fields to write has to be provided in its own
# array of length num_local_elements.
# We support two types: T8_VTK_SCALAR - One double per element.
#                  and  T8_VTK_VECTOR - Three doubles per element.
function output_data_to_vtu(mesh::T8codeFVMesh, equations, solver, u_, out)
    vars = varnames(cons2cons, equations)

    vtk_data = Vector{t8_vtk_data_field_t}(undef, 2 * nvariables(equations))

    for v in eachvariable(equations)
        let
            data = [u_[element].u[v] for element in eachelement(mesh, solver)]
            data_ptr = pointer(data)

            GC.@preserve data begin
                vtk_data[v] = t8_vtk_data_field_t(T8_VTK_SCALAR,
                                                  NTuple{8192, Cchar}(rpad("$(vars[v])\0",
                                                                           8192, ' ')),
                                                  data_ptr)
            end
        end
    end
    for v in eachvariable(equations)
        data_ = Vector{eltype(u_[1].u)}(undef, 3 * mesh.number_elements)
        for element in eachelement(mesh, solver)
            idx = 3 * (element - 1)
            slope_ = Trixi.get_variable_wrapped(u_[element].slope, equations, v)
            for d in 1:ndims(equations)
                data_[idx + d] = slope_[d]
            end
            for d in 1:(3 - ndims(equations))
                data_[idx + ndims(equations) + d] = zero(eltype(u_[1].slope))
            end
        end

        GC.@preserve data_ begin
            vtk_data[nvariables(equations) + v] = t8_vtk_data_field_t(T8_VTK_VECTOR,
                                                                      NTuple{8192, Cchar
                                                                             }(rpad("slope_$(vars[v])\0",
                                                                                    8192,
                                                                                    ' ')),
                                                                      pointer(data_))
        end
    end

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

# Simple meshes

# Directly ported from: `src/t8_cmesh/t8_cmesh_examples.c: t8_cmesh_new_periodic_hybrid`.
function cmesh_new_periodic_hybrid(comm, n_dims)::t8_cmesh_t
    vertices = [ # Just all vertices of all trees. partly duplicated
        -1.0, -1.0, 0, # tree 0, triangle
        0, -1.0, 0,
        0, 0, 0,
        -1.0, -1.0, 0, # tree 1, triangle
        0, 0, 0,
        -1.0, 0, 0,
        0, -1.0, 0,    # tree 2, quad
        1.0, -1.0, 0,
        0, 0, 0,
        1.0, 0, 0,
        -1.0, 0, 0,    # tree 3, quad
        0, 0, 0,
        -1.0, 1.0, 0,
        0, 1.0, 0,
        0, 0, 0,       # tree 4, triangle
        1.0, 0, 0,
        1.0, 1.0, 0,
        0, 0, 0,       # tree 5, triangle
        1.0, 1.0, 0,
        0, 1.0, 0,
    ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    #
    # This is how the cmesh looks like. The numbers are the tree numbers:
    # Domain size [-1,1]^2
    #
    #   +---+---+
    #   |   |5 /|
    #   | 3 | / |
    #   |   |/ 4|
    #   +---+---+
    #   |1 /|   |
    #   | / | 2 |
    #   |/0 |   |
    #   +---+---+
    #

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # Use linear geometry
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 2, T8_ECLASS_QUAD)
    t8_cmesh_set_tree_class(cmesh, 3, T8_ECLASS_QUAD)
    t8_cmesh_set_tree_class(cmesh, 4, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 5, T8_ECLASS_TRIANGLE)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[(1 + 9):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 2, @views(vertices[(1 + 18):end]), 4)
    t8_cmesh_set_tree_vertices(cmesh, 3, @views(vertices[(1 + 30):end]), 4)
    t8_cmesh_set_tree_vertices(cmesh, 4, @views(vertices[(1 + 42):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 5, @views(vertices[(1 + 51):end]), 3)

    t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 0, 2, 0, 0, 0)
    t8_cmesh_set_join(cmesh, 0, 3, 2, 3, 0)

    t8_cmesh_set_join(cmesh, 1, 3, 0, 2, 1)
    t8_cmesh_set_join(cmesh, 1, 2, 1, 1, 0)

    t8_cmesh_set_join(cmesh, 2, 4, 3, 2, 0)
    t8_cmesh_set_join(cmesh, 2, 5, 2, 0, 1)

    t8_cmesh_set_join(cmesh, 3, 5, 1, 1, 0)
    t8_cmesh_set_join(cmesh, 3, 4, 0, 0, 0)

    t8_cmesh_set_join(cmesh, 4, 5, 1, 2, 0)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

function cmesh_new_periodic_quad(comm, n_dims)::t8_cmesh_t
    vertices = [ # Just all vertices of all trees. partly duplicated
        -1.0, -1.0, 0, # tree 0, quad
        1.0, -1.0, 0,
        -1.0, 1.0, 0,
        1.0, 1.0, 0,

        # rotated:
        # -1.0, 0.0, 0,  # tree 0, quad
        # 0.0, -1.0, 0,
        # 0.0, 1.0, 0,
        # 1.0, 0.0, 0,
    ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    #
    # This is how the cmesh looks like. The numbers are the tree numbers:
    #
    #   +---+
    #   |   |
    #   | 0 |
    #   |   |
    #   +---+
    #

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # Use linear geometry
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_QUAD)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 4)

    t8_cmesh_set_join(cmesh, 0, 0, 0, 1, 0)
    t8_cmesh_set_join(cmesh, 0, 0, 2, 3, 0)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

function cmesh_new_periodic_tri(comm, n_dims)::t8_cmesh_t
    vertices = [ # Just all vertices of all trees. partly duplicated
        -1.0, -1.0, 0, # tree 0, triangle
        1.0, -1.0, 0,
        1.0, 1.0, 0,
        -1.0, -1.0, 0, # tree 1, triangle
        1.0, 1.0, 0,
        -1.0, 1.0, 0,

        # rotated:
        # -1.0, 0, 0,  # tree 0, triangle
        # 0, -1.0, 0,
        # 1.0, 0, 0,
        # -1.0, 0, 0,  # tree 1, triangle
        # 1.0, 0, 0,
        # 0, 1.0, 0,
    ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    #
    # This is how the cmesh looks like. The numbers are the tree numbers:
    #
    #   +---+
    #   |1 /|
    #   | / |
    #   |/0 |
    #   +---+
    #

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # /* Use linear geometry */
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[(1 + 9):end]), 3)

    t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 0, 1, 0, 1, 0)
    t8_cmesh_set_join(cmesh, 0, 1, 2, 0, 1)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

function cmesh_new_periodic_tri2(comm, n_dims)::t8_cmesh_t
    vertices = [ # Just all vertices of all trees. partly duplicated
        -1.0, -1.0, 0,  # tree 0, triangle
        0, -1.0, 0,
        0, 0, 0,
        -1.0, -1.0, 0,  # tree 1, triangle
        0, 0, 0,
        -1.0, 0, 0,
        0, -1.0, 0,     # tree 2, triangle
        1.0, -1.0, 0,
        1.0, 0, 0,
        0, -1.0, 0,     # tree 3, triangle
        1.0, 0, 0,
        0, 0, 0,
        -1.0, 0, 0,     # tree 4, triangle
        0, 0, 0,
        -1.0, 1.0, 0,
        -1.0, 1.0, 0,   # tree 5, triangle
        0, 0, 0,
        0, 1.0, 0,
        0, 0, 0,        # tree 6, triangle
        1.0, 0, 0,
        0, 1.0, 0,
        0, 1.0, 0,      # tree 7, triangle
        1.0, 0, 0,
        1.0, 1.0, 0,
    ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    #
    # This is how the cmesh looks like. The numbers are the tree numbers:
    #
    #   +---+---+
    #   |\ 5|\ 7|
    #   | \ | \ |
    #   |4 \| 6\|
    #   +---+---+
    #   |1 /|3 /|
    #   | / | / |
    #   |/0 |/ 2|
    #   +---+---+
    #

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # /* Use linear geometry */
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 2, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 3, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 4, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 5, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 6, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 7, T8_ECLASS_TRIANGLE)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[(1 + 9):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 2, @views(vertices[(1 + 18):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 3, @views(vertices[(1 + 27):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 4, @views(vertices[(1 + 36):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 5, @views(vertices[(1 + 45):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 6, @views(vertices[(1 + 54):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 7, @views(vertices[(1 + 63):end]), 3)

    t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 0, 3, 0, 1, 0)
    t8_cmesh_set_join(cmesh, 0, 5, 2, 1, 0)

    t8_cmesh_set_join(cmesh, 1, 4, 0, 2, 1)
    t8_cmesh_set_join(cmesh, 1, 2, 1, 0, 0)

    t8_cmesh_set_join(cmesh, 2, 3, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 2, 7, 2, 1, 0)

    t8_cmesh_set_join(cmesh, 3, 6, 0, 2, 1)

    t8_cmesh_set_join(cmesh, 4, 5, 0, 2, 1)
    t8_cmesh_set_join(cmesh, 4, 7, 1, 0, 0)

    t8_cmesh_set_join(cmesh, 5, 6, 0, 1, 0)

    t8_cmesh_set_join(cmesh, 6, 7, 0, 2, 1)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

# Directly ported from: `src/t8_cmesh/t8_cmesh_examples.c: t8_cmesh_new_periodic_hybrid`.
function cmesh_new_periodic_hybrid2(comm, n_dims)::t8_cmesh_t
    vertices = [  # Just all vertices of all trees. partly duplicated
        -2.0, -2.0, 0,  # tree 0, triangle
        0, -2.0, 0,
        -2.0, 0, 0,
        -2.0, 2.0, 0,   # tree 1, triangle
        -2.0, 0, 0,
        0, 2.0, 0,
        2.0, -2.0, 0,   # tree 2, triangle
        2.0, 0, 0,
        0, -2.0, 0,
        2.0, 2.0, 0,    # tree 3, triangle
        0, 2.0, 0,
        2.0, 0, 0,
        0, -2.0, 0,     # tree 4, quad
        2.0, 0, 0,
        -2.0, 0, 0,
        0, 2.0, 0,
    ]
    #
    # This is how the cmesh looks like. The numbers are the tree numbers:
    # Domain size [-2,2]^2
    #
    # +----------+
    # | 1  /\  3 |
    # |   /  \   |
    # |  /    \  |
    # | /      \ |
    # |/   4    \|
    # |\        /|
    # | \      / |
    # |  \    /  |
    # | 0 \  / 2 |
    # |    \/    |
    # +----------+
    #

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # Use linear geometry
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 2, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 3, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 4, T8_ECLASS_QUAD)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[(1 + 9):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 2, @views(vertices[(1 + 18):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 3, @views(vertices[(1 + 27):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 4, @views(vertices[(1 + 36):end]), 4)

    t8_cmesh_set_join(cmesh, 0, 4, 0, 0, 0)
    t8_cmesh_set_join(cmesh, 0, 2, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 0, 1, 2, 1, 0)

    t8_cmesh_set_join(cmesh, 1, 4, 0, 3, 0)
    t8_cmesh_set_join(cmesh, 1, 3, 2, 1, 0)

    t8_cmesh_set_join(cmesh, 2, 4, 0, 2, 1)
    t8_cmesh_set_join(cmesh, 2, 3, 1, 2, 0)

    t8_cmesh_set_join(cmesh, 3, 4, 0, 1, 1)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end
end # @muladd
