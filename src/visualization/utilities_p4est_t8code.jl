# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#############################################################################
# 2D

# The P4estMesh and T8codeMesh can make use of the efficient search functionality
# based on the tree structure to speed up the process of finding the
# elements in which the curve points are located.
# We hard-code `Float64` here since some parts will be accessed from a Julia
# function called as callback from C. Thus, we cannot use multiple dispatch
# easily and use concrete types instead. See the comment on
# `SearchPointsInP4estT8codeMesh2DHelper` below.
# TODO: Implement the efficient search functionality also for the 2D T8codeMesh.
function unstructured_2d_to_1d_curve(u, mesh::P4estMesh{2, 2, Float64},
                                     equations, dg::DGSEM, cache,
                                     curve, slice,
                                     point, nvisnodes,
                                     solution_variables)
    # TODO: Currently, the efficient search functionality is only implemented
    #       for linear meshes. If the mesh is not linear, we fall back to the naive
    #       but general approach.
    if length(mesh.nodes) != 2
        return unstructured_2d_to_1d_curve_general(u, mesh, equations,
                                                   dg, cache,
                                                   curve, slice,
                                                   point, nvisnodes,
                                                   solution_variables)
    end
    # From here on, we know that we only have to deal with a linear mesh.

    # If no curve is defined, create an axis curve.
    original_nodes = cache.elements.node_coordinates
    if curve === nothing
        curve = axis_curve(view(original_nodes, 1, :, :, :),
                           view(original_nodes, 2, :, :, :),
                           slice, point, nvisnodes)
    end

    # Set up data structure.
    @assert size(curve, 1) == 2
    n_points_curve = size(curve, 2)

    # Get the number of variables after applying the transformation
    # to solution variables.
    u_node = get_node_vars(u, equations, dg, 1, 1, 1)
    var_node = solution_variables(u_node, equations)
    n_variables = length(var_node)
    data_on_curve = Array{eltype(var_node)}(undef, n_points_curve, n_variables)

    # Iterate over every point on the curve and determine the solution value at
    # the given point.
    # We can use the efficient search functionality of p4est/t8code to speed up
    # the process. However, the logic is only implemented for linear meshes so far.

    # Retrieve the element in which each point on the curve is located as well as
    # the local coordinates of the point in the element.
    data = search_points_in_p4est_t8code_mesh_2d(mesh, curve)

    # We use the DG interpolation to get the solution value at the point.
    # Thus, we first setup some data for interpolation.
    nodes = dg.basis.nodes
    baryweights = barycentric_weights(nodes)
    # These Vandermonde matrices are really 1×n_nodes matrices, i.e.,
    # row vectors. We allocate memory here to improve performance.
    vandermonde_x = polynomial_interpolation_matrix(nodes, zero(eltype(curve)))
    vandermonde_y = similar(vandermonde_x)

    n_nodes = length(nodes)
    temp_data = Array{eltype(data_on_curve)}(undef,
                                             n_nodes,
                                             n_variables)
    unstructured_data = Array{eltype(data_on_curve)}(undef,
                                                     n_nodes, n_nodes,
                                                     n_variables)

    for idx_point in eachindex(data)
        query = data[idx_point]
        element = query.index
        @assert query.found
        # The normalization in p4est/t8code is [0, 1] but we need [-1, 1] for DGSEM.
        normalized_coordinates = 2 * SVector(query.x, query.y) .- 1

        # Interpolate to a single point in each element.
        # These Vandermonde matrices are really 1×n_nodes matrices, i.e.,
        # row vectors.
        polynomial_interpolation_matrix!(vandermonde_x, nodes,
                                         normalized_coordinates[1], baryweights)
        polynomial_interpolation_matrix!(vandermonde_y, nodes,
                                         normalized_coordinates[2], baryweights)

        # First, we transform the conserved variables `u` to the solution variables
        # before interpolation.
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            val = solution_variables(u_node, equations)
            for v in eachindex(val)
                unstructured_data[i, j, v] = val[v]
            end
        end

        # Next, we interpolate the solution variables to the located point.
        for v in 1:n_variables
            for i in 1:n_nodes
                res_i = zero(eltype(temp_data))
                for n in 1:n_nodes
                    res_i = res_i + vandermonde_y[n] * unstructured_data[i, n, v]
                end
                temp_data[i, v] = res_i
            end
            res_v = zero(eltype(data_on_curve))
            for n in 1:n_nodes
                res_v = res_v + vandermonde_x[n] * temp_data[n, v]
            end
            data_on_curve[idx_point, v] = res_v
        end
    end

    mesh_vertices_x = nothing
    return calc_arc_length(curve), data_on_curve, mesh_vertices_x
end

# This struct collects a point on the curve and the corresponding element index
# We hard-code `Float64` here since these structs will be accessed from a Julia
# function called as callback from C. Thus, we cannot use multiple dispatch
# easily and use concrete types instead.
struct SearchPointsInP4estT8codeMesh2DHelper
    x::Float64
    y::Float64
    index::Int64
    found::Bool
end

#############################################################################
# Code specialized to the P4estMesh
function search_in_p4est_2d_quadrant_fn(p4est_ptr::Ptr{p4est_t},
                                        which_tree::p4est_topidx_t,
                                        quadrant_ptr::Ptr{p4est_quadrant_t},
                                        local_num::p4est_locidx_t,
                                        null_pointer::Ptr{Cvoid})::Cint

    # Continue the search
    return Cint(1)
end

function search_in_p4est_2d_point_fn(p4est_ptr::Ptr{p4est_t},
                                     which_tree::p4est_topidx_t,
                                     quadrant_ptr::Ptr{p4est_quadrant_t},
                                     local_num::p4est_locidx_t,
                                     query_ptr::Ptr{Cvoid})::Cint
    # Currently, we do not store the `mapping` but only the resulting
    # `tree_node_coordinates` in the mesh. Thus, we need to use them
    # to convert the reference coordinates of the `element` to physical
    # coordinates (assuming polydeg == 1) and then check whether the
    # coordinates of the points are inside the `element`.

    # Get the references coordinates of the element.
    # Note: This assumes that we are in 2D.
    quadrant = PointerWrapper(quadrant_ptr)

    level = quadrant.level[]
    p4est_root_length = 1 << P4EST_MAXLEVEL
    p4est_quad_length = 1 << (P4EST_MAXLEVEL - level)

    x0 = quadrant.x[] / p4est_root_length
    y0 = quadrant.y[] / p4est_root_length

    x1 = x0 + p4est_quad_length / p4est_root_length
    y1 = y0 + p4est_quad_length / p4est_root_length

    r00 = SVector(x0, y0)
    r10 = SVector(x1, y0)
    r01 = SVector(x0, y1)

    # Get the bounding physical coordinates of the tree and use them to
    # compute the affine transformation.
    # Note: This assumes additionally that the polynomial degree is 1.
    p4est = PointerWrapper(p4est_ptr)
    user_data = Ptr{Float64}(pointer(p4est.user_pointer))
    number_of_trees = p4est.connectivity.num_trees[]
    tree_node_coordinates = PtrArray(user_data,
                                     (2, 2, 2, number_of_trees))
    a0, A = get_affine_transformation(r00, r10, r01,
                                      tree_node_coordinates,
                                      which_tree)

    # Load the query data
    query = unsafe_load(Ptr{SearchPointsInP4estT8codeMesh2DHelper}(query_ptr))

    # Do nothing if the point has already been found elsewhere.
    if query.found
        return Cint(0)
    end

    # If the point has not already been found, we check whether it is inside
    # the parallelogram defined by the element.
    point = SVector(query.x, query.y)

    # Compute coefficients to express the `point` in the basis of the
    # parallelogram.
    coefficients = A \ (point - a0)

    # Check if the point is inside the parallelogram.
    tolerance = 1.0e-13
    is_inside = -tolerance <= coefficients[1] <= 1 + tolerance &&
                -tolerance <= coefficients[2] <= 1 + tolerance

    if is_inside
        if local_num >= 0
            # If we are in a valid element (leaf of the tree), we store
            # the element id and the coefficients of the point in the
            # query data structure.
            index = local_num + 1
            new_query = SearchPointsInP4estT8codeMesh2DHelper(coefficients[1],
                                                              coefficients[2],
                                                              index,
                                                              true)
            unsafe_store!(Ptr{SearchPointsInP4estT8codeMesh2DHelper}(query_ptr),
                          new_query)
        end

        return Cint(1)
    else
        return Cint(0)
    end
end

@inline function get_affine_transformation(r00, r10, r01,
                                           tree_node_coordinates,
                                           which_tree)
    # Get the bounding physical coordinates of the tree
    t00 = SVector(tree_node_coordinates[1, 1, 1, which_tree + 1],
                  tree_node_coordinates[2, 1, 1, which_tree + 1])
    t10 = SVector(tree_node_coordinates[1, 2, 1, which_tree + 1],
                  tree_node_coordinates[2, 2, 1, which_tree + 1])
    t01 = SVector(tree_node_coordinates[1, 1, 2, which_tree + 1],
                  tree_node_coordinates[2, 1, 2, which_tree + 1])

    # Transform the reference coordinates to physical coordinates.
    # Note: This requires the same assumptions as above (linear hex mesh in 2D)
    p00 = t00 +
          r00[1] * (t10 - t00) +
          r00[2] * (t01 - t00)
    p10 = t00 +
          r10[1] * (t10 - t00) +
          r10[2] * (t01 - t00)
    p01 = t00 +
          r01[1] * (t10 - t00) +
          r01[2] * (t01 - t00)

    # Get the base point a0 and the basis vectors a1, a2 spanning the
    # parallelogram in physical coordinates.
    # Note: This requires the same assumptions as above.
    a0 = p00
    a1 = p10 - p00
    a2 = p01 - p00

    # Get the transformation matrix A to compute
    # the coefficients of the point in the basis of the parallelogram.
    A = SMatrix{2, 2}(a1[1], a2[1],
                      a1[2], a2[2])

    return a0, A
end

function search_points_in_p4est_t8code_mesh_2d(mesh::P4estMesh,
                                               curve::Array{Float64, 2})
    quadrant_fn = @cfunction(search_in_p4est_2d_quadrant_fn,
                             Cint,
                             (Ptr{p4est_t}, p4est_topidx_t,
                              Ptr{p4est_quadrant_t}, p4est_locidx_t,
                              Ptr{Cvoid}))
    point_fn = @cfunction(search_in_p4est_2d_point_fn,
                          Cint,
                          (Ptr{p4est_t}, p4est_topidx_t,
                           Ptr{p4est_quadrant_t}, p4est_locidx_t,
                           Ptr{Cvoid}))

    data = Vector{SearchPointsInP4estT8codeMesh2DHelper}(undef, size(curve, 2))
    for i in 1:size(curve, 2)
        data[i] = SearchPointsInP4estT8codeMesh2DHelper(curve[1, i],
                                                        curve[2, i],
                                                        typemin(Int64),
                                                        false)
    end
    queries = sc_array_new_data(pointer(data),
                                sizeof(eltype(data)),
                                length(data))

    call_post = 0
    mesh.p4est.user_pointer = pointer(mesh.tree_node_coordinates)
    p4est_search_local(pointer(mesh.p4est), call_post, quadrant_fn, point_fn,
                       queries)

    return data
end

#############################################################################
# 3D

# The P4estMesh and T8codeMesh can make use of the efficient search functionality
# based on the tree structure to speed up the process of finding the
# elements in which the curve points are located.
# We hard-code `Float64` here since some parts will be accessed from a Julia
# function called as callback from C. Thus, we cannot use multiple dispatch
# easily and use concrete types instead. See the comment on
# `SearchPointsInP4estT8codeMesh3DHelper` below.
function unstructured_3d_to_1d_curve(u,
                                     mesh::Union{P4estMesh{3, 3, Float64},
                                                 T8codeMesh{3, Float64}},
                                     equations, dg::DGSEM, cache,
                                     curve, solution_variables)
    # TODO: Currently, the efficient search functionality is only implemented
    #       for linear meshes. If the mesh is not linear, we fall back to the naive
    #       but general approach.
    if length(mesh.nodes) != 2
        return unstructured_3d_to_1d_curve_general(u, equations, solver, cache,
                                                   curve, solution_variables)
    end
    # From here on, we know that we only have to deal with a linear mesh.

    # Set up data structure.
    @assert size(curve, 1) == 3
    n_points_curve = size(curve, 2)

    # Get the number of variables after applying the transformation to solution variables.
    u_node = get_node_vars(u, equations, dg, 1, 1, 1, 1)
    var_node = solution_variables(u_node, equations)
    n_variables = length(var_node)
    data_on_curve = Array{eltype(var_node)}(undef, n_points_curve, n_variables)

    # Iterate over every point on the curve and determine the solution value at
    # the given point.
    # We can use the efficient search functionality of p4est/t8code to speed up
    # the process. However, the logic is only implemented for linear meshes so far.

    # Retrieve the element in which each point on the curve is located as well as
    # the local coordinates of the point in the element.
    data = search_points_in_p4est_t8code_mesh_3d(mesh, curve)

    # We use the DG interpolation to get the solution value at the point.
    # Thus, we first setup some data for interpolation.
    nodes = dg.basis.nodes
    baryweights = barycentric_weights(nodes)
    # These Vandermonde matrices are really 1×n_nodes matrices, i.e.,
    # row vectors. We allocate memory here to improve performance.
    vandermonde_x = polynomial_interpolation_matrix(nodes, zero(eltype(curve)))
    vandermonde_y = similar(vandermonde_x)
    vandermonde_z = similar(vandermonde_x)

    n_nodes = length(nodes)
    temp_data = Array{eltype(data_on_curve)}(undef,
                                             n_nodes, n_nodes + 1,
                                             n_variables)
    unstructured_data = Array{eltype(data_on_curve)}(undef,
                                                     n_nodes, n_nodes, n_nodes,
                                                     n_variables)

    for idx_point in eachindex(data)
        query = data[idx_point]
        element = query.index
        @assert query.found
        # The normalization in p4est/t8code is [0, 1] but we need [-1, 1] for DGSEM.
        normalized_coordinates = 2 * SVector(query.x, query.y, query.z) .- 1

        # Interpolate to a single point in each element.
        # These Vandermonde matrices are really 1×n_nodes matrices, i.e.,
        # row vectors.
        polynomial_interpolation_matrix!(vandermonde_x, nodes,
                                         normalized_coordinates[1], baryweights)
        polynomial_interpolation_matrix!(vandermonde_y, nodes,
                                         normalized_coordinates[2], baryweights)
        polynomial_interpolation_matrix!(vandermonde_z, nodes,
                                         normalized_coordinates[3], baryweights)

        # First, we transform the conserved variables `u` to the solution variables
        # before interpolation.
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            val = solution_variables(u_node, equations)
            for v in eachindex(val)
                unstructured_data[i, j, k, v] = val[v]
            end
        end

        # Next, we interpolate the solution variables to the located point.
        for v in 1:n_variables
            for i in 1:n_nodes
                for ii in 1:n_nodes
                    res_ii = zero(eltype(temp_data))
                    for n in 1:n_nodes
                        res_ii = res_ii +
                                 vandermonde_z[n] *
                                 unstructured_data[i, ii, n, v]
                    end
                    temp_data[i, ii, v] = res_ii
                end
                res_i = zero(eltype(temp_data))
                for n in 1:n_nodes
                    res_i = res_i + vandermonde_y[n] * temp_data[i, n, v]
                end
                temp_data[i, n_nodes + 1, v] = res_i
            end
            res_v = zero(eltype(temp_data))
            for n in 1:n_nodes
                res_v = res_v + vandermonde_x[n] * temp_data[n, n_nodes + 1, v]
            end
            data_on_curve[idx_point, v] = res_v
        end
    end

    mesh_vertices_x = nothing
    return calc_arc_length(curve), data_on_curve, mesh_vertices_x
end

# This struct collects a point on the curve and the corresponding element index
# We hard-code `Float64` here since these structs will be accessed from a Julia
# function called as callback from C. Thus, we cannot use multiple dispatch
# easily and use concrete types instead.
struct SearchPointsInP4estT8codeMesh3DHelper
    x::Float64
    y::Float64
    z::Float64
    index::Int64
    found::Bool
end

#############################################################################
# Code specialized to the P4estMesh
function search_in_p4est_3d_quadrant_fn(p4est_ptr::Ptr{p8est_t},
                                        which_tree::p4est_topidx_t,
                                        quadrant_ptr::Ptr{p8est_quadrant_t},
                                        local_num::p4est_locidx_t,
                                        null_pointer::Ptr{Cvoid})::Cint

    # Continue the search
    return Cint(1)
end

function search_in_p4est_3d_point_fn(p4est_ptr::Ptr{p8est_t},
                                     which_tree::p4est_topidx_t,
                                     quadrant_ptr::Ptr{p8est_quadrant_t},
                                     local_num::p4est_locidx_t,
                                     query_ptr::Ptr{Cvoid})::Cint
    # Currently, we do not store the `mapping` but only the resulting
    # `tree_node_coordinates` in the mesh. Thus, we need to use them
    # to convert the reference coordinates of the `element` to physical
    # coordinates (assuming polydeg == 1) and then check whether the
    # coordinates of the points are inside the `element`.

    # Get the references coordinates of the element.
    # Note: This assumes that we are in 3D.
    quadrant = PointerWrapper(quadrant_ptr)

    level = quadrant.level[]
    p4est_root_length = 1 << P4EST_MAXLEVEL
    p4est_quad_length = 1 << (P4EST_MAXLEVEL - level)

    x0 = quadrant.x[] / p4est_root_length
    y0 = quadrant.y[] / p4est_root_length
    z0 = quadrant.z[] / p4est_root_length

    x1 = x0 + p4est_quad_length / p4est_root_length
    y1 = y0 + p4est_quad_length / p4est_root_length
    z1 = z0 + p4est_quad_length / p4est_root_length

    r000 = SVector(x0, y0, z0)
    r100 = SVector(x1, y0, z0)
    r010 = SVector(x0, y1, z0)
    r001 = SVector(x0, y0, z1)

    # Get the bounding physical coordinates of the tree and use them to
    # compute the affine transformation.
    # Note: This assumes additionally that the polynomial degree is 1.
    p4est = PointerWrapper(p4est_ptr)
    user_data = Ptr{Ptr{Float64}}(pointer(p4est.user_pointer))
    number_of_trees = p4est.connectivity.num_trees[]
    tree_node_coordinates = PtrArray(unsafe_load(user_data, 1),
                                     (3, 2, 2, 2, number_of_trees))
    a0, A = get_affine_transformation(r000, r100, r010, r001,
                                      tree_node_coordinates,
                                      which_tree)

    # Load the query data
    query = unsafe_load(Ptr{SearchPointsInP4estT8codeMesh3DHelper}(query_ptr))

    # Do nothing if the point has already been found elsewhere.
    if query.found
        return Cint(0)
    end

    # If the point has not already been found, we check whether it is inside
    # the parallelepiped defined by the element.
    point = SVector(query.x, query.y, query.z)

    # Compute coefficients to express the `point` in the basis of the
    # parallelepiped.
    coefficients = A \ (point - a0)

    # Check if the point is inside the parallelepiped.
    tolerance = 1.0e-13
    is_inside = -tolerance <= coefficients[1] <= 1 + tolerance &&
                -tolerance <= coefficients[2] <= 1 + tolerance &&
                -tolerance <= coefficients[3] <= 1 + tolerance

    if is_inside
        if local_num >= 0
            # If we are in a valid element (leaf of the tree), we store
            # the element id and the coefficients of the point in the
            # query data structure.
            index = local_num + 1
            new_query = SearchPointsInP4estT8codeMesh3DHelper(coefficients[1],
                                                              coefficients[2],
                                                              coefficients[3],
                                                              index,
                                                              true)
            unsafe_store!(Ptr{SearchPointsInP4estT8codeMesh3DHelper}(query_ptr),
                          new_query)
        end

        return Cint(1)
    else
        return Cint(0)
    end
end

@inline function get_affine_transformation(r000, r100, r010, r001,
                                           tree_node_coordinates,
                                           which_tree)
    # Get the bounding physical coordinates of the tree
    t000 = SVector(tree_node_coordinates[1, 1, 1, 1, which_tree + 1],
                   tree_node_coordinates[2, 1, 1, 1, which_tree + 1],
                   tree_node_coordinates[3, 1, 1, 1, which_tree + 1])
    t100 = SVector(tree_node_coordinates[1, 2, 1, 1, which_tree + 1],
                   tree_node_coordinates[2, 2, 1, 1, which_tree + 1],
                   tree_node_coordinates[3, 2, 1, 1, which_tree + 1])
    t010 = SVector(tree_node_coordinates[1, 1, 2, 1, which_tree + 1],
                   tree_node_coordinates[2, 1, 2, 1, which_tree + 1],
                   tree_node_coordinates[3, 1, 2, 1, which_tree + 1])
    t001 = SVector(tree_node_coordinates[1, 1, 1, 2, which_tree + 1],
                   tree_node_coordinates[2, 1, 1, 2, which_tree + 1],
                   tree_node_coordinates[3, 1, 1, 2, which_tree + 1])

    # Transform the reference coordinates to physical coordinates.
    # Note: This requires the same assumptions as above (linear hex mesh in 3D)
    p000 = t000 +
           r000[1] * (t100 - t000) +
           r000[2] * (t010 - t000) +
           r000[3] * (t001 - t000)
    p100 = t000 +
           r100[1] * (t100 - t000) +
           r100[2] * (t010 - t000) +
           r100[3] * (t001 - t000)
    p010 = t000 +
           r010[1] * (t100 - t000) +
           r010[2] * (t010 - t000) +
           r010[3] * (t001 - t000)
    p001 = t000 +
           r001[1] * (t100 - t000) +
           r001[2] * (t010 - t000) +
           r001[3] * (t001 - t000)

    # Get the base point a0 and the basis vectors a1, a2, a3 spanning the
    # parallelepiped in physical coordinates.
    # Note: This requires the same assumptions as above.
    a0 = p000
    a1 = p100 - p000
    a2 = p010 - p000
    a3 = p001 - p000

    # Get the transformation matrix A to compute
    # the coefficients of the point in the basis of the parallelepiped.
    A = SMatrix{3, 3}(a1[1], a2[1], a3[1],
                      a1[2], a2[2], a3[2],
                      a1[3], a2[3], a3[3])

    return a0, A
end

function search_points_in_p4est_t8code_mesh_3d(mesh::P4estMesh,
                                               curve::Array{Float64, 2})
    quadrant_fn = @cfunction(search_in_p4est_3d_quadrant_fn,
                             Cint,
                             (Ptr{p8est_t}, p4est_topidx_t,
                              Ptr{p8est_quadrant_t}, p4est_locidx_t,
                              Ptr{Cvoid}))
    point_fn = @cfunction(search_in_p4est_3d_point_fn,
                          Cint,
                          (Ptr{p8est_t}, p4est_topidx_t,
                           Ptr{p8est_quadrant_t}, p4est_locidx_t,
                           Ptr{Cvoid}))

    data = Vector{SearchPointsInP4estT8codeMesh3DHelper}(undef, size(curve, 2))
    for i in 1:size(curve, 2)
        data[i] = SearchPointsInP4estT8codeMesh3DHelper(curve[1, i],
                                                        curve[2, i],
                                                        curve[3, i],
                                                        typemin(Int64),
                                                        false)
    end
    queries = sc_array_new_data(pointer(data),
                                sizeof(eltype(data)),
                                length(data))

    temp_vertex = zeros(Float64, 3)
    GC.@preserve temp_vertex begin
        user_data = [pointer(mesh.tree_node_coordinates),
            pointer(temp_vertex)]

        GC.@preserve user_data begin
            call_post = 0
            mesh.p4est.user_pointer = pointer(user_data)
            p8est_search_local(pointer(mesh.p4est), call_post, quadrant_fn, point_fn,
                               queries)
        end
    end

    return data
end

#############################################################################
# Code specialized to the T8codeMesh

function search_points_in_t8code_mesh_3d_callback_element(forest::t8_forest_t,
                                                          ltreeid::t8_locidx_t,
                                                          element::Ptr{t8_element_t},
                                                          is_leaf::Cint,
                                                          leaf_elements::Ptr{t8_element_array_t},
                                                          tree_leaf_index::t8_locidx_t)::Cint
    # Continue the search
    return Cint(1)
end

function search_points_in_t8code_mesh_3d_callback_query(forest::t8_forest_t,
                                                        ltreeid::t8_locidx_t,
                                                        element::Ptr{t8_element_t},
                                                        is_leaf::Cint,
                                                        leaf_elements::Ptr{t8_element_array_t},
                                                        tree_leaf_index::t8_locidx_t,
                                                        queries_ptr::Ptr{sc_array_t},
                                                        query_indices_ptr::Ptr{sc_array_t},
                                                        query_matches::Ptr{Cint},
                                                        num_active_queries::Csize_t)::Cvoid
    # It would be nice if we could just use
    #   t8_forest_element_points_inside(forest, ltreeid, element, coords,
    #                                   num_active_queries, query_matches, tolerance)
    # However, this does not work since the coordinates of the points
    # on the curve are given in physical coordinates but t8code only
    # knows reference coordinates (in [0, 1]).
    # Currently, we do not store the `mapping` but only the resulting
    # `tree_node_coordinates` in the mesh. Thus, we need to use them
    # to convert the reference coordinates of the `element` to physical
    # coordinates (assuming polydeg == 1) and then check whether the
    # coordinates of the points are inside the `element`.

    # Get the references coordinates of the element.
    # Note: This assumes that we are in 3D and that the element is a hexahedron.
    user_data = Ptr{Ptr{Float64}}(t8_forest_get_user_data(forest))
    vertex = PtrArray(unsafe_load(user_data, 2), (3,))
    eclass_scheme = t8_forest_get_eclass_scheme(forest, T8_ECLASS_HEX)
    t8_element_vertex_reference_coords(eclass_scheme, element, 0, vertex)
    r000 = SVector(vertex[1], vertex[2], vertex[3])
    t8_element_vertex_reference_coords(eclass_scheme, element, 1, vertex)
    r100 = SVector(vertex[1], vertex[2], vertex[3])
    t8_element_vertex_reference_coords(eclass_scheme, element, 2, vertex)
    r010 = SVector(vertex[1], vertex[2], vertex[3])
    t8_element_vertex_reference_coords(eclass_scheme, element, 4, vertex)
    r001 = SVector(vertex[1], vertex[2], vertex[3])

    # Get the bounding physical coordinates of the tree.
    # Note: This assumes additionally that the polynomial degree is 1.
    number_of_trees = t8_forest_get_num_global_trees(forest)
    tree_node_coordinates = PtrArray(unsafe_load(user_data, 1),
                                     (3, 2, 2, 2, number_of_trees))
    a0, A = get_affine_transformation(r000, r100, r010, r001,
                                      tree_node_coordinates,
                                      ltreeid)
    invA = inv(A)

    # Loop over all points that need to be found
    queries = PointerWrapper(queries_ptr)
    query_indices = PointerWrapper(query_indices_ptr)
    for i in 1:num_active_queries
        # t8code uses 0-based indexing, we use 1-based ondexing in Julia.
        query_index = unsafe_load_sc(Csize_t, query_indices, i) + 1
        query = unsafe_load_sc(SearchPointsInP4estT8codeMesh3DHelper, queries,
                               query_index)

        # Do nothing if the point has already been found elsewhere.
        if query.found
            unsafe_store!(query_matches, 1, i)
            continue
        end

        # If the point has not already been found, we check whether it is inside
        # the parallelepiped defined by the element.
        point = SVector(query.x, query.y, query.z)

        # Compute coefficients to express the `point` in the basis of the
        # parallelepiped.
        coefficients = invA * (point - a0)

        # Check if the point is inside the parallelepiped.
        tolerance = 1.0e-13
        is_inside = -tolerance <= coefficients[1] <= 1 + tolerance &&
                    -tolerance <= coefficients[2] <= 1 + tolerance &&
                    -tolerance <= coefficients[3] <= 1 + tolerance

        if is_inside
            unsafe_store!(query_matches, 1, i)

            if is_leaf == 1
                # If we are in a valid element (leaf of the tree), we store
                # the element id and the coefficients of the point in the
                # query data structure.
                index = t8_forest_get_tree_element_offset(forest, ltreeid) +
                        tree_leaf_index + 1
                new_query = SearchPointsInP4estT8codeMesh3DHelper(coefficients[1],
                                                                  coefficients[2],
                                                                  coefficients[3],
                                                                  index,
                                                                  true)
                unsafe_store_sc!(queries, new_query, query_index)
            end
        else
            unsafe_store!(query_matches, 0, i)
        end
    end

    return nothing
end

function search_points_in_p4est_t8code_mesh_3d(mesh::T8codeMesh,
                                               curve::Array{Float64, 2})
    element_fn = @cfunction(search_points_in_t8code_mesh_3d_callback_element,
                            Cint,
                            (t8_forest_t, t8_locidx_t, Ptr{t8_element_t},
                             Cint, Ptr{t8_element_array_t}, t8_locidx_t))
    query_fn = @cfunction(search_points_in_t8code_mesh_3d_callback_query,
                          Cvoid,
                          (t8_forest_t, t8_locidx_t, Ptr{t8_element_t},
                           Cint, Ptr{t8_element_array_t}, t8_locidx_t,
                           Ptr{sc_array_t}, Ptr{sc_array_t},
                           Ptr{Cint}, Csize_t))

    data = Vector{SearchPointsInP4estT8codeMesh3DHelper}(undef, size(curve, 2))
    for i in eachindex(data)
        data[i] = SearchPointsInP4estT8codeMesh3DHelper(curve[1, i],
                                                        curve[2, i],
                                                        curve[3, i],
                                                        typemin(Int64),
                                                        false)
    end
    queries = sc_array_new_data(pointer(data),
                                sizeof(eltype(data)),
                                length(data))

    temp_vertex = zeros(Float64, 3)
    GC.@preserve temp_vertex begin
        user_data = [pointer(mesh.tree_node_coordinates),
            pointer(temp_vertex)]

        GC.@preserve user_data begin
            t8_forest_set_user_data(pointer(mesh.forest), pointer(user_data))
            t8_forest_search(pointer(mesh.forest), element_fn, query_fn, queries)
        end
    end

    return data
end
end # @muladd
