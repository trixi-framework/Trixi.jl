
# Directly ported from: `src/t8_cmesh/t8_cmesh_examples.c: t8_cmesh_new_periodic_hybrid`.
function cmesh_new_periodic_hybrid(comm, n_dims)::t8_cmesh_t
	vertices = [  # Just all vertices of all trees. partly duplicated
		0, 0, 0,                    # tree 0, triangle
		0.5, 0, 0,
		0.5, 0.5, 0,
		0, 0, 0,                    # tree 1, triangle
		0.5, 0.5, 0,
		0, 0.5, 0,
		0.5, 0, 0,                  # tree 2, quad
		1, 0, 0,
		0.5, 0.5, 0,
		1, 0.5, 0,
		0, 0.5, 0,                  # tree 3, quad
		0.5, 0.5, 0,
		0, 1, 0,
		0.5, 1, 0,
		0.5, 0.5, 0,                # tree 4, triangle
		1, 0.5, 0,
		1, 1, 0,
		0.5, 0.5, 0,                # tree 5, triangle
		1, 1, 0,
		0.5, 1, 0,
	]

	# Generally, one can define other geometries. But besides linear the other
	# geometries in t8code do not have C interface yet.
	linear_geom = t8_geometry_linear_new(n_dims)

	#
	# This is how the cmesh looks like. The numbers are the tree numbers:
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

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[1 +  0:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[1 +  9:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 2, @views(vertices[1 + 18:end]), 4)
    t8_cmesh_set_tree_vertices(cmesh, 3, @views(vertices[1 + 30:end]), 4)
    t8_cmesh_set_tree_vertices(cmesh, 4, @views(vertices[1 + 42:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 5, @views(vertices[1 + 51:end]), 3)

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

function cmesh_new_periodic_tri(comm, n_dims)::t8_cmesh_t
	vertices = [ # Just all vertices of all trees. partly duplicated
		0, 0, 0,                    # tree 0, triangle
		1.0, 0, 0,
		1.0, 1.0, 0,
		0, 0, 0,                    # tree 1, triangle
		1.0, 1.0, 0,
		0, 1.0, 0,
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

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[1 +  0:end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[1 +  9:end]), 3)

	t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
	t8_cmesh_set_join(cmesh, 0, 1, 0, 1, 0)
	t8_cmesh_set_join(cmesh, 0, 1, 2, 0, 1)

	t8_cmesh_commit(cmesh, comm)

	return cmesh
end

function cmesh_new_periodic_quad(comm, n_dims)::t8_cmesh_t
	vertices = [ # Just all vertices of all trees. partly duplicated
		0, 0, 0,                    # tree 0, quad
		1.0, 0, 0,
		0, 1.0, 0,
		1.0, 1.0, 0,

        # -2.0, 0.0, 0,                    # tree 0, quad
        # 0.0, -2.0, 0,
        # 0.0, 2.0, 0,
        # 2.0, 0.0, 0,
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
	# t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_QUAD)

	t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[1+0:end]), 4)
	# t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[1 +  12:end]), 4)

	t8_cmesh_set_join(cmesh, 0, 0, 0, 1, 0)
	t8_cmesh_set_join(cmesh, 0, 0, 2, 3, 0)

	t8_cmesh_commit(cmesh, comm)

	return cmesh
end
