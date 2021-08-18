# TODO: move these into StartUpDG for v0.11.0
@inline num_faces(elem::Tri) = 3
@inline num_faces(elem::Quad) = 4

#     compute_triangle_area(tri)
#
# Computes the area of a triangle given `tri`, which is a tuple of three points (vectors),
# using the [Shoelace_formula](https://en.wikipedia.org/wiki/Shoelace_formula).
function compute_triangle_area(tri)
    A, B, C = tri
    return 0.5 * (A[1] * (B[2] - C[2]) + B[1] * (C[2]-A[2]) + C[1] * (A[2] - B[2]))
end

#   reference_plotting_triangulation(reference_plotting_coordinates)
#
# Computes a triangulation of the points in `reference_plotting_coordinates`, which is a tuple containing
# vectors of plotting points on the reference element (e.g., reference_plotting_coordinates = (r,s)).
# The reference element is assumed to be [-1,1]^d.
#
# This function returns `t` which is a `3 x N_tri` Matrix{Int} containing indices of triangles in the
# triangulation of the plotting points, with zero-volume triangles removed.
#
# For example, r[t[1, i]] returns the first reference coordinate of the 1st point on the ith triangle.
function reference_plotting_triangulation(reference_plotting_coordinates, tol=50*eps())
  # on-the-fly triangulation of plotting nodes on the reference element
  tri_in = Triangulate.TriangulateIO()
  tri_in.pointlist = permutedims(hcat(reference_plotting_coordinates...))
  tri_out, _ = Triangulate.triangulate("Q", tri_in)
  triangles = tri_out.trianglelist

  # filter out sliver triangles
  has_volume = fill(true, size(triangles, 2))
  for i in axes(triangles, 2)
      ids = @view triangles[:, i]
      x_points = @view tri_out.pointlist[1, ids]
      y_points = @view tri_out.pointlist[2, ids]
      area = compute_triangle_area(zip(x_points, y_points))
      if abs(area) < tol
          has_volume[i] = false
      end
  end
  return permutedims(triangles[:, findall(has_volume)])
end

function transform_to_solution_variables!(u, solution_variables, equations)
  for (i, u_i) in enumerate(u)
    u[i] = solution_variables(u_i, equations)
  end
end

#     global_plotting_triangulation_Triplot(u_plot, rst_plot, xyz_plot)
#
# Returns (plotting_coordinates_x, plotting_coordinates_y, ..., plotting_values, plotting_triangulation).
# Output can be used with TriplotRecipes.DGTriPseudocolor(...).
#
# Inputs:
#   - xyz_plot = plotting points (tuple of matrices of size (Nplot, K))
#   - u_plot = matrix of size (Nplot, K) representing solution to plot.
#   - t = triangulation of reference plotting points
function global_plotting_triangulation_Triplot(xyz_plot, u_plot, t)

  @assert size(first(xyz_plot), 1) == size(u_plot, 1) "Row dimension of u_plot does not match row dimension of xyz_plot"

  # build discontinuous data on plotting triangular mesh
  num_plotting_points, num_elements = size(u_plot)
  num_reference_plotting_triangles = size(t, 1)
  num_plotting_elements_total = num_reference_plotting_triangles * num_elements

  # each column of `tp` corresponds to a vertex of a plotting triangle
  tp = zeros(Int32, 3, num_plotting_elements_total)
  zp = similar(tp, eltype(u_plot))
  for e = 1:num_elements
    for i = 1:num_reference_plotting_triangles
      tp[:, i + (e-1)*num_reference_plotting_triangles] .= @views t[i, :] .+ (e-1) * num_plotting_points
      zp[:, i + (e-1)*num_reference_plotting_triangles] .= @views u_plot[t[i, :], e]
    end
  end
  return vec.(xyz_plot)..., zp, tp
end

#     mesh_plotting_wireframe(rd::RefElemData{2}, md::MeshData{2})
#
# Generates data for plotting a mesh wireframe given StartUpDG data types.
# Returns (plotting_coordinates_x, plotting_coordinates_y) for a 2D mesh wireframe.
function mesh_plotting_wireframe(rd::RefElemData{2}, md::MeshData{2}; num_plotting_points=25)

  # Construct 1D plotting interpolation matrix `Vp1D` for a single face
  @unpack N, Fmask = rd
  vandermonde_matrix_1D = StartUpDG.vandermonde(Line(), N, StartUpDG.nodes(Line(), N))
  rplot = LinRange(-1, 1, num_plotting_points)
  Vp1D = StartUpDG.vandermonde(Line(), N, rplot) / vandermonde_matrix_1D

  num_face_points = N+1
  num_faces_total = num_faces(rd.elementType) * md.num_elements
  xf, yf = map(x->reshape(view(x, Fmask, :), num_face_points, num_faces_total), md.xyz)

  num_face_plotting_points = size(Vp1D, 1)
  x_mesh, y_mesh = ntuple(_->zeros(num_face_plotting_points, num_faces_total), 2)
  for f in 1:num_faces_total
    mul!(view(x_mesh, :, f), Vp1D, view(xf, :, f))
    mul!(view(y_mesh, :, f), Vp1D, view(yf, :, f))
  end

  return x_mesh, y_mesh
end

