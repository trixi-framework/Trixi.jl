"""
  function compute_triangle_area(tri)

Computes the area of a triangle given a tuple of three points `tri`.
Formula from https://en.wikipedia.org/wiki/Shoelace_formula
"""
function compute_triangle_area(tri)
  A, B, C = tri
  return .5*(A[1] * (B[2] - C[2]) + B[1] * (C[2] - A[2]) + C[1] * (A[2] - B[2]))
end

"""
  function plotting_triangulation(rst_plot)

Computes a triangulation of the points in `rst_plot`, which is a tuple containing vectors of plotting
points on the reference element. Returns a `3 x N_tri` matrix containing a triangulation of the
plotting points, with zero-volume triangles removed.
"""
function plotting_triangulation(rst_plot, tol=100*eps())

  # on-the-fly triangulation of plotting nodes on the reference element
  triin = Triangulate.TriangulateIO()
  triin.pointlist = permutedims(hcat(rst_plot...))
  triout, _ = triangulate("Q", triin)
  t = triout.trianglelist

  # filter out sliver triangles
  has_volume = fill(true,size(t,2))
  for i in axes(t,2)
    ids = @view t[:,i]
    x_points = @view triout.pointlist[1,ids]
    y_points = @view triout.pointlist[2,ids]
    area = compute_triangle_area(zip(x_points, y_points))
    if abs(area) < tol
      has_volume[i] = false
    end
  end
  return t[:, findall(has_volume)]
end

# TODO: move these into StartUpDG for v0.11.0
@inline num_faces(elem::Tri) = 3
@inline num_faces(elem::Quad) = 4

"""
  function plotting_triangulate(u_plot, rst_plot, xyz_plot)

Returns (plotting_coordinates_x, plotting_coordinates_y, ..., plotting_values, plotting_triangulation).

Inputs:
  - u_plot = matrix of size (Nplot,K) representing solution to plot.
  - rst_plot = tuple of vector of reference plotting points of length = Nplot
  - xyz_plot = plotting points (tuple of matrices of size (Nplot,K))
"""

function plotting_triangulation(u_plot, rst_plot, xyz_plot)

    @assert size(first(xyz_plot), 1) == size(u_plot, 1) "Row dimension of u_plot does not match row dimension of xyz_plot"
    @assert size(first(rst_plot), 1) == size(u_plot, 1) "Row dimension of u_plot does not match row dimension of rst_plot"

    Nplot, K = size(u_plot)

    t = plotting_triangulation(rst_plot)

    # build discontinuous data on plotting triangular mesh
    num_ref_elements = size(t, 2)
    num_elements_total = num_ref_elements * K
    tp = zeros(Int, 3, num_elements_total)
    zp = similar(tp, eltype(u_plot))
    for e = 1:K
      for i = 1:size(t, 2)
        tp[:,i + (e-1)*num_ref_elements] .= @views t[:, i] .+ (e-1) * Nplot
        zp[:,i + (e-1)*num_ref_elements] .= @views u_plot[t[:, i], e]
      end
    end
    return vec.(xyz_plot)..., zp, tp
end

"""
  function plotting_wireframe(rd::RefElemData{2}, md::MeshData{2})

Returns (plotting_coordinates_x, plotting_coordinates_y) for a 2D mesh wireframe.
"""
function plotting_wireframe(rd::RefElemData{2}, md::MeshData{2}, num_plotting_points = 25)

  # Construct 1D plotting interpolation matrix `Vp1D` for a single face
  @unpack N, Fmask = rd
  vandermonde_matrix_1D = vandermonde(Line(), N, nodes(Line(), N))
  rplot = LinRange(-1, 1, num_plotting_points)
  Vp1D = vandermonde(Line(), N, rplot) / vandermonde_matrix_1D

  num_face_points = N+1
  num_faces_total = num_faces(rd.elementType) * md.num_elements
  xf, yf = map(x->reshape(view(x, Fmask, :), num_face_points, num_faces_total), md.xyz)

  num_face_plotting_points = size(Vp1D, 1)
  x_mesh, y_mesh = ntuple(_->zeros(num_face_plotting_points, num_faces_total), 2)
  for f in 1:num_faces_total
    mul!(view(x_mesh, :, f), Vp1D, view(xf, :, f))
    mul!(view(y_mesh, :, f), Vp1D, view(yf, :, f))
  end

  separate_with_NaNs(x) = vec(vcat(x, fill(NaN, 1, num_faces_total)))
  x_mesh, y_mesh = separate_with_NaNs.((x_mesh, y_mesh))

  return x_mesh, y_mesh
end