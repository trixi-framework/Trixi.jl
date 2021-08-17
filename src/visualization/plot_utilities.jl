# TODO: move these into StartUpDG for v0.11.0
@inline num_faces(elem::Tri) = 3
@inline num_faces(elem::Quad) = 4


#     plotting_triangulate(u_plot, rst_plot, xyz_plot)
#
# Returns (plotting_coordinates_x, plotting_coordinates_y, ..., plotting_values, plotting_triangulation).
#
# Inputs:
#   - u_plot = matrix of size (Nplot,K) representing solution to plot.
#   - rst_plot = tuple of vector of reference plotting points of length = Nplot
#   - xyz_plot = plotting points (tuple of matrices of size (Nplot,K))
function global_plotting_triangulation_Triplot(u_plot, rst_plot, xyz_plot)

    @assert size(first(xyz_plot), 1) == size(u_plot, 1) "Row dimension of u_plot does not match row dimension of xyz_plot"
    @assert size(first(rst_plot), 1) == size(u_plot, 1) "Row dimension of u_plot does not match row dimension of rst_plot"

    Nplot, K = size(u_plot)

    t = reference_plotting_triangulation(rst_plot)

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

#     plotting_wireframe(rd::RefElemData{2}, md::MeshData{2})
#
# Returns (plotting_coordinates_x, plotting_coordinates_y) for a 2D mesh wireframe.
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

  return x_mesh, y_mesh
end

function PlotData2D(u, mesh::AbstractMeshData, equations, dg::DGMulti, cache;
                    solution_variables=nothing, nvisnodes=2*polydeg(dg))
  rd = dg.basis
  md = mesh.md

  # interpolation matrix from nodal points to plotting points
  @unpack Vp = rd
  interpolate_to_plotting_points!(out, x) = mul!(out, Vp, x)

  solution_variables_ = digest_solution_variables(equations, solution_variables)
  variable_names = SVector(varnames(solution_variables_, equations))

  num_plotting_points = size(Vp, 1)
  nvars = nvariables(equations)
  uEltype = eltype(first(u))
  u_plot_local = StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(uEltype, num_plotting_points), nvars))
  u_plot = zeros(uEltype, num_plotting_points, md.num_elements)
  for e in eachelement(mesh, dg, cache)

    # interpolate solution to plotting nodes element-by-element
    StructArrays.foreachfield(interpolate_to_plotting_points!, u_plot_local, view(u, :, e))

    # transform nodewise solution according to `solution_variables`
    for (i, u_i) in enumerate(u_plot_local)
      u_plot[i, e] = solution_variables_(u_i, equations)[variable_id]
    end
  end

  # interpolate nodal coordinates to
  x_plot, y_plot = map(x->Vp * x, md.xyz)

  # construct a triangulation of the plotting nodes
  t = reference_plotting_triangulation(rd.rstp) # rstp = reference coordinates of plotting points

  xfplot, yfplot = plotting_wireframe(rd, md, num_plotting_points=nvisnodes)

  # Set the plotting values of solution on faces to nothing - they're not used for Plots.jl since
  # only 2D heatmap plots are supported through TriplotBase/TriplotRecipes.
  ufplot = nothing
  return UnstructuredPlotData2D(x_plot, y_plot, u_plot, t, xfplot, yfplot, ufplot, variable_names)
end
