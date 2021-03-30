
using Trixi
using Plots

include("curve_interpolant.jl")
include("quadrilateral_mappings.jl")
#include("element_geometry.jl")
include("containers_2d.jl")
include("unstructured_quad_mesh.jl")

poly_deg = 7
nnodes   = poly_deg + 1

# cheby_nodes, _ = chebyshev_gauss_lobatto_nodes_weights(nnodes)

##
#  test out the interpolation and interpolation derivative routines

# bary_weights = Trixi.barycentric_weights(cheby_nodes)
#
# effs = zeros(nnodes)
#
# for j in 1:nnodes
#   effs[j] = sinpi(cheby_nodes[j]) #cheby_nodes.^3 + cheby_nodes
# end
#
# exes = range(-1,stop=1,length=50)
#
# resu_f  = zeros(50)
# resu_fp = zeros(50)
#
# for j in 1:50
#     resu_f[j]  = lagrange_interpolation(exes[j], cheby_nodes, effs, bary_weights)
#     resu_fp[j] = lagrange_interpolation_derivative(exes[j], cheby_nodes, effs, bary_weights)
# end
#
# plot(exes,resu_f)
# plot!(exes,resu_fp)

##
# test out the Gamma curves construction on a single element

# GammaCurves = Array{GammaCurve, 1}(undef, 4)
#
# x_vals = zeros(nnodes)
# y_vals = zeros(nnodes)
#
# # side 1
# for j in 1:nnodes
#    x_vals[j] = 2.0 + cheby_nodes[j]
#    y_vals[j] = 0.0
# end
# GammaCurves[1] = GammaCurve(Float64, poly_deg, x_vals, y_vals)
#
# # side 2
# for j in 1:nnodes
#    x_vals[j] = 3.0 * 0.5 * (1.0 - cheby_nodes[j])
#    y_vals[j] = 3.0 * 0.5 * (1.0 + cheby_nodes[j])
# end
# GammaCurves[2] = GammaCurve(Float64, poly_deg, x_vals, y_vals)
#
# # side 3
# for j in 1:nnodes
#    x_vals[j] = 0.0
#    y_vals[j] = 2.0 + cheby_nodes[j]
# end
# GammaCurves[3] = GammaCurve(Float64, poly_deg, x_vals, y_vals)
#
# # side 4
# for j in 1:nnodes
#    x_vals[j] = cospi(0.25*(1.0 + cheby_nodes[j]))
#    y_vals[j] = sinpi(0.25*(1.0 + cheby_nodes[j]))
# end
# GammaCurves[4] = GammaCurve(Float64, poly_deg, x_vals, y_vals)
#
#  # x_grid = zeros(nnodes,nnodes)
#  # y_grid = zeros(nnodes,nnodes)
#
# corners = zeros(4,2)
#
# corners[1,1] = 1.0
# corners[1,2] = 0.0
#
# corners[2,1] = 3.0
# corners[2,2] = 0.0
#
# corners[3,1] = 0.0
# corners[3,2] = 3.0
#
# corners[4,1] = 0.0
# corners[4,2] = 1.0

# for j in 1:nnodes
#   for i in 1:nnodes
#       x_grid[i,j], y_grid[i,j] = transfinite_quad_map(cheby_nodes[i], cheby_nodes[j], GammaCurves)
# #      x_grid[i,j], y_grid[i,j] = straight_side_quad_map(corners, cheby_nodes[i], cheby_nodes[j])
#   end
# end

#plot(           x_grid ,          y_grid , linecolor=:black, legend = false, aspect_ratio=:equal)
#plot!(transpose(x_grid),transpose(y_grid), linecolor=:black, legend = false, aspect_ratio=:equal)


##
#  test out the geometry routine on a single element

lgl_nodes, _ = Trixi.gauss_lobatto_nodes_weights(nnodes)
# test the general version
#geometry_test = ElementGeometry(Float64, poly_deg, lgl_nodes, corners, GammaCurves, true)

# test the curved only version
# geometry_test = ElementGeometry(Float64, poly_deg, lgl_nodes, GammaCurves)

# test the straight sided version
#geometry_test = ElementGeometry(Float64, poly_deg, lgl_nodes, corners)


#plot(           geometry_test.x ,          geometry_test.y , linecolor=:black, legend = false, aspect_ratio=:equal)
#plot!(transpose(geometry_test.x),transpose(geometry_test.y), linecolor=:black, legend = false, aspect_ratio=:equal)

#println(geometry_test.n_vecs[:,4,:])

# string_test = Vector{String}(undef,4)
# string_test[1] = "---"
# string_test[2] = "---"
# string_test[3] = "---"
# string_test[4] = "Circle"
#
# # curved element container
# #container_test = GeneralElementContainer2D(Float64, poly_deg, lgl_nodes, GammaCurves, string_test)
#
# # curved element container
# container_test = GeneralElementContainer2D(Float64, poly_deg, lgl_nodes, corners, string_test)
#
#
# plot(           container_test.geometry.x ,          container_test.geometry.y , linecolor=:black, legend = false, aspect_ratio=:equal)
# plot!(transpose(container_test.geometry.x),transpose(container_test.geometry.y), linecolor=:black, legend = false, aspect_ratio=:equal)
#
# edge_info = Vector{Int64}(undef,6)
# edge_info[1] = 23
# edge_info[2] = 120
# edge_info[3] = 56
# edge_info[4] = 24
# edge_info[5] = 2
# edge_info[6] = -4
#
# edge_test = GeneralInterfaceContainer2D(edge_info, poly_deg)

##
#  test the creation of a quad mesh from a file
#mesh_file = "BoxAroundCircle8.mesh"
mesh_file = "PeriodicXandY7.mesh"
mesh_test = UnstructuredQuadMesh(Float64, mesh_file, poly_deg, lgl_nodes)

# plot the first element

#for j in 2:40
for j in 2:16
   plot!(          mesh_test.elements[j].geometry.x ,          mesh_test.elements[j].geometry.y , linecolor=:black, legend = false, aspect_ratio=:equal)
   plot!(transpose(mesh_test.elements[j].geometry.x),transpose(mesh_test.elements[j].geometry.y), linecolor=:black, legend = false, aspect_ratio=:equal)
end

plot!(           mesh_test.elements[1].geometry.x ,          mesh_test.elements[1].geometry.y , linecolor=:black, legend = false, aspect_ratio=:equal)
plot!(transpose(mesh_test.elements[1].geometry.x),transpose(mesh_test.elements[1].geometry.y), linecolor=:black, legend = false, aspect_ratio=:equal)

# plot the others
