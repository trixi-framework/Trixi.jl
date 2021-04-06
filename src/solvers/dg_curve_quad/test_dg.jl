
using Trixi
#using Plots
using PyPlot

include("curve_interpolant.jl")
include("quadrilateral_mappings.jl")
include("containers_2d.jl")
include("unstructured_quad_mesh.jl")
include("dg_curve_2d.jl")

###############################################################################
# Get the equations

equations = CompressibleEulerEquations2D(1.4)
num_eqns = 4

initial_condition = initial_condition_convergence_test
source_term = source_terms_convergence_test
boundary_conditions = boundary_condition_periodic # this is unused currently

###############################################################################
# Get the DG approximation space

poly_deg = 7
surface_flux = flux_lax_friedrichs
solver = DGSEM(poly_deg, surface_flux)

###############################################################################
# Get the curved quad mesh from a file

#mesh_file = "BoxAroundCircle8.mesh"
mesh_file = "PeriodicXandY7.mesh"
mesh      = UnstructuredQuadMesh(Float64, mesh_file, num_eqns, poly_deg, solver.basis.nodes)

# #for j in 2:40
# for j in 2:16
#    plot!(          mesh.elements[j].geometry.x ,          mesh.elements[j].geometry.y , linecolor=:black, legend = false, aspect_ratio=:equal)
#    plot!(transpose(mesh.elements[j].geometry.x),transpose(mesh.elements[j].geometry.y), linecolor=:black, legend = false, aspect_ratio=:equal)
# end
# plot!(          mesh.elements[1].geometry.x ,          mesh.elements[1].geometry.y , linecolor=:black, legend = false, aspect_ratio=:equal)
# plot!(transpose(mesh.elements[1].geometry.x),transpose(mesh.elements[1].geometry.y), linecolor=:black, legend = false, aspect_ratio=:equal)

###############################################################################
# test out creating the cache

cache = create_cache(mesh, equations, solver, Float64)

###############################################################################
# Construct and fill the initial condition

u = zeros( num_eqns , nnodes(solver) , nnodes(solver) , nelements(mesh) )

for eID in eachelement(mesh)
  for j in eachnode(solver)
    for i in eachnode(solver)
      x_vec = ( mesh.elements[eID].geometry.x[i,j] , mesh.elements[eID].geometry.y[i,j])
      u[:,i,j,eID] = initial_condition(x_vec, 0.0, equations)
    end
  end
end

###############################################################################
# test out the right-hand-side computation
du = zeros( num_eqns , nnodes(solver) , nnodes(solver) , nelements(mesh) )

rhs!(du, u, 0.0, mesh, equations, initial_condition, boundary_conditions, source_term, solver, cache)


###############################################################################
# move everything into appropriate arrays for plotting
# all_x = zeros( nnodes(solver) , nnodes(solver) , nelements(mesh) )
# all_y = zeros( nnodes(solver) , nnodes(solver) , nelements(mesh) )
# solu  = zeros( nnodes(solver) , nnodes(solver) , nelements(mesh) )
# for eID in eachelement(mesh)
#   all_x[:,:,eID] = mesh.elements[eID].geometry.x
#   all_y[:,:,eID] = mesh.elements[eID].geometry.y
#   solu[:,:,eID]  = u0[1,:,:,eID]
# end
#
# for eID in eachelement(mesh)
#   #plot_surface( all_x[:,:,eID] , all_y[:,:,eID] , solu[:,:,eID])
#   surf( all_x[:,:,eID] , all_y[:,:,eID] , solu[:,:,eID] , cmap=ColorMap("plasma"))
# end
