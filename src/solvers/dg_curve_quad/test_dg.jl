
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
num_eqns = Trixi.nvariables(equations)

initial_condition = initial_condition_convergence_test
source_term = source_terms_convergence_test
boundary_conditions = boundary_condition_periodic
#boundary_conditions = boundary_condition_convergence_test

###############################################################################
# Get the DG approximation space

poly_deg = 3
surface_flux = flux_hll # flux_lax_friedrichs
solver = DGSEM(poly_deg, surface_flux)

###############################################################################
# Get the curved quad mesh from a file

#mesh_file = "BoxAroundCircle8.mesh"
mesh_file = "PeriodicXandY10.mesh"
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

u0 = zeros( num_eqns , nnodes(solver) , nnodes(solver) , nelements(mesh) )

for eID in eachelement(mesh)
  for j in eachnode(solver), i in eachnode(solver)
    x_vec = ( mesh.elements[eID].geometry.x[i,j] , mesh.elements[eID].geometry.y[i,j] )
    u0[:,i,j,eID] = initial_condition(x_vec, 0.0, equations)
  end
end


###############################################################################
# throw this into a time loop and see what happens

tspan = (0.0, 0.5)
ode_algorithm = Trixi.CarpenterKennedy2N54()

u = copy(u0)
du = zeros( num_eqns , nnodes(solver) , nnodes(solver) , nelements(mesh) )
u_tmp = similar(du)

# hack together my own time loop for debugging before I modify the semidiscretization routines
let
t     = first(tspan)
t_end = last(tspan)
# fixed time step for now. TODO: calc_time_step will need modified to incorporate metric terms
dt = 1e-3
finalstep = false

while !finalstep
  if isnan(dt)
    error("time step size `dt` is NaN")
  end

# if the next iteration would push the simulation beyond the end time, set dt accordingly
  if t + dt > t_end || isapprox(t + dt, t_end)
    dt = t_end - t
    finalstep = true
  end

# one time step
  u_tmp .= 0
  for stage in eachindex(ode_algorithm.c)
    t_stage = t + dt * ode_algorithm.c[stage]
    rhs!(du, u, t_stage, mesh, equations, initial_condition, boundary_conditions, source_term, solver, cache)

    a_stage    = ode_algorithm.a[stage]
    b_stage_dt = ode_algorithm.b[stage] * dt
    @timeit_debug timer() "Runge-Kutta step" begin
      for i in eachindex(u)
        u_tmp[i] = du[i] - u_tmp[i] * a_stage
        u[i] += u_tmp[i] * b_stage_dt
      end
    end
  end
  t += dt
  println(t)
end

###############################################################################
# Construct and fill the exact solution at t_end and compute the L_inf error

u_exact = similar(u)

for eID in eachelement(mesh)
  for j in eachnode(solver), i in eachnode(solver)
    x_vec = ( mesh.elements[eID].geometry.x[i,j] , mesh.elements[eID].geometry.y[i,j])
    u_exact[:,i,j,eID] = initial_condition(x_vec, t_end, equations)
  end
end

linf_error = zeros(Trixi.nvariables(equations))
for eID in eachelement(mesh)
  for j in eachnode(solver), i in eachnode(solver)
    diff = u[:,i,j,eID] - u_exact[:,i,j,eID]
    linf_error = @. max(linf_error, abs(diff))
  end
end

println(linf_error)

end # let block

###############################################################################
# move everything into appropriate arrays for plotting
# all_x = zeros( nnodes(solver) , nnodes(solver) , nelements(mesh) )
# all_y = zeros( nnodes(solver) , nnodes(solver) , nelements(mesh) )
# solu0 = zeros( nnodes(solver) , nnodes(solver) , nelements(mesh) )
# solu  = zeros( nnodes(solver) , nnodes(solver) , nelements(mesh) )
# for eID in eachelement(mesh)
#   all_x[:,:,eID] = mesh.elements[eID].geometry.x
#   all_y[:,:,eID] = mesh.elements[eID].geometry.y
#   solu0[:,:,eID] = u0[1,:,:,eID]
#   solu[:,:,eID]  = u[1,:,:,eID]
# end
#
# for eID in eachelement(mesh)
# #  plot_surface( all_x[:,:,eID] , all_y[:,:,eID] , solu[:,:,eID])
#   surf( all_x[:,:,eID] , all_y[:,:,eID] , solu0[:,:,eID] , cmap=ColorMap("plasma"))
# end
#
# figure()
# for eID in eachelement(mesh)
# #  plot_surface( all_x[:,:,eID] , all_y[:,:,eID] , solu[:,:,eID])
#   surf( all_x[:,:,eID] , all_y[:,:,eID] , solu[:,:,eID] , cmap=ColorMap("plasma"))
# end
