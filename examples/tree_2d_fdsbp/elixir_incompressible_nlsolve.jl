# !!! warning "Experimental feature"
#     This is an experimental feature and may change in any future releases.

using NLsolve
using LinearAlgebra
using Trixi

###############################################################################
# semidiscretization of the two-dimensional incompressible Euler equations

equations = IncompressibleEulerEquations2D()

#initial_condition = initial_condition_constant
initial_condition = initial_condition_pulse

D_SBP = derivative_operator(SummationByPartsOperators.MattssonNordström2004(),
                            derivative_order=1, accuracy_order=4,
                            xmin=0.0, xmax=1.0, N=10)
solver = DG(D_SBP, nothing #= mortar =#,
            SurfaceIntegralStrongForm(flux_lax_friedrichs),
            VolumeIntegralStrongForm())

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=0,
                n_cells_max=30_000,
                periodicity=false)
# FIXME: this is a hack. incompressible Euler is using Wall BCs but not
#        via the BoundaryConditionWall formalism in other parts of Trixi

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solver, setup for the nonlinear solve

# set the time span and number of time steps
tspan = (0.0, 0.1)
n_steps = 10
dt = (tspan[2] - tspan[1])/n_steps

# initialize the solution using the initial condition
u0 = compute_coefficients(first(tspan), semi)

# define the spatial operator (note this will need to be time dependent in the future)
spatial_operator = x -> Trixi.space_approximation(x, semi, tspan[1])

# control
#L_av_x = spatial_operator(u0)

function backward_euler_step(op::Function, x_old, dt, t)
# (note this will need to be time dependent in the future for proper boundary conditions)
  N = size(x_old, 1)
  T = zeros(N)
  stop_idx = convert(Int64, 2*N/3)

  function f!(F, x)
    T[1:stop_idx] = x[1:stop_idx] - x_old[1:stop_idx]
    # # the values of op(x) are negated because the operator function returns the rhs!
    # # that must move back to the left hand side of the equations
    F .= T - dt * op(x)
  end

  result_ = nlsolve(f!, x_old, ftol=1e-12) # can set ftol or xtol for stopping
  #println(result_.residual_norm)
  return result_.zero # this is x_new
end

# approximate the Jacobian matrix using automatic differentiation via ForwardDiff
#J = jacobian_ad_forward(semi);

# u_old = similar(u0) # initialize the time loop
# u_old = u0
# u_new = similar(u_old)
# # time integration loop
# for k in 0:n_steps-1
#   t = (k+1) * dt
#   u_new = backward_euler_step(spatial_operator, u_old, dt, t)
#   u_old = u_new
# end

t = 0.0
# first step
u_new = backward_euler_step(spatial_operator, u0, dt, t)
u0 = u_new
# second step
u_new = backward_euler_step(spatial_operator, u0, dt, t)
# u0 = u_new
# # three step
# u_new = backward_euler_step(spatial_operator, u0, dt, t)
# u0 = u_new
# # four step
# u_new = backward_euler_step(spatial_operator, u0, dt, t)
# u0 = u_new
# # five step
# u_new = backward_euler_step(spatial_operator, u0, dt, t)
# u0 = u_new
# # six step
# u_new = backward_euler_step(spatial_operator, u0, dt, t)
# u0 = u_new
# # seven step
# u_new = backward_euler_step(spatial_operator, u0, dt, t)
# u0 = u_new
# # eight step
# u_new = backward_euler_step(spatial_operator, u0, dt, t)
# u0 = u_new
# # nine step
# u_new = backward_euler_step(spatial_operator, u0, dt, t)
# u0 = u_new
# # ten step
# u_new = backward_euler_step(spatial_operator, u0, dt, t)


# for plotting
u_plotter  = Trixi.wrap_array(u_new, mesh, equations, solver, semi.cache)
pd = PlotData2D(u_plotter, mesh, equations, solver, semi.cache)

plot(pd["v2"])
