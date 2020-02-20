module ShallowWaterEquations

using ...Trixi
using ..Equations # Use everything to allow method extension via "function Equations.<method>"
using ...Auxiliary: parameter
using StaticArrays: SVector, MVector, MMatrix

# Export all symbols that should be available from Equations
export ShallowWater
export initial_conditions
export sources
export calcflux!
export riemann!
export calc_max_dt
export cons2prim


# Main data structure for system of equations "ShallowWater"
struct ShallowWater <: AbstractEquation{2}
  name::String
  initial_conditions::String
  sources::String
  varnames_cons::SVector{2, String}
  varnames_prim::SVector{2, String}
  gravityacc::Float64
  riemann_solver::String

  function ShallowWater()
    name = "shallowwater"
    initial_conditions = parameter("initial_conditions")
    sources = parameter("sources", "none")
    # if initial condition is the manufactured solution convergence test, set the source term accordingly
    if initial_conditions == "convergence_test"
     sources = "convergence_test"
    end
    varnames_cons = ["h", "h_v"]
    varnames_prim = ["h", "u"]
    gravityacc = 1.0
    # I do not understand that "hllc" is there, what does it mean
    # riemann_solver = parameter("riemann_solver", "hllc", valid=["ec","es", "laxfriedrichs"])
    riemann_solver = parameter("riemann_solver","laxfriedrichs", valid=["laxfriedrichs"])
    new(name, initial_conditions, sources, varnames_cons, varnames_prim, gravityacc, riemann_solver)
  end
end


# Set initial conditions at physical location `x` for time `t`
function Equations.initial_conditions(equation::ShallowWater, x, t)
  name = equation.initial_conditions
  if name == "constant"
    return [1.0, 2.0]
  elseif name == "convergence_test"
    c = 2.0 # waterheight offset
    L = 1.0 # domain length
    f = 1/L # periodicity frequency
    omega = 2 * pi * f
    h = c + sin(omega*(x-t))
    h_v = h*1.0
    return [h, h_v]
  else
    error("Unknown initial condition '$name'")
  end
end


# Apply source terms
function Equations.sources(equation::ShallowWater, ut, u, x, cell_id, t, n_nodes)
  name = equation.sources
  if name == "convergence_test"
    # these values need to be consistent to the initial condition
    c = 2.0 # waterheight offset
    L = 1.0 # domain length
    f = 1/L # periodicity frequency
    omega = 2 * pi * f
    for i = 1:n_nodes
      x_loc = x[i,cell_id]
      h = c + sin(omega*(x_loc-t))
      h_x = omega*cos(omega*(x_loc-t))
      # source term computation
      # first source term is zero
      # second source term is h_t*v + h_x*v^2 + g*h*h_x = g*h*h_x because of the specific choice
      ut[2,i,cell_id] += equation.gravityacc*h*h_x
    end
  else
    error("Unknown source term '$name'")
  end
end


# Calculate flux at a given cell id
@inline function Equations.calcflux!(f::AbstractArray{Float64}, equation::ShallowWater,
                                     u::Array{Float64, 3}, cell_id::Int, n_nodes::Int)
  @inbounds for i = 1:n_nodes
    h   = u[1, i, cell_id]
    h_v = u[2, i, cell_id]
    @views calcflux!(f[:, i], equation, h, h_v)
  end
end

# Calculate flux for a give state
@inline function Equations.calcflux!(f::AbstractArray{Float64}, equation::ShallowWater, h::Float64,
                                     h_v::Float64)
  v = h_v/h
  p = 0.5*equation.gravityacc*h*h

  f[1]  = h_v
  f[2]  = h_v * v + p
end


# Calculate flux across interface with different states on both sides (Riemann problem)
function Equations.riemann!(flux_surfaces::Array{Float64, 2},
                            u_surfaces::Array{Float64, 3}, surface_id::Int,
                            equation::ShallowWater, n_nodes::Int)

  # Store for convenience
  h_ll   = u_surfaces[1, 1, surface_id]
  h_v_ll = u_surfaces[1, 2, surface_id]
  h_rr   = u_surfaces[2, 1, surface_id]
  h_v_rr = u_surfaces[2, 2, surface_id]

  # Obtain left and right fluxes
  f_ll = zeros(MVector{2})
  f_rr = zeros(MVector{2})
  calcflux!(f_ll, equation, h_ll, h_v_ll)
  calcflux!(f_rr, equation, h_rr, h_v_rr)

  # Calculate speed and sound speed
  v_ll = h_v_ll / h_ll
  c_ll = sqrt(equation.gravityacc * h_ll)
  v_rr = h_v_rr / h_rr
  c_rr = sqrt(equation.gravityacc * h_rr)

  if equation.riemann_solver == "laxfriedrichs"
    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
    flux_surfaces[1, surface_id] = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (h_rr - h_ll)
    flux_surfaces[2, surface_id] = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (h_v_rr - h_v_ll)
  else
    error("unknown Riemann solver '$(equation.riemann_solver)'")
  end
end

# Determine maximum stable time step based on polynomial degree and CFL number
function Equations.calc_max_dt(equation::ShallowWater, u::Array{Float64, 3},
                               cell_id::Int, n_nodes::Int,
                               invjacobian::Float64, cfl::Float64)
  λ_max = 0.0
  for i = 1:n_nodes
    h   = u[1, i, cell_id]
    h_v = u[2, i, cell_id]
    v = h_v/h
    c = sqrt(equation.gravityacc * h)
    λ_max = max(λ_max, abs(v) + c)
  end

  dt = cfl * 2 / (invjacobian * λ_max) / (2 * (n_nodes - 1) + 1)

  return dt
end


# Convert conservative variables to primitive
function Equations.cons2prim(equation::ShallowWater, cons::Array{Float64, 3})
  prim = similar(cons)
  @. prim[1, :, :] = cons[1, :, :]
  @. prim[2, :, :] = cons[2, :, :] / cons[1, :, :]
  return prim
end

end # module
