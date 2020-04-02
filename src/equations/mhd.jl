module MhdEquations

using ...Trixi
using ..Equations # Use everything to allow method extension via "function Equations.<method>"
using ...Auxiliary: parameter
using StaticArrays: SVector, MVector, MMatrix, MArray

# Export all symbols that should be available from Equations
export Mhd
export initial_conditions
export sources
export calcflux!
export riemann!
export calc_max_dt
export cons2prim
export cons2entropy
export cons2indicator
export cons2indicator!

# TODO: Riemann solvers, time step calculation

# Main data structure for system of equations "Mhd"
struct Mhd <: AbstractEquation{8}
  name::String
  initial_conditions::String
  sources::String
  varnames_cons::SVector{8, String}
  varnames_prim::SVector{8, String}
  gamma::Float64
  surface_flux_type::Symbol
  volume_flux_type::Symbol

  function Mhd()
    name = "mhd"
    initial_conditions = parameter("initial_conditions")
    sources = parameter("sources", "none")
    varnames_cons = ["rho", "rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3"]
    varnames_prim = ["rho", "v1", "v2", "v3", "p", "B1", "B2", "B3"]
    gamma = 1.4 # 1.6666666666666667
    surface_flux_type = Symbol(parameter("surface_flux_type", "laxfriedrichs",
                                         valid=["laxfriedrichs","central"]))
    volume_flux_type = Symbol(parameter("volume_flux_type", "central",
                                        valid=["central"]))
    new(name, initial_conditions, sources, varnames_cons, varnames_prim, gamma,
        surface_flux_type, volume_flux_type)
  end
end


# Set initial conditions at physical location `x` for time `t`
function Equations.initial_conditions(equation::Mhd, x::AbstractArray{Float64}, t::Real)
  name = equation.initial_conditions
  if name == "constant"
    rho = 1.0
    rho_v1 = 0.1
    rho_v2 = -0.2
    rho_v3 = -0.5
    rho_e = 10.0
    B1 = 3.0
    B2 = -1.2
    B3 = 0.4
    return [rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3]
  elseif name == "convergence_test"
    # smooth Alfvén wave test from Derigs et al. FLASH (2016)
    # domain must be set to [0, 1/cos(α)] x [0, 1/sin(α)], γ = 5/3
    α = 0.25*pi
    x_perp = x[1]*cos(α) + x[2]*sin(α)
    B_perp = 0.1*sin(2.0*pi*x_perp)
    rho = 1.0
    v1 = -B_perp*sin(α)
    v2 = B_perp*cos(α)
    v3 = 0.1*cos(2.0*pi*x_perp)
    p = 0.1
    B1 = cos(α) + v1
    B2 = sin(α) + v2
    B3 = v3
    return prim2cons(equation, [rho, v1, v2, v3, p, B1, B2, B3])
  elseif name == "orszag_tang"
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [0, 1] x [0, 1], γ = 5/3
    rho = 1.0
    v1 = -sin(2.0*pi*x[2])
    v2 = sin(2.0*pi*x[1])
    v3 = 0.0
    p = 1.0/equation.gamma
    B1 = -sin(2.0*pi*x[2])/equation.gamma
    B2 = sin(4.0*pi*x[1])/equation.gamma
    B3 = 0.0
    return prim2cons(equation, [rho, v1, v2, v3, p, B1, B2, B3])
  elseif name == "rotor"
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [0, 1] x [0, 1], γ = 1.4
    Δx = x[1] - 0.5
    Δy = x[2] - 0.5
    r = sqrt(Δx^2 + Δy^2)
    f = (0.115 - r)/0.015
    if r <= 0.1
      rho = 10.0
      v1 = -20.0*Δy
      v2 = 20.0*Δx
    elseif r >= 0.115
      rho = 1.0
      v1 = 0.0
      v2 = 0.0
    else
      rho = 1.0 + 9.0*f
      v1 = -20.0*f*Δy
      v2 = 20.0*f*Δx
    end
    v3 = 0.0
    p = 1.0
    B1 = 5.0/sqrt(4.0*pi)
    B2 = 0.0
    B3 = 0.0
    return prim2cons(equation, [rho, v1, v2, v3, p, B1, B2, B3])
  elseif name == "mhd_blast"
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [-0.5, 0.5] x [-0.5, 0.5], γ = 1.4
    r = sqrt(x[1]^2 + x[2]^2)
    f = (0.1 - r)/0.01
    if r <= 0.09
      p = 1000.0
    elseif r >= 0.1
      p = 0.1
    else
      p = 0.1 + 999.9*f
    end
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    B1 = 100.0/sqrt(4.0*pi)
    B2 = 0.0
    B3 = 0.0
    return prim2cons(equation, [rho, v1, v2, v3, p, B1, B2, B3])
  else
    error("Unknown initial condition '$name'")
  end
end


# Apply source terms
function Equations.sources(equation::Mhd, ut, u, x, element_id, t, n_nodes)
  name = equation.sources
  error("Unknown source term '$name'")
end


# Calculate 2D flux (element version)
@inline function Equations.calcflux!(f1::AbstractArray{Float64},
                                     f2::AbstractArray{Float64},
                                     equation::Mhd,
                                     u::AbstractArray{Float64}, element_id::Int,
                                     n_nodes::Int)
  for j = 1:n_nodes
    for i = 1:n_nodes
      rho    = u[1, i, j, element_id]
      rho_v1 = u[2, i, j, element_id]
      rho_v2 = u[3, i, j, element_id]
      rho_v3 = u[4, i, j, element_id]
      rho_e  = u[5, i, j, element_id]
      B1     = u[6, i, j, element_id]
      B2     = u[7, i, j, element_id]
      B3     = u[8, i, j, element_id]
      @views calcflux!(f1[:, i, j], f2[:, i, j], equation, rho, rho_v1, rho_v2, rho_v3, rho_e,
                       B1, B2, B3)
    end
  end
end


# Calculate 2D flux (pointwise version)
@inline function Equations.calcflux!(f1::AbstractArray{Float64},
                                     f2::AbstractArray{Float64},
                                     equation::Mhd,
                                     rho::Float64, rho_v1::Float64,
                                     rho_v2::Float64, rho_v3::Float64,
                                     rho_e::Float64, B1::Float64,
                                     B2::Float64, B3::Float64)
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  mag_en = 0.5*(B1^2 + B2^2 + B3^2)
  v_dot_B = v1*B1 + v2*B2 + v3*B3
  p = (equation.gamma - 1)*(rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2) - mag_en)

  f1[1] = rho_v1
  f1[2] = rho_v1*v1 + p + mag_en - B1^2
  f1[3] = rho_v1*v2 - B1*B2
  f1[4] = rho_v1*v3 - B1*B3
  f1[5] = (rho_e + p + mag_en)*v1 - B1*v_dot_B
  f1[6] = 0.0
  f1[7] = v1*B2 - v2*B1
  f1[8] = v1*B3 - v3*B1

  f2[1] = rho_v2
  f2[2] = rho_v2*v1 - B1*B2
  f2[3] = rho_v2*v2 + p + mag_en - B2^2
  f2[4] = rho_v2*v3 - B2*B3
  f2[5] = (rho_e + p + mag_en)*v2 - B2*v_dot_B
  f2[6] = v2*B1 - v1*B2
  f2[7] = 0.0
  f2[8] = v2*B3 - v3*B2
end


# Calculate 2D two-point flux (decide which volume flux type to use)
@inline function Equations.calcflux_twopoint!(f1::AbstractArray{Float64},
                                              f2::AbstractArray{Float64},
                                              f1_diag::AbstractArray{Float64},
                                              f2_diag::AbstractArray{Float64},
                                              equation::Mhd,
                                              u::AbstractArray{Float64},
                                              element_id::Int, n_nodes::Int)
  calcflux_twopoint!(f1, f2, f1_diag, f2_diag, Val(equation.volume_flux_type),
                     equation, u, element_id, n_nodes)
end


# Calculate 2D two-point flux (element version)
@inline function Equations.calcflux_twopoint!(f1::AbstractArray{Float64},
                                              f2::AbstractArray{Float64},
                                              f1_diag::AbstractArray{Float64},
                                              f2_diag::AbstractArray{Float64},
                                              twopoint_flux_type::Val,
                                              equation::Mhd,
                                              u::AbstractArray{Float64},
                                              element_id::Int, n_nodes::Int)
  # Calculate regular volume fluxes
  calcflux!(f1_diag, f2_diag, equation, u, element_id, n_nodes)


  for j = 1:n_nodes
    for i = 1:n_nodes
      # Set diagonal entries (= regular volume fluxes due to consistency)
      for v in 1:nvariables(equation)
        f1[v, i, i, j] = f1_diag[v, i, j]
        f2[v, j, i, j] = f2_diag[v, i, j]
      end

      # Flux in x-direction
      for l = i + 1:n_nodes
        @views symmetric_twopoint_flux!(f1[:, l, i, j], twopoint_flux_type,
                                        equation, 1, # 1-> x-direction
                                        u[1, i, j, element_id], u[2, i, j, element_id],
                                        u[3, i, j, element_id], u[4, i, j, element_id],
                                        u[5, i, j, element_id], u[6, i, j, element_id],
                                        u[7, i, j, element_id], u[8, i, j, element_id],
                                        u[1, l, j, element_id], u[2, l, j, element_id],
                                        u[3, l, j, element_id], u[4, l, j, element_id],
                                        u[5, l, j, element_id], u[6, l, j, element_id],
                                        u[7, l, j, element_id], u[8, l, j, element_id])
        for v in 1:nvariables(equation)
          f1[v, i, l, j] = f1[v, l, i, j]
        end
      end

      # Flux in y-direction
      for l = j + 1:n_nodes
        @views symmetric_twopoint_flux!(f2[:, l, i, j], twopoint_flux_type,
                                        equation, 2, # 2 -> y-direction
                                        u[1, i, j, element_id], u[2, i, j, element_id],
                                        u[3, i, j, element_id], u[4, i, j, element_id],
                                        u[5, i, j, element_id], u[6, i, j, element_id],
                                        u[7, i, j, element_id], u[8, i, j, element_id],
                                        u[1, i, l, element_id], u[2, i, l, element_id],
                                        u[3, i, l, element_id], u[4, i, l, element_id],
                                        u[5, i, l, element_id], u[6, i, l, element_id],
                                        u[7, i, l, element_id], u[8, i, l, element_id])
        for v in 1:nvariables(equation)
          f2[v, j, i, l] = f2[v, l, i, j]
        end
      end
    end
  end
end


# Central two-point flux (identical to weak form volume integral, except for floating point errors)
@inline function symmetric_twopoint_flux!(f::AbstractArray{Float64}, ::Val{:central},
                                          equation::Mhd, orientation::Int,
                                          rho_ll::Float64,
                                          rho_v1_ll::Float64,
                                          rho_v2_ll::Float64,
                                          rho_v3_ll::Float64,
                                          rho_e_ll::Float64,
                                          B1_ll::Float64,
                                          B2_ll::Float64,
                                          B3_ll::Float64,
                                          rho_rr::Float64,
                                          rho_v1_rr::Float64,
                                          rho_v2_rr::Float64,
                                          rho_v3_rr::Float64,
                                          rho_e_rr::Float64,
                                          B1_rr::Float64,
                                          B2_rr::Float64,
                                          B3_rr::Float64)
  # Calculate regular 1D fluxes
  f_ll = MVector{8, Float64}(undef)
  f_rr = MVector{8, Float64}(undef)
  calcflux1D!(f_ll, equation, rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
              B1_ll, B2_ll, B3_ll, orientation)
  calcflux1D!(f_rr, equation, rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
              B1_rr, B2_rr, B3_rr, orientation)
  # Average regular fluxes
  @. f[:] = 0.5*(f_ll + f_rr)
end


# Calculate 1D flux in for a single point
@inline function calcflux1D!(f::AbstractArray{Float64}, equation::Mhd, rho::Float64,
                             rho_v1::Float64, rho_v2::Float64, rho_v3::Float64, rho_e::Float64,
                             B1::Float64, B2::Float64, B3::Float64, orientation::Int)
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  mag_en = 0.5*(B1^2 + B2^2 + B3^2)
  p = (equation.gamma - 1) * (rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2) - mag_en)
  if orientation == 1
    f[1]  = rho_v1
    f[2]  = rho_v1 * v1 + p + mag_en - B1^2
    f[3]  = rho_v1 * v2 - B1 * B2
    f[4]  = rho_v1 * v3 - B1 * B3
    f[5]  = (rho_e + p + mag_en) * v1 - B1 * (v1 * B1 + v2 * B2 + v3 * B3)
    f[6]  = 0.0
    f[7]  = v1 * B2 - v2 * B1
    f[8]  = v1 * B3 - v3 * B1
  else
    f[1]  = rho_v2
    f[2]  = rho_v2 * v1 - B1 * B2
    f[3]  = rho_v2 * v2 + p + mag_en - B2^2
    f[4]  = rho_v2 * v3 - B2 * B3
    f[5]  = (rho_e + p + mag_en) * v2 - B2 * (v1 * B1 + v2 * B2 + v3 * B3)
    f[6]  = v2 * B1 - v1 * B2
    f[7]  = 0.0
    f[8]  = v2 * B3 - v3 * B2
  end
end


# Calculate flux across interface with different states on both sides (EC mortar version)
function Equations.riemann!(surface_flux::AbstractArray{Float64, 3},
                            fstarnode::AbstractVector{Float64},
                            u_surfaces_left::AbstractArray{Float64, 3},
                            u_surfaces_right::AbstractArray{Float64, 3},
                            surface_id::Int,
                            equation::Mhd, n_nodes::Int,
                            orientations::Vector{Int})
  # Call pointwise Riemann solver
  # i -> left, j -> right
  for j = 1:n_nodes
    for i = 1:n_nodes
      # Store flux in pre-allocated `fstarnode` to avoid allocations in loop
      riemann!(fstarnode,
               u_surfaces_left[1, i, surface_id],
               u_surfaces_left[2, i, surface_id],
               u_surfaces_left[3, i, surface_id],
               u_surfaces_left[4, i, surface_id],
               u_surfaces_left[5, i, surface_id],
               u_surfaces_left[6, i, surface_id],
               u_surfaces_left[7, i, surface_id],
               u_surfaces_left[8, i, surface_id],
               u_surfaces_right[1, j, surface_id],
               u_surfaces_right[2, j, surface_id],
               u_surfaces_right[3, j, surface_id],
               u_surfaces_right[4, j, surface_id],
               u_surfaces_right[5, j, surface_id],
               u_surfaces_right[6, j, surface_id],
               u_surfaces_right[7, j, surface_id],
               u_surfaces_right[8, j, surface_id],
               equation, orientations[surface_id])

      # Copy flux back to actual flux array
      for v in 1:nvariables(equation)
        surface_flux[v, i, j] = fstarnode[v]
      end
    end
  end
end


# Calculate flux across interface with different states on both sides (surface version)
function Equations.riemann!(surface_flux::AbstractMatrix{Float64},
                            fstarnode::AbstractVector{Float64},
                            u_surfaces::AbstractArray{Float64, 4},
                            surface_id::Int,
                            equation::Mhd, n_nodes::Int,
                            orientations::Vector{Int})
  # Call pointwise Riemann solver
  for i = 1:n_nodes
    # Store flux in pre-allocated `fstarnode` to avoid allocations in loop
    riemann!(fstarnode,
             u_surfaces[1, 1, i, surface_id],
             u_surfaces[1, 2, i, surface_id],
             u_surfaces[1, 3, i, surface_id],
             u_surfaces[1, 4, i, surface_id],
             u_surfaces[1, 5, i, surface_id],
             u_surfaces[1, 6, i, surface_id],
             u_surfaces[1, 7, i, surface_id],
             u_surfaces[1, 8, i, surface_id],
             u_surfaces[2, 1, i, surface_id],
             u_surfaces[2, 2, i, surface_id],
             u_surfaces[2, 3, i, surface_id],
             u_surfaces[2, 4, i, surface_id],
             u_surfaces[2, 5, i, surface_id],
             u_surfaces[2, 6, i, surface_id],
             u_surfaces[2, 7, i, surface_id],
             u_surfaces[2, 8, i, surface_id],
             equation, orientations[surface_id])

    # Copy flux back to actual flux array
    for v in 1:nvariables(equation)
      surface_flux[v, i] = fstarnode[v]
    end
  end
end


# Calculate flux across interface with different states on both sides (pointwise version)
function Equations.riemann!(surface_flux::AbstractArray{Float64, 1},
                            rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll,
                            rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr,
                            equation::Mhd, orientation::Int)
  # Calculate primitive variables and fast magnetoacoustic wave speeds
  # left
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2 + v3_ll^2)
  p_ll = (equation.gamma - 1)*(rho_e_ll - 0.5*rho_ll*v_mag_ll^2 - 0.5*(B1_ll^2 + B2_ll^2 + B3_ll^2))
  # sound speed squared
  a_square_ll = equation.gamma * p_ll / rho_ll
  # Alfvén wave speeds
  b1_ll = B1_ll / sqrt(rho_ll)
  b2_ll = B2_ll / sqrt(rho_ll)
  b3_ll = B3_ll / sqrt(rho_ll)
  b_square_ll = b1_ll^2 + b2_ll^2 + b3_ll^2
  # fast magnetoacoustic wave speed
  if orientation == 1 # x-direction
    b_dir = b1_ll
  else
    b_dir = b2_ll
  end
  cf_ll = sqrt(0.5*(a_square_ll + b_square_ll) +
               0.5*sqrt((a_square_ll + b_square_ll)^2 - 4.0*a_square_ll*b_dir^2))
  # right
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2 + v3_rr^2)
  p_rr = (equation.gamma - 1)*(rho_e_rr - 0.5*rho_rr*v_mag_rr^2 - 0.5*(B1_rr^2 + B2_rr^2 + B3_rr^2))
  # sound speed squared
  a_square_rr = equation.gamma * p_rr / rho_rr
  # Alfvén wave speeds
  b1_rr = B1_rr / sqrt(rho_rr)
  b2_rr = B2_rr / sqrt(rho_rr)
  b3_rr = B3_rr / sqrt(rho_rr)
  b_square_rr = b1_rr^2 + b2_rr^2 + b3_rr^2
  # fast magnetoacoustic wave speed
  if orientation == 1 # x-direction
    b_dir = b1_rr
  else
    b_dir = b2_rr
  end
  cf_rr = sqrt(0.5*(a_square_rr + b_square_rr) +
               0.5*sqrt((a_square_rr + b_square_rr)^2 - 4.0*a_square_rr*b_dir^2))
  # Obtain left and right fluxes
  f_ll = zeros(MVector{8})
  f_rr = zeros(MVector{8})
  calcflux1D!(f_ll, equation, rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
              B1_ll, B2_ll, B3_ll, orientation)
  calcflux1D!(f_rr, equation, rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
              B1_rr, B2_rr, B3_rr, orientation)
  if equation.surface_flux_type == :laxfriedrichs
    λ_max = max(v_mag_ll, v_mag_rr) + max(cf_ll, cf_rr)
    surface_flux[1] = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (rho_rr    - rho_ll)
    surface_flux[2] = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (rho_v1_rr - rho_v1_ll)
    surface_flux[3] = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (rho_v2_rr - rho_v2_ll)
    surface_flux[4] = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (rho_v3_rr - rho_v3_ll)
    surface_flux[5] = 1/2 * (f_ll[5] + f_rr[5]) - 1/2 * λ_max * (rho_e_rr  - rho_e_ll)
    surface_flux[6] = 1/2 * (f_ll[6] + f_rr[6]) - 1/2 * λ_max * (B1_rr     - B1_ll)
    surface_flux[7] = 1/2 * (f_ll[7] + f_rr[7]) - 1/2 * λ_max * (B2_rr     - B2_ll)
    surface_flux[8] = 1/2 * (f_ll[8] + f_rr[8]) - 1/2 * λ_max * (B3_rr     - B3_ll)
  elseif equation.surface_flux_type in (:central)
    symmetric_twopoint_flux!(surface_flux, Val(equation.surface_flux_type),
                             equation, orientation,
                             rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll,
                             rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr)
  else
    error("unknown Riemann solver '$(string(equation.surface_flux_type))'")
  end
end


# Determine maximum stable time step based on polynomial degree and CFL number
function Equations.calc_max_dt(equation::Mhd, u::Array{Float64, 4},
                               element_id::Int, n_nodes::Int,
                               invjacobian::Float64, cfl::Float64)
  λ_max = 0.0
  for j = 1:n_nodes
    for i = 1:n_nodes
      rho    = u[1, i, j, element_id]
      rho_v1 = u[2, i, j, element_id]
      rho_v2 = u[3, i, j, element_id]
      rho_v3 = u[4, i, j, element_id]
      rho_e  = u[5, i, j, element_id]
      B1     = u[6, i, j, element_id]
      B2     = u[7, i, j, element_id]
      B3     = u[8, i, j, element_id]
      v1 = rho_v1/rho
      v2 = rho_v2/rho
      v3 = rho_v3/rho
      v_mag = sqrt(v1^2 + v2^2 + v3^2)
      p = (equation.gamma - 1)*(rho_e - 0.5*rho*v_mag^2 - 0.5*(B1^2 + B2^2 + B3^2))
      a_square = equation.gamma * p / rho
      b1 = B1/sqrt(rho)
      b2 = B2/sqrt(rho)
      b3 = B3/sqrt(rho)
      b_square = b1^2 + b2^2 + b3^2
      cf_x_direction = sqrt(0.5*(a_square + b_square) +
                            0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b1^2))
      cf_y_direction = sqrt(0.5*(a_square + b_square) +
                            0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b2^2))
      cf_max = max(cf_x_direction,cf_y_direction)
      λ_max = max(λ_max, v_mag + cf_max)
    end
  end

  dt = cfl * 2 / (invjacobian * λ_max) / n_nodes

  return dt
end


# Convert conservative variables to primitive
function Equations.cons2prim(equation::Mhd, cons::Array{Float64, 4})
  prim = similar(cons)
  @. prim[1, :, :, :] = cons[1, :, :, :]
  @. prim[2, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. prim[3, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. prim[4, :, :, :] = cons[4, :, :, :] / cons[1, :, :, :]
  @. prim[5, :, :, :] = ((equation.gamma - 1)
                         *(cons[5, :, :, :] - 0.5 * (cons[2, :, :, :] * prim[2, :, :, :] +
                                                     cons[3, :, :, :] * prim[3, :, :, :] +
                                                     cons[4, :, :, :] * prim[4, :, :, :])
                                            - 0.5 * (cons[6, :, :, :] * cons[6, :, :, :] +
                                                     cons[7, :, :, :] * cons[7, :, :, :] +
                                                     cons[8, :, :, :] * cons[8, :, :, :])))
  @. prim[6, :, :, :] = cons[6, :, :, :]
  @. prim[7, :, :, :] = cons[7, :, :, :]
  @. prim[8, :, :, :] = cons[8, :, :, :]
  return prim
end


# Convert conservative variables to entropy
function Equations.cons2entropy(equation::Mhd,cons::Array{Float64, 4},n_nodes::Int,n_elements::Int)
  entropy = similar(cons)
  v = zeros(3,n_nodes,n_nodes,n_elements)
  B = zeros(3,n_nodes,n_nodes,n_elements)
  v_square = zeros(n_nodes,n_nodes,n_elements)
  p = zeros(n_nodes,n_nodes,n_elements)
  s = zeros(n_nodes,n_nodes,n_elements)
  rho_p = zeros(n_nodes,n_nodes,n_elements)
# velocities
  @. v[1, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. v[2, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. v[3, :, :, :] = cons[4, :, :, :] / cons[1, :, :, :]
# magnetic fields
  @. B[1, :, :, :] = cons[6, :, :, :]
  @. B[2, :, :, :] = cons[7, :, :, :]
  @. B[3, :, :, :] = cons[8, :, :, :]
# kinetic energy, pressure, entropy
  @. v_square[ :, :, :] = v[1, :, :, :]*v[1, :, :, :]+v[2, :, :, :]*v[2, :, :, :] +
                          v[3, :, :, :]*v[3, :, :, :]
  @. p[ :, :, :] = ((equation.gamma - 1)*(cons[5, :, :, :] - 0.5*cons[1, :, :, :]*v_square[:,:,:] -
                            0.5*(B[1, :, :, :]*B[1, :, :, :] + B[2, :, :, :]*B[2, :, :, :] +
                                 B[3, :, :, :]*B[3, :, :, :])))
  @. s[ :, :, :] = log(p[:, :, :]) - equation.gamma*log(cons[1, :, :, :])
  @. rho_p[ :, :, :] = cons[1, :, :, :] / p[ :, :, :]

  @. entropy[1, :, :, :] = (equation.gamma - s[:,:,:])/(equation.gamma-1) -
                           0.5*rho_p[:,:,:]*v_square[:,:,:]
  @. entropy[2, :, :, :] = rho_p[:,:,:]*v[1,:,:,:]
  @. entropy[3, :, :, :] = rho_p[:,:,:]*v[2,:,:,:]
  @. entropy[4, :, :, :] = rho_p[:,:,:]*v[3,:,:,:]
  @. entropy[5, :, :, :] = -rho_p[:,:,:]
  @. entropy[6, :, :, :] = rho_p[:,:,:]*B[1,:,:,:]
  @. entropy[7, :, :, :] = rho_p[:,:,:]*B[2,:,:,:]
  @. entropy[8, :, :, :] = rho_p[:,:,:]*B[3,:,:,:]

  return entropy
end

# Convert primitive to conservative variables
function prim2cons(equation::Mhd, prim::AbstractArray{Float64})
  cons = similar(prim)
  cons[1] = prim[1]
  cons[2] = prim[2] * prim[1]
  cons[3] = prim[3] * prim[1]
  cons[4] = prim[4] * prim[1]
  cons[5] = prim[5]/(equation.gamma-1)+0.5*(cons[2]*prim[2] + cons[3]*prim[3] + cons[4]*prim[4])+
            0.5*(prim[6]*prim[6] + prim[7]*prim[7] + prim[8]*prim[8])
  cons[6] = prim[6]
  cons[7] = prim[7]
  cons[8] = prim[8]
  return cons
end


# Convert conservative variables to indicator variable for discontinuities (elementwise version)
@inline function Equations.cons2indicator!(indicator::AbstractArray{Float64}, equation::Mhd,
                                           cons::AbstractArray{Float64},
                                           element_id::Int, n_nodes::Int)
  for j in 1:n_nodes
    for i in 1:n_nodes
      indicator[1, i, j] = cons2indicator(equation,
                                          cons[1, i, j, element_id], cons[2, i, j, element_id],
                                          cons[3, i, j, element_id], cons[4, i, j, element_id],
                                          cons[5, i, j, element_id], cons[6, i, j, element_id],
                                          cons[7, i, j, element_id], cons[8, i, j, element_id])
    end
  end
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function Equations.cons2indicator(equation::Mhd, rho, rho_v1, rho_v2, rho_v3, rho_e,
                                          B1, B2, B3)
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho

  # Calculate pressure
  p = (equation.gamma - 1)*(rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2) - 0.5*(B1^2 + B2^2 + B3^2))

  # Indicator variable is rho * p
  return rho * p
end

end # module
