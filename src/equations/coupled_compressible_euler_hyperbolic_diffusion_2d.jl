@muladd begin

  struct CouplingCompressibleEulerHyperbolicDiffusion2D{RealT<:Real} <: AbstractCouplingEquations{2, 1}
    coupling_nu::RealT     # diffusion constant at the coupling interface
  end
  
  function CouplingCompressibleEulerHyperbolicDiffusion2D(u::NTuple{7,<:Real})
    CouplingCompressibleEulerHyperbolicDiffusion2D(SVector(u))
  end
  
  function CouplingCompressibleEulerHyperbolicDiffusion2D(u_compressible_euler::NTuple{4,<:Real}, u_hyperbolic_diffusion::NTuple{3,<:Real})
    CouplingCompressibleEulerHyperbolicDiffusion2D(vcat(u_compressible_euler, u_hyperbolic_diffusion))
  end
  
  varnames(::typeof(cons2cons), ::CouplingLinearScalarAdvectionEquation2D) = ("rho", "rho_v1", "rho_v2", "rho_e", "phi", "q1", "q2")
  varnames(::typeof(cons2prim), ::CouplingLinearScalarAdvectionEquation2D) = ("rho", "v1", "v2", "p", "phi", "q1", "q2")

  default_analysis_errors(::CouplingLinearScalarAdvectionEquation2D) = (:l2_error, :linf_error, :residual)
  
  @inline function residual_steady_state(du, ::CouplingLinearScalarAdvectionEquation2D)
    abs(du[1])
  end
   
  # Calculate 1D flux in for a single point
  @inline function flux(u, orientation::Integer, equations::CouplingLinearScalarAdvectionEquation2D)
    rho, rho_v1, rho_v2, rho_e, phi, q1, q2 = u
    @unpack inv_Tr = equations
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
  
    if orientation == 1
      f1 = rho_v1
      f2 = rho_v1 * v1 + p
      f3 = rho_v1 * v2
      f4 = (rho_e + p) * v1
      f5 = -equations.nu*q1
      f6 = -phi * inv_Tr
      f7 = zero(phi)
    else
      f1 = rho_v2
      f2 = rho_v2 * v1
      f3 = rho_v2 * v2 + p
      f4 = (rho_e + p) * v2
      f5 = -equations.nu*q2
      f6 = zero(phi)
      f7 = -phi * inv_Tr
    end
  
    return SVector(f1, f2, f3, f4, f5, f6, f7)
  end
  
  # Note, this directional vector is not normalized
  @inline function flux(u, normal_direction::AbstractVector, equations::CouplingLinearScalarAdvectionEquation2D)
    rho, rho_v1, rho_v2, rho_e, phi, q1, q2 = u
    @unpack inv_Tr = equations
 
    rho, v1, v2, p, phi, q1, q2 = cons2prim(u, equations)
  
    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    rho_v_normal = rho * v_normal
    f1 = rho_v_normal
    f2 = rho_v_normal * v1 + p * normal_direction[1]
    f3 = rho_v_normal * v2 + p * normal_direction[2]
    f4 = (rho_e + p) * v_normal
  
    f5 = -equations.nu * (normal_direction[1] * q1 + normal_direction[2] * q2)
    f6 = -phi * inv_Tr * normal_direction[1]
    f7 = -phi * inv_Tr * normal_direction[2]
  
    return SVector(f1, f2, f3, f4, f5, f6, f7)
  end
  
  
  # Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
  @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CouplingLinearScalarAdvectionEquation2D)
    sqrt(equations.nu * equations.inv_Tr)
  end
  
  @inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::CouplingLinearScalarAdvectionEquation2D)
    sqrt(equations.nu * equations.inv_Tr) * norm(normal_direction)
  end
  
   
  @inline have_constant_speed(::CouplingLinearScalarAdvectionEquation2D) = Val(true)
  
  @inline function max_abs_speeds(eq::CouplingLinearScalarAdvectionEquation2D)
    λ = sqrt(eq.nu * eq.inv_Tr)
    return λ, λ
  end
  
  
  # Convert conservative variables to primitive
  @inline cons2prim(u, equations::CouplingLinearScalarAdvectionEquation2D) = u
  

  # Calculate entropy for a conservative state `u` (here: same as total energy)
  @inline entropy(u, equations::CouplingLinearScalarAdvectionEquation2D) = energy_total(u, equations)
  
  
  # Calculate total energy for a conservative state `u`
  @inline function energy_total(u, equations::CouplingLinearScalarAdvectionEquation2D)
    # energy function as found in equations (2.5.12) in the book "I Do Like CFD, Vol. 1"
    rho, rho_v1, rho_v2, rho_e, phi, q1, q2 = u
    return (rho_v1^2 + rho_v2^2) / (2 * rho) + 0.5 * (phi^2 + equations.Lr^2 * (q1^2 + q2^2))
  end
  
  
  end # @muladd
  