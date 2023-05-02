@muladd begin

  struct CouplingCompressibleEulerHyperbolicDiffusion2D{RealT<:Real} <: AbstractCouplingEquations{2, 7}
    coupling_nu::RealT     # diffusion constant at the coupling interface
  end
  

  function CouplingCompressibleEulerHyperbolicDiffusion2D(u::NTuple{7,<:Real})
    CouplingCompressibleEulerHyperbolicDiffusion2D(SVector(u))
  end
  

  function CouplingCompressibleEulerHyperbolicDiffusion2D(u_compressible_euler::NTuple{4,<:Real}, u_hyperbolic_diffusion::NTuple{3,<:Real})
    CouplingCompressibleEulerHyperbolicDiffusion2D(vcat(u_compressible_euler, u_hyperbolic_diffusion))
  end
  

  varnames(::typeof(cons2cons), ::CouplingCompressibleEulerHyperbolicDiffusion2D) = ("rho", "rho_v1", "rho_v2", "rho_e", "phi", "q1", "q2")
  varnames(::typeof(cons2prim), ::CouplingCompressibleEulerHyperbolicDiffusion2D) = ("rho", "v1", "v2", "p", "phi", "q1", "q2")

  default_analysis_errors(::CouplingCompressibleEulerHyperbolicDiffusion2D) = (:l2_error, :linf_error, :residual)
  

  @inline function residual_steady_state(du, ::CouplingCompressibleEulerHyperbolicDiffusion2D)
    abs(du[1])
  end
   

  # Calculate 1D flux in for a single point
  @inline function flux(u, orientation::Integer, equations::CouplingCompressibleEulerHyperbolicDiffusion2D)
    rho, rho_v1, rho_v2, rho_e, phi, q1, q2 = u
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
  
    if orientation == 1
      f1 = 0.0
      f2 = 0.0
      f3 = 0.0
      f4 = -equations.coupling_nu*phi
      f5 = equations.coupling_nu*phi
      f6 = 0.0
      f7 = 0.0
    else
      f1 = 0.0
      f2 = 0.0
      f3 = 0.0
      f4 = -equations.coupling_nu*phi
      f5 = equations.coupling_nu*phi
      f6 = 0.0
      f7 = 0.0
    end
  
    return SVector(f1, f2, f3, f4, f5, f6, f7)
  end
  

  # Note, this directional vector is not normalized
  @inline function flux(u, normal_direction::AbstractVector, equations::CouplingCompressibleEulerHyperbolicDiffusion2D)
    rho, rho_v1, rho_v2, rho_e, phi, q1, q2 = u 
    rho, v1, v2, p, phi, q1, q2 = cons2prim(u, equations)

    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    rho_v_normal = rho * v_normal
    f1 = 0.0
    f2 = 0.0
    f3 = 0.0
    f4 = -equations.coupling_nu * phi
    
    f5 = equations.coupling_nu * phi
    f6 = 0.0
    f7 = 0.0
  
    return SVector(f1, f2, f3, f4, f5, f6, f7)
  end
  
  
  # Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
  @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CouplingCompressibleEulerHyperbolicDiffusion2D)
    return equations.coupling_nu
  end
  

  @inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::CouplingCompressibleEulerHyperbolicDiffusion2D)
    # rho, rho_v1, rho_v2, rho_e, phi, q1, q2 = (u_ll + u_rr)
    # rho, v1, v2, p, phi, q1, q2 = (cons2prim(u_ll, equations) + cons2prim(u_rr, equations))/2

    # v = (q1 * normal_direction[1] + q2 * normal_direction[2])

    # return abs(v) * norm(normal_direction)

    return equations.coupling_nu
  end
  
   
  @inline have_constant_speed(::CouplingCompressibleEulerHyperbolicDiffusion2D) = Val(true)
  

  @inline function max_abs_speeds(eq::CouplingCompressibleEulerHyperbolicDiffusion2D)
    λ = sqrt(eq.nu * eq.inv_Tr)
    return λ, λ
  end
  
  
  # Convert conservative variables to primitive
  @inline cons2prim(u, equations::CouplingCompressibleEulerHyperbolicDiffusion2D) = u
  

  # Calculate entropy for a conservative state `u` (here: same as total energy)
  @inline entropy(u, equations::CouplingCompressibleEulerHyperbolicDiffusion2D) = energy_total(u, equations)
  
  
  # Calculate total energy for a conservative state `u`
  @inline function energy_total(u, equations::CouplingCompressibleEulerHyperbolicDiffusion2D)
    # energy function as found in equations (2.5.12) in the book "I Do Like CFD, Vol. 1"
    rho, rho_v1, rho_v2, rho_e, phi, q1, q2 = u
    return (rho_v1^2 + rho_v2^2) / (2 * rho) + 0.5 * (phi^2 + equations.Lr^2 * (q1^2 + q2^2))
  end
  
  
  # Calculate minimum and maximum wave speeds for HLL-type fluxes
  @inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
    equations::CouplingCompressibleEulerHyperbolicDiffusion2D)
    λ_min = 0
    λ_max = abs(equations.coupling_nu)

    return λ_min, λ_max
  end


  @inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CouplingCompressibleEulerHyperbolicDiffusion2D)

    λ_min = 0
    λ_max = abs(equations.coupling_nu)

    return λ_min, λ_max
  end

  end # @muladd
  