@muladd begin

  struct CouplingPolytropicEuler2D{RealT<:Real} <: AbstractCouplingEquations{2, 6}
    coupling_nu::RealT     # diffusion constant at the coupling interface
  end
  

  function CouplingPolytropicEuler2D(u::NTuple{6,<:Real})
    CouplingPolytropicEuler2D(SVector(u))
  end
  

  function CouplingPolytropicEuler2D(u_polytropic_euler_A::NTuple{4,<:Real}, u_polytropic_euler_B::NTuple{3,<:Real})
    CouplingPolytropicEuler2D(vcat(u_polytropic_euler_A, u_polytropic_euler_B))
  end
  

  varnames(::typeof(cons2cons), ::CouplingPolytropicEuler2D) = ("rho", "rho_v1", "rho_v2", "rho", "rho_v1", "rho_v2")
  varnames(::typeof(cons2prim), ::CouplingPolytropicEuler2D) = ("rho", "v1", "v2", "rho", "v1", "v2")

  default_analysis_errors(::CouplingPolytropicEuler2D) = (:l2_error, :linf_error, :residual)
  

  @inline function residual_steady_state(du, ::CouplingPolytropicEuler2D)
    abs(du[1])
  end
   

  # Calculate 1D flux in for a single point
  @inline function flux(u, orientation::Integer, equations::CouplingPolytropicEuler2D)
    rho_a, rho_v1_a, rho_v2_a, rho_b, rho_v1_b, rho_v2_b = u
    v1_a = rho_v1_a / rho_a
    v2_a = rho_v2_a / rho_a
    v1_b = rho_v1_b / rho_b
    v2_b = rho_v2_b / rho_b
  
    if orientation == 1
      f1 = rho_b
      f2 = v1_b
      f3 = v2_b
      f4 = rho_a
      f5 = v1_a
      f6 = v2_a
    else
      f1 = rho_b
      f2 = v1_b
      f3 = v2_b
      f4 = rho_a
      f5 = v1_a
      f6 = v2_a
    end
  
    return SVector(f1, f2, f3, f4, f5, f6)
  end
  

  # Note, this directional vector is not normalized
  @inline function flux(u, normal_direction::AbstractVector, equations::CouplingPolytropicEuler2D)
    rho_a, rho_v1_a, rho_v2_a, rho_b, rho_v1_b, rho_v2_b = u
    rho_a, v1_a, v2_a, rho_b, v1_b, v2_b = cons2prim(u, equations)

    v_normal_a = v1_a * normal_direction[1] + v2_a * normal_direction[2]
    v_normal_b = v1_b * normal_direction[1] + v2_b * normal_direction[2]

    f1 = rho_b
    f2 = v1_b
    f3 = v2_b
    f4 = rho_a
    f5 = v1_a
    f6 = v2_a

    return SVector(f1, f2, f3, f4, f5, f6) 
  end
  
  
  # Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
  @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CouplingPolytropicEuler2D)
    return equations.coupling_nu
  end
  

  @inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::CouplingPolytropicEuler2D)
    # rho_ll, rho_v1_ll, rho_v2_ll, rho_rr, rho_v1_rr, rho_v2_rr = (u_ll + u_rr)
    # rho, v1, v2, p, phi, q1, q2 = (cons2prim(u_ll, equations) + cons2prim(u_rr, equations))/2

    # v = (q1 * normal_direction[1] + q2 * normal_direction[2])

    # @infiltrate

    # Polytropic speed of sound is c = sqrt(gamma kappa rho^(gamma-1)).
    return 1.0
  end
  
   
  @inline have_constant_speed(::CouplingPolytropicEuler2D) = Val(true)
  

  @inline function max_abs_speeds(eq::CouplingPolytropicEuler2D)
    λ = eq.nu
    return 1.0, 1.0
  end
  
  
  # Convert conservative variables to primitive
  @inline cons2prim(u, equations::CouplingPolytropicEuler2D) = u
  

  # Calculate entropy for a conservative state `u` (here: same as total energy)
  @inline entropy(u, equations::CouplingPolytropicEuler2D) = energy_total(u, equations)
  
  
  # Calculate total energy for a conservative state `u`
  @inline function energy_total(u, equations::CouplingPolytropicEuler2D)
    # energy function as found in equations (2.5.12) in the book "I Do Like CFD, Vol. 1"
    rho_a, rho_v1_a, rho_v2_a, rho_b, rho_v1_b, rho_v2_b = u
    return (rho_v1_a^2 + rho_v2_a^2 + rho_v1_b^2 + rho_v2_b^2) / (2 * rho_a * rho_b)
  end
  
  
  # Calculate minimum and maximum wave speeds for HLL-type fluxes
  @inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
    equations::CouplingPolytropicEuler2D)
    λ_min = 0
    λ_max = abs(equations.coupling_nu)

    return 1.0, 1.0
  end


  @inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
    equations::CouplingPolytropicEuler2D)

    λ_min = 0
    λ_max = abs(equations.coupling_nu)

    return 1.0, 1.0
  end

  end # @muladd
  