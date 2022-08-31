# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


    @doc raw"""
        LinearizedEulerEquations2D(rho, lambda)
    
    The linearized Euler equations
    ```math
    \partial t
    \begin{pmatrix}
      v_1 \\ v_2 \\ p
    \end{pmatrix}
    +
    \partial x
    \begin{pmatrix}
      \frac{1}{\rho} p \\ 0 \\ \lambda v1
    \end{pmatrix}
    +
    \partial y
    \begin{pmatrix}
      0 \\ \frac{1}{\rho} p \\ \lambda v2
    \end{pmatrix}
    =
    \begin{pmatrix}
    0 \\ 0 \\ 0
    \end{pmatrix}
    ```
    for an ideal acoustic wave with constant density ``\rho`` and second lame constant ``\lambda``
    Here, ``p`` is the pressure and ``v_1``,``v_2`` are the velocities.
    """
    struct LinearizedEulerEquations2D{RealT<:Real} <: AbstractLinearizedEulerEquations2D{2, 3}
      rho::RealT
      lambda::RealT
    end
    
    function LinearizedEulerEquations2D(rho::Real, lambda::Real)
      return LinearizedEulerEquations2D(rho, lambda)
    end
    
    function LinearizedEulerEquations2D(; rho::Real, lambda::Real)
      return LinearizedEulerEquations2D(rho, lambda)
    end
    
    varnames(::typeof(cons2prim), ::LinearizedEulerEquations2D) = ("v1", "v2", "p")
    varnames(::typeof(cons2cons), ::LinearizedEulerEquations2D) = ("v1", "v2", "p")
     
      
    # Set initial conditions at physical location `x` for time `t`
    """
    initial_condition_constant(x, t, equations::LinearizedEulerEquations2D)
    
    A constant initial condition.
    """
    function initial_condition_constant(x, t, equations::LinearizedEulerEquations2D)
        v1 = 0.0
        v2 = 0.0
        p = 0.0
        return SVector(v1, v2, p)
    end
    
    """
        initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations2D)
    
    A smooth initial condition used for convergence tests in combination with
    [`source_terms_convergence_test`](@ref).
    """
    function initial_condition_convergence_test(x, t, equations::LinearizedEulerEquations2D)
      v1 = 2 + sin(2* pi * (x[1] - t))
      v2 = 4 + cos(2* pi * (x[2] - t))
      p = 6 + sin(2 * pi * (x[1] - t)) + cos(2 * pi * (x[2] - t))
      
      return SVector(v1, v2, p)
    end
    
    """
      source_terms_convergence_test(u, x, t, equations::LinearizedEulerEquations2D)
    
    Source terms used for convergence tests in combination with
    [`initial_condition_convergence_test`](@ref).
    """
    function source_terms_convergence_test(u, x, t, equations::LinearizedEulerEquations2D)
      c = equations.lambda * 2 * pi
      a = 1/equations.rho * 2 * pi
      omega = 2 * pi
    
      si = sin(omega * (x[2] - t))
      co = cos(omega * (x[1] - t))
    
      du1 = -omega * co + a * co
      du2 = omega * si - a * si
      du3 = -omega * co + omega * si + c * co - c * si
    
      return SVector(du1, du2, du3)
    end
    
    
    # Calculate 1D flux for a single point
    @inline function flux(u, orientation::Integer, equations::LinearizedEulerEquations2D)
      v1, v2, p = u
    
      if orientation == 1
        f1 = 1/equations.rho * p
        f2 = 0.0
        f3 = equations.lambda * v1
      else
        f1 = 0.0
        f2 = 1/equations.rho * p
        f3 = equations.lambda * v2
      end
      
      return SVector(f1, f2, f3)
    end
    
    
    # Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
    @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::LinearizedEulerEquations2D)
      v_ll = u_ll[orientation]
      v_rr = u_rr[orientation]
    
      λ_max = sqrt(equations.lambda/equations.rho)
    end
    
    
    # Calculate 1D flux for a single point in the normal direction
    # Note, this directional vector is not normalized
    @inline function flux(u, normal_direction::AbstractVector, equations::LinearizedEulerEquations2D)
      v1, v2, p = cons2prim(u, equations)
    
      f1 = normal_direction[1] * (1/equations.rho * p)
      f2 = normal_direction[2] * (1/equations.rho * p)
      f3 = ( normal_direction[1] * (equations.lambda * v1)
           + normal_direction[2] * (equations.lambda * v2))
    
      return SVector(f1, f2, f3)
    end
    
    
    # Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
    @inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::LinearizedEulerEquations2D)
        
        λ_max = sqrt(equations.lambda/equations.rho) * norm(normal_direction)
    end
      
    # Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the mean values
    @inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr, orientation_or_normal_direction, 
                                                                  equations::LinearizedEulerEquations2D)
        λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)
        diss = -0.5 * λ * (u_rr - u_ll)
        return SVector(diss[1], diss[2], diss[3])
    end
      
    
    @inline function max_abs_speeds(u, equations::LinearizedEulerEquations2D)
        v1, v2, p = u
      
        return abs(v1), abs(v2)
    end
      
      
    # Convert conservative variables to primitive
    @inline function cons2prim(u, equations::LinearizedEulerEquations2D)
        return SVector(u[1], u[2], u[3])
    end
      
      
    # Convert conservative variables to entropy
    @inline cons2entropy(u, equations::LinearizedEulerEquations2D) = u
        
      
    # Convert primitive to conservative variables
    @inline function prim2cons(prim, equations::LinearizedEulerEquations2D)
        v1, v2, p = prim
        return SVector(v1, v2, p)
    end
      
      
    @inline function pressure(u, equations::LinearizedEulerEquations2D)
       p = u[3]
       return p
    end
    
    
    end # @muladd
    