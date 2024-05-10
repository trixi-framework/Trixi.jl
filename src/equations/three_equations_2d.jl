# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
    @doc raw"""
      ThreeEquations2D(gamma)
        
      The three equations model for two phase flows.
        
      """
    struct ThreeEquations2D{RealT <: Real} <: AbstractThreeEquationsEquations{2, 6}
        gamma::RealT
        k0::RealT
        rho_0::RealT
        gravity::RealT

        function ThreeEquations2D(gamma, k0, rho_0, gravity)
            return new{typeof(gamma)}(gamma, k0, rho_0, gravity)
        end
    end

    have_nonconservative_terms(::ThreeEquations2D) = True()
    function varnames(::typeof(cons2cons), ::ThreeEquations2D)
        ("alpha_rho", "alpha_rho_v1", "alpha_rho_v2", "alpha", "phi", "dummy")
    end
    varnames(::typeof(cons2prim), ::ThreeEquations2D) = ("rho", "v1", "v2", "alpha", "phi", "dummy")
    varnames(::typeof(cons2entropy), ::ThreeEquations2D) = ("rho", "v1", "v2", "alpha", "phi", "dummy")

    n_nonconservative_terms(::ThreeEquations2D) = 1

    # Set initial conditions at physical location `x` for time `t`
    """
        initial_condition_constant(x, t, equations::ThreeEquations2D)
        
    A constant initial condition to test free-stream preservation.
    """
    function initial_condition_constant(x, t, equations::ThreeEquations2D)
        alpha_rho = 1000.0
        alpha_rho_v1 = 0.0
        alpha_rho_v2 = 0.0
        alpha = 1.0
        phi = x[2]
        return SVector(alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi,0.0)
    end

    function source_terms_gravity(u, x, t, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha = u
        du1 = 0.0

        period = 1.5
        amplitude = 0.05

        du2 = -alpha_rho * 4*pi^2*amplitude/period^2*cos(2*pi*t / period)
        # du3 = -alpha_rho * equations.gravity
        du3 = 0.0

        # if alpha > 2e-3
          # du2 = -alpha_rho * 5.0
          # du2 = 0.0
        # else
        #   du2 = 0.0
        #   du3 = 0.0
        # end

        # du4 = abs(min(alpha-1e-3,0.0))
        du4 = 0.0
        du5 = 0.0

        return SVector(du1, du2, du3, du4, du5,0.0)
    end

    # function boundary_condition_wall(u_inner, orientation,
    #                                  direction, x, t,
    #                                  surface_flux_function,
    #                                  equations::ThreeEquations2D)

    #     # Boundary state is equal to the inner state except for the velocity. For boundaries
    #     # in the -x/+x direction, we multiply the velocity in the x direction by -1.
    #     # Similarly, for boundaries in the -y/+y direction, we multiply the velocity in the
    #     # y direction by -1
    #     # if direction in (1, 2) # x direction
    #     if orientation == 1
    #         u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4],
    #                              u_inner[5])
    #     else # y direction
    #         u_boundary = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4],
    #                              u_inner[5])
    #     end

    #     # flux = surface_flux_function(u_inner, u_boundary, orientation, equations)

    #     # Calculate boundary flux
    #     if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    #         flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    #     else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    #         flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    #     end

    #     # println(direction)
    #     # println(surface_flux_function)
    #     # println(pressure(u_inner, equations))
    #     # println(pressure(u_boundary, equations))
    #     # println(u_inner)
    #     # println(u_boundary)
    #     # println(flux)
    #     # println("")

    #     return flux
    # end

    function boundary_condition_wall(u_inner, orientation,
                                     direction, x, t,
                                     surface_flux_function,
                                     equations::ThreeEquations2D)

        if string(surface_flux_function) == "flux_nonconservative_ThreeEquations_well"
          # Boundary state is equal to the inner state except for the velocity. For boundaries
          # in the -x/+x direction, we multiply the velocity in the x direction by -1.
          # Similarly, for boundaries in the -y/+y direction, we multiply the velocity in the
          # y direction by -1
          # if direction in (1, 2) # x direction
          # if orientation == 1
          #     u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4],
          #                          u_inner[5])
          # else # y direction
          #     u_boundary = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4],
          #                          u_inner[5])
          # end

          # # flux = surface_flux_function(u_inner, u_boundary, orientation, equations)

          # # Calculate boundary flux
          # if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
          #     flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
          # else # u_boundary is "left" of boundary, u_inner is "right" of boundary
          #     flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
          # end

          z = zero(eltype(u_inner))

          flux = SVector(z, z, z, z, z, z)

        else

          alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u_inner
          # v1 = alpha_rho_v1 / alpha_rho
          # v2 = alpha_rho_v2 / alpha_rho

          p = alpha * pressure(u_inner, equations)

          z = zero(eltype(u_inner))

          # Calculate boundary flux
          if orientation == 1
              flux = SVector(z, p, z, z, z, z)
          else
              flux = SVector(z, z, p, z, z, z)
          end

          # println(direction)
          # println(surface_flux_function)
          # println(Symbol(surface_flux_function))
          # println(pressure(u_inner, equations))
          # println(u_inner)
          # println(flux)
          # println("")

        end

        return flux
    end

    @inline function boundary_condition_wall(u_inner, normal_direction::AbstractVector,
                                             x, t,
                                             surface_flux_function,
                                             equations::ThreeEquations2D)
      
      # Normalize the outward pointing direction.
      normal = normal_direction / norm(normal_direction)
   
      # compute the normal velocity
      u_normal = normal[1] * u_inner[2] + normal[2] * u_inner[3]

      period = 1.5
      amplitude = 0.05

      # u_wall = u_inner[1] * 0.5 * sin(2*pi*t/1.0)
      # u_wall = u_inner[1] * 2*pi*amplitude/period*sin(2*pi*t / period)
      u_wall = 0.0
    
      # create the "external" boundary solution state
      u_boundary = SVector(u_inner[1],
                           u_inner[2] - 2.0 * u_normal * normal[1] + u_wall * abs(normal[1]),
                           u_inner[3] - 2.0 * u_normal * normal[2] + u_wall * abs(normal[2]),
                           u_inner[4],
                           u_inner[5],
                           u_inner[6])

      # println("u_inner = ")
      # display(u_inner)

      # println("u_boundary = ")
      # display(u_boundary)
    
      # calculate the boundary flux
      flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

      return flux
    end

    # @inline function boundary_condition_wall(u_inner, normal_direction::AbstractVector,
    #                                          x, t,
    #                                          surface_flux_function,
    #                                          equations::ThreeEquations2D)
    #   
    #   if string(surface_flux_function) == "flux_nonconservative_ThreeEquations_well"

    #     z = zero(eltype(u_inner))
    #     flux = SVector(z, z, z, z, z)
    #     return flux

    #   end

    #   alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u_inner

    #   normal = normal_direction / norm(normal_direction)
    #   # compute the normal velocity
    #   # u_normal = normal_direction[1] * u_inner[2] + normal_direction[2] * u_inner[3]
    # 
    #   p = alpha * pressure(u_inner, equations)
    #   z = zero(eltype(u_inner))

    #   flux = SVector(z, normal[1]*p, normal[2]*p, z, z)
   
    #   return flux
    # end

    # Calculate 1D flux for a single point
    @inline function flux(u, orientation::Integer, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi = u
        v1 = alpha_rho_v1 / alpha_rho
        v2 = alpha_rho_v2 / alpha_rho
        p = pressure(u, equations)
        if orientation == 1
            f1 = alpha_rho_v1
            f2 = alpha_rho_v1 * v1 + alpha * p
            f3 = alpha_rho_v1 * v2
            f4 = 0.0
            f5 = 0.0
            f6 = 0.0
        else
            f1 = alpha_rho_v2
            f2 = alpha_rho_v1 * v2
            f3 = alpha_rho_v2 * v2 + alpha * p
            f4 = 0.0
            f5 = 0.0
            f6 = 0.0
        end
        return SVector(f1, f2, f3, f4, f5, f6)
    end

    # Calculate 1D flux for a single point in the normal direction
    # Note, this directional vector is not normalized
    @inline function flux(u, normal_direction::AbstractVector, equations::ThreeEquations2D)
        rho, v1, v2, alpha, phi = cons2prim(u, equations)

        v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
        rho_v_normal = rho * v_normal

        p = pressure(u, equations)

        f1 = alpha * rho_v_normal
        f2 = alpha * rho_v_normal * v1 + alpha * p * normal_direction[1]
        f3 = alpha * rho_v_normal * v2 + alpha * p * normal_direction[2]
        f4 = 0.0
        f5 = 0.0
        f6 = 0.0

        return SVector(f1, f2, f3, f4, f5, f6)
    end

    @inline function flux_nonconservative_ThreeEquations(u_ll, u_rr, orientation::Integer,
                                                  equations::ThreeEquations2D)
        v1_ll = u_ll[2] / u_ll[1]
        v2_ll = u_ll[3] / u_ll[1]
        alpha_rr = u_rr[4]

        z = zero(eltype(u_ll))

        if orientation == 1
            f = SVector(z, z, z, v1_ll * alpha_rr, z, z)
        else
            f = SVector(z, z, z, v2_ll * alpha_rr, z, z)
        end

        return f
    end

    @inline function flux_nonconservative_ThreeEquations(u_ll, u_rr,
                                                  normal_direction_ll::AbstractVector,
                                                  normal_direction_average::AbstractVector,
                                                  equations::ThreeEquations2D)
        v1_ll = u_ll[2] / u_ll[1]
        v2_ll = u_ll[3] / u_ll[1]
        alpha_rr = u_rr[4]

        v_dot_n_ll = v1_ll * normal_direction_ll[1] + v2_ll * normal_direction_ll[2]

        z = zero(eltype(u_ll))

        f = SVector(z, z, z, v_dot_n_ll * alpha_rr, z, z)

        return f
    end

    @inline function flux_nonconservative_ThreeEquations_well(u_ll, u_rr, orientation::Integer,
                                                       equations::ThreeEquations2D)
        v1_ll = u_ll[2] / u_ll[1]
        v2_ll = u_ll[3] / u_ll[1]
        alpha_rr = u_rr[4]
        phi_ll = u_ll[5]
        phi_rr = u_rr[5]

        well_balanced = -u_ll[1] / equations.rho_0 * equations.k0 *
                        exp( equations.rho_0 * phi_ll / equations.k0) *
                        exp(-equations.rho_0 * phi_rr / equations.k0)

        z = zero(eltype(u_ll))

        if orientation == 1
            f = SVector(z, well_balanced, z, v1_ll * alpha_rr, z, z)
        else
            f = SVector(z, z, well_balanced, v2_ll * alpha_rr, z, z)
        end

        return f
    end

    @inline function flux_nonconservative_ThreeEquations_well(u_ll, u_rr,
                                                       normal_direction_ll::AbstractVector,
                                                       normal_direction_average::AbstractVector,
                                                       equations::ThreeEquations2D)
        v1_ll = u_ll[2] / u_ll[1]
        v2_ll = u_ll[3] / u_ll[1]
        alpha_rr = u_rr[4]

        phi_ll = u_ll[5]
        phi_rr = u_rr[5]

        v_dot_n_ll = v1_ll * normal_direction_ll[1] + v2_ll * normal_direction_ll[2]

        # Non well-balanced force contribution.
        force = -u_ll[1] * phi_rr
         
        # Well-balanced force contribution.
        # force = -u_ll[1] * (equations.k0/equations.rho_0) *
        #                 exp(-equations.rho_0/equations.k0 * phi_ll) *
        #                 exp( equations.rho_0/equations.k0 * phi_rr)

        z = zero(eltype(u_ll))
        f = SVector(z, force * normal_direction_average[1], force * normal_direction_average[2], v_dot_n_ll * alpha_rr, z, z)

        return f
    end

    # Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
    # maximum velocity magnitude plus the maximum speed of sound
    @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                         equations::ThreeEquations2D)
        rho_ll, v1_ll, v2_ll, alpha_ll, phi_ll, dummy_ll = cons2prim(u_ll, equations)
        rho_rr, v1_rr, v2_rr, alpha_rr, phi_rr, dummy_rr = cons2prim(u_rr, equations)

        # Get the velocity value in the appropriate direction
        if orientation == 1
            v_ll = v1_ll
            v_rr = v1_rr
        else # orientation == 2
            v_ll = v2_ll
            v_rr = v2_rr
        end
        # Calculate sound speeds
        c_ll = sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                    (rho_ll / equations.rho_0)^(equations.gamma - 1))
        c_rr = sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                    (rho_rr / equations.rho_0)^(equations.gamma - 1))

        λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
    end

    @inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::ThreeEquations2D)
        rho_ll, v1_ll, v2_ll, alpha_ll, phi_ll, dummy_ll = cons2prim(u_ll, equations)
        rho_rr, v1_rr, v2_rr, alpha_rr, phi_rr, dummy_rr = cons2prim(u_rr, equations)

        # Calculate normal velocities and sound speed
        # left
        v_ll = (v1_ll * normal_direction[1]
                +
                v2_ll * normal_direction[2])
        c_ll = sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                    (rho_ll / equations.rho_0)^(equations.gamma - 1))
        # right
        v_rr = (v1_rr * normal_direction[1]
                +
                v2_rr * normal_direction[2])
        c_rr = sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                    (rho_rr / equations.rho_0)^(equations.gamma - 1))

        return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction)
    end

    # Calculate minimum and maximum wave speeds for HLL-type fluxes
    @inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                         equations::ThreeEquations2D)
        rho_ll, v1_ll, v2_ll, alpha_ll, phi_ll, dummy_ll = cons2prim(u_ll, equations)
        rho_rr, v1_rr, v2_rr, alpha_rr, phi_rr, dummy_rr = cons2prim(u_rr, equations)

        if orientation == 1 # x-direction
            λ_min = v1_ll - sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                         (rho_ll / equations.rho_0)^(equations.gamma - 1))
            λ_max = v1_rr + sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                         (rho_rr / equations.rho_0)^(equations.gamma - 1))
        else # y-direction
            λ_min = v2_ll - sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                         (rho_ll / equations.rho_0)^(equations.gamma - 1))
            λ_max = v2_rr + sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                         (rho_rr / equations.rho_0)^(equations.gamma - 1))
        end

        return λ_min, λ_max
    end

    @inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::ThreeEquations2D)
        rho_ll, v1_ll, v2_ll, alpha_ll, phi_ll, dummy_ll = cons2prim(u_ll, equations)
        rho_rr, v1_rr, v2_rr, alpha_rr, phi_rr, dummy_rr = cons2prim(u_rr, equations)

        v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
        v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

        norm_ = norm(normal_direction)
        # The v_normals are already scaled by the norm
        λ_min = v_normal_ll -
                sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                     (rho_ll / equations.rho_0)^(equations.gamma - 1)) * norm_
        λ_max = v_normal_rr +
                sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                     (rho_rr / equations.rho_0)^(equations.gamma - 1)) * norm_

        return λ_min, λ_max
    end

    @inline function max_abs_speeds(u, equations::ThreeEquations2D)
        rho, v1, v2, alpha, phi, dummy = cons2prim(u, equations)
        c = sqrt(equations.gamma * (equations.k0 / equations.rho_0) *
                 (rho / equations.rho_0)^(equations.gamma - 1))

        return abs(v1) + c, abs(v2) + c
    end

    # Convert conservative variables to primitive
    @inline function cons2prim(u, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi, dummy = u

        rho = alpha_rho / alpha
        v1 = alpha_rho_v1 / alpha_rho
        v2 = alpha_rho_v2 / alpha_rho

        return SVector(rho, v1, v2, alpha, phi, dummy)
    end

    # Convert conservative variables to primitive
    @inline function cons2entropy(u, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi, dummy = u

        rho = alpha_rho / alpha
        v1 = alpha_rho_v1 / alpha_rho
        v2 = alpha_rho_v2 / alpha_rho

        phi = max(1e-6, max(max_abs_speeds(u, equations)...) )

        # vn = sqrt(v1*v1 + v2*v2)
        # vn = log10(sqrt(alpha_rho_v1^2 + alpha_rho_v2^2))
        vn = log10(max(1e-6, sqrt(v1^2 + v2^2)))

        alpha_log = log10(max(1e-6, alpha))
        p = pressure(u, equations)

        # return SVector(alpha_rho, vn, log10(alpha), alpha, phi)
        # return SVector(alpha_rho, vn, alpha_log, alpha, phi)
        return SVector(alpha_rho, p, alpha_log, alpha, phi, dummy)
    end

    # Convert primitive to conservative variables
    @inline function prim2cons(prim, equations::ThreeEquations2D)
        rho, v1, v2, alpha, phi, dummy = prim
        alpha_rho = rho * alpha
        alpha_rho_v1 = alpha_rho * v1
        alpha_rho_v2 = alpha_rho * v2
        return SVector(alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi, dummy)
    end

    @inline function isvalid(u, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi, dummy = u
        # return alpha_rho > 0.0 && alpha > 0.0 && alpha <= 1.0
        return alpha_rho > 0.0 && alpha > 0.0 # && alpha <= 1.0
    end

    @inline function density(u, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi, dummy = u
        return alpha_rho / alpha
    end

    @inline function alpha_rho(u, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi, dummy = u
        return alpha_rho
    end

    @inline function alpha(u, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi, dummy = u
        return alpha
    end

    @inline function pressure(u, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi, dummy = u
        # return alpha * equations.k0 * ((alpha_rho / alpha / equations.rho_0)^(equations.gamma) - 1)
        return equations.k0 * ((alpha_rho / alpha / equations.rho_0)^(equations.gamma) - 1)
    end

    @inline function density_pressure(u, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi, dummy = u
        rho = alpha_rho / alpha
        rho_times_p = pressure(u, equations) * rho
        return rho_times_p
    end

    # Calculate the error for the "water-at-rest" test case 
    @inline function water_at_rest_error(u, equations::ThreeEquations2D)
        alpha_rho, alpha_rho_v1, alpha_rho_v2, alpha, phi, dummy = u
        rho0 = equations.rho_0 *
               exp(-(equations.gravity * equations.rho_0 / equations.k0) * (phi - 1.0))
        return abs(alpha_rho / alpha - rho0)
    end
end # @muladd  
