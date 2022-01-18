# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

  function limiter_rueda_gassner!(u, alpha, mesh::AbstractMesh{1}, integrator, semi, limiter!)
    @unpack solver,  equations, solver, cache = semi 
    @unpack element_ids_dgfv = semi.cache
    @unpack alpha_max = semi.solver.volume_integral.indicator
    @unpack beta, stage = limiter!
    @unpack semi_fv, tolerance, iterations_newton, u_safe, u_latest_stage,
            tmp_lates_stage, node_dg, node_tmp, du_dα, dp_du, alpha_max  = limiter!.cache
        

    # pure FV solution for stage s
    get_usafe!(u_safe, limiter!, integrator)
    u_safe = wrap_array(u_safe, semi_fv)
        
    # no need to calculate pure dg sol:
    #        u = (1-α) u_dg + α u_FV
    # <-> u_dg = (u- α u_FV) / (1-α)

    @threaded for element in  eachelement(solver, cache)
      # if alpha is already max, theres no need to correct
      if abs(alpha[element] - alpha_max) < tolerance
        continue
      end
      cor = 0.0

      # check for corrections in density
      for j in eachnode(solver)

        # safe solution
        ρ_safe = u_safe[1,j,element]
        if ρ_safe < 0
          error("safe value for density not safe")
        end
        α_p = beta * ρ_safe - u[1,j,element]

        # correct, if ap <= 0 is not fulfilled 
        if (α_p > tolerance)
          p_dg = (u[1,j,element] - alpha[element] * u_safe[1,j,element]) / (1 - alpha[element])

          # avoid divison by 0
          if abs(u_safe[1,j,element] - p_dg) < tolerance
            continue
          end

          tmp = α_p / ((u_safe[1,j,element] - p_dg))

          # Correction is calculated for each DOF.
          # For the element, the maximum correction is taken from all DOF.
          cor = max(cor, tmp)
        end
                
      end

      # Correct density
      if cor > 0        
        if alpha[element] + cor > alpha_max
          cor =  alpha_max - alpha[element]
          alpha[element] = alpha_max
        else
          alpha[element] += cor 
        end
        correct_u!(u, semi, element, u_safe, alpha, cor)      
      end




      # pressure 
      cor = 0.0
      for j in eachnode(solver)
        tmp = 0.0
        usafe_node = get_node_vars(u_safe, equations, solver, j, element)
        _, _, p_safe = cons2prim(usafe_node, equations)
        if p_safe < 0
          error("safe value for pressure not safe in stage $stage")
        end
        u_node = get_node_vars(u, equations, solver, j, element)
        _, v1, p_newton = cons2prim(u_node, equations) 
        α_p = beta * p_safe - p_newton
        if (α_p >  tolerance)

          # Newton's method
          # Calculate ∂p/∂α  using the chain rule
          # ∂p/∂α =  ∂p/∂u * ∂u/∂α
          for newton_stage in 1:iterations_newton
            # compute  ∂u/∂α
            for vars in eachvariable(equations)
              node_dg[vars] = (u[vars,j,element] - alpha[element] * u_safe[vars,j,element]) / (1 - alpha[element])
              du_dα[vars] =  (u_safe[vars,j,element] - node_dg[vars])
            end
                            
            # compute  ∂p/∂u 
            dp_du[1] = (equations.gamma - 1) * (0.5 * v1^2) 
            dp_du[2] = (equations.gamma - 1) * (-v1)
            dp_du[3] = (equations.gamma - 1)

            dp_dα = dot(dp_du, du_dα)
            
            # avoid divison by 0
            if abs(dp_dα) < tolerance
              continue 
            end

            tmp +=  α_p / dp_dα

            # calc corrected u in newton stage
            for vars in eachvariable(equations)
              node_tmp[vars]  = u[vars,j,element] + tmp * (u_safe[vars,j,element] - node_dg[vars])
            end

            # get new pressure value
            _, v1, p_newton = cons2prim(node_tmp, equations) 
            α_p = beta * p_safe - p_newton

            if α_p <= tolerance
              break
            end

            if newton_stage == iterations_newton
              error("Number of iterations ($iterations_newton) not enough to correct pressure")
            end
          end
           cor = max(cor, tmp)
        end
                
      end

      # Correct pressure
      if cor > 0
        if alpha[element] + cor > alpha_max
          cor =  alpha_max - alpha[element]
          alpha[element] = alpha_max
        else
          alpha[element] += cor 
        end
        correct_u!(u, semi, element, u_safe, alpha, cor)

      end                   
    end
    return nothing
  end

  # # For Carpenter-Kennedy Time Integration
  # function get_usafe!(u_safe, limiter!, integrator)
  #   @unpack stage = limiter!
  #   @unpack semi_fv, u_latest_stage, tmp_lates_stage  = limiter!.cache
  #   @unpack t, dt, uprev, f = integrator
  #   @unpack k, tmp = integrator.cache
  #   @unpack A2end, B1, B2end, c2end = integrator.cache.tab

  #   if stage == 1
  #     f(k, uprev, semi_fv, t)
  #     # tmp = dt * k
  #     for i in 1:length(uprev)
  #       u_safe[i] = uprev[i] + B1 * dt * k[i]
  #     end
  #   else
  #     if stage == 5 
  #       dt =limiter!.dt
  #       t = limiter!.t
  #     end
  #     f(k, u_latest_stage, semi_fv, t + c2end[stage-1] * dt)
  #     # tmp_lates_stage += dt * k
  #     for d in 1:length(u_safe)
  #       tmp_lates_stage[d] = tmp_lates_stage[d] + dt * k[d]
  #       u_safe[d] = u_latest_stage[d] + B2end[stage-1] * tmp_lates_stage[d]
  #     end
  #   end
  # end


  function correct_u!(u::AbstractArray{<:Any,3}, semi, element, u_safe, alpha, cor)
    @unpack solver, equations = semi
    for i in eachnode(solver)
      for vars in eachvariable(equations)
        sol_dg = (u[vars,i,element] - alpha[element] * u_safe[vars,i,element]) / (1 - alpha[element])
        u[vars,i,element] = u[vars,i,element] + cor * (u_safe[vars,i,element] - sol_dg)
      end            
    end 
  end


end # @muladd