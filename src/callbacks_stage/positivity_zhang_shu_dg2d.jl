# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function limiter_zhang_shu!(u, threshold::Real, variable,
                            mesh::AbstractMesh{2}, equations, dg::DGSEM, cache)
  @unpack weights = dg.basis

  # Beta Correction Factor
  beta = 0.1

  #println("here")

  for element in eachelement(dg, cache)

    alpha_old = cache.elements.alpha[element]

    if abs(cache.elements.alpha[element] - 1.0) < eps(1.)
      #println("alpha: ",cache.elements.alpha[element])
      #println("abbruch?!")
      continue # Abbruch, hier läuft schon pures FV
    end

    for j in eachnode(dg), i in eachnode(dg)

      cache.elements.FFV_m_FDG[:,i,j,element] = cache.elements.FFV_m_FDG[:,i,j,element] * (-cache.elements.inverse_jacobian[element])

      # Berechne "sichere" FV-Lösung 
      for v in eachvariable(equations)
        cache.elements.u_safe[v,i,j,element] = u[v,i,j,element] + equations.delta_t * (cache.elements.FFV_m_FDG[v,i,j,element]) * (1.0 - cache.elements.alpha[element])
      end 

      # Berechne sicheren Druck aus der sicheren Lösung
      p_safe = pressure(cache.elements.u_safe[:,i,j,element], equations)
  
      if p_safe < 0.
        println("no hope")
        return # Negativer Druck mit FV? Let's goo
      end

      # Berechne sichere Dichte und aktuelle Dichte
      rho_safe = density(cache.elements.u_safe[:,i,j,element], equations)

      if rho_safe < 0
        println("no hope 2")
        return # Negative Dichte mit FV? No hope!
      end
    end

    # Initialisiere Korrektur bei -0
    corr = - eps(100.)

    for j in eachnode(dg), i in eachnode(dg)

      # Dichte Korrektur
      rho_safe = density(cache.elements.u_safe[:,i,j,element], equations)
      rho = density(u[:,i,j,element], equations)
      a = (beta * rho_safe - rho) 

      a_comp = SVector{ncomponents(equations), real(equations)}(beta * cache.elements.u_safe[v+3,i,j,element] - u[v+3,i,j,element] for v in eachcomponent(equations))
 
      if a > 0. || any(a_comp .> 0) # This DOF needs a correction
        if abs(density(cache.elements.FFV_m_FDG[:,i,j,element], equations)) < eps(100.)
          #println("FFV_m_FDG: ",abs(density(cache.elements.FFV_m_FDG[:,i,j,element], equations)))
          continue # It wouldn't help :( )
        end
        corr_help = a / (density(cache.elements.FFV_m_FDG[:,i,j,element], equations))
        corr_help_comp = SVector{ncomponents(equations), real(equations)}(a_comp[v] / cache.elements.FFV_m_FDG[v+3,i,j,element] for v in eachcomponent(equations))
        corr = max(corr, corr_help, maximum(corr_help_comp))#, corr_help1, corr_help2)
        #println("corr: ",corr)
      end
    end

    if corr > 0.
      alphacont = cache.elements.alpha[element]
      cache.elements.alpha[element] = cache.elements.alpha[element] + corr * 1.0/equations.delta_t
      if cache.elements.alpha[element] > 1.0
        cache.elements.alpha[element] = 1.0
        corr = (1.0 - alphacont) * equations.delta_t
      end 
      #println("rho_old: ",density(u[:,2,2,element],equations))
      #println("corr: ",corr)
      #println("FFVmFDG: ",cache.elements.FFV_m_FDG[4,1,1,element]+cache.elements.FFV_m_FDG[5,1,1,element])
      for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)
        set_node_vars!(u, u_node + corr * cache.elements.FFV_m_FDG[:,i,j,element],
                       equations, dg, i, j, element)
      end 
      #println("rho_new: ",density(u[:,2,2,element],equations))
    end 

    # correct pressure
    corr = -eps(100.)

    notInIter = false 

    for j in eachnode(dg), i in eachnode(dg)
      
      pres = pressure(u[:,i,j,element], equations)

      p_safe = pressure(cache.elements.u_safe[:,i,j,element], equations)

      p_goal = beta * p_safe

      ap = p_goal - pres

      if ap <= 0.0 
        continue
      end 

      u_curr = u[:,i,j,element]
      corr1  = 0.0
      
      dp_du = zeros(nvariables(equations))
      for iter in 1:10
        dp_du = dpdu(u_curr, equations)
        dp_dalpha = dot(dp_du, cache.elements.FFV_m_FDG[:,i,j,element])
        if abs(dp_dalpha) < eps(100.)
          #println("noo")
          break
        end 

        #println("dp_dalpha: ",u_curr)

        #println("corr1 before: ",corr1)
        corr1 = corr1 + ap / dp_dalpha 
        #println("corr1 after: ",corr1)

        u_curr = u[:,i,j,element] + corr1 * cache.elements.FFV_m_FDG[:,i,j,element] 

        pres = pressure(u_curr, equations) 

        ap = p_goal - pres 

        if ap <= eps(100.0) && ap > -eps(Float32) * p_goal 
          break 
        end
        #println("still here after hmm")
      end 
      corr = max(corr, corr1)
      #println("corr: ",corr)

      #println("p: ",pres)
      #println("p_safe: ",p_safe)
    end 

    if corr > 0.0
      alphacont = cache.elements.alpha[element]
      cache.elements.alpha[element] = cache.elements.alpha[element] + corr * 1/equations.delta_t
      if cache.elements.alpha[element] > 1.0
        cache.elements.alpha[element] = 1.0
        corr = (1.0 - alpha_cont) * equations.delta_t
      end 
      #println("p_old: ",pressure(u[:,2,2,element],equations))
      #println("corr: ",corr)
      #println("FFVmFDG: ",cache.elements.FFV_m_FDG[4,1,1,element]+cache.elements.FFV_m_FDG[5,1,1,element])
      for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)
        set_node_vars!(u, u_node + corr * cache.elements.FFV_m_FDG[:,i,j,element],
                       equations, dg, i, j, element)
      end 
      #println("p_new: ",pressure(u[:,2,2,element],equations))
    end 

  end

  return nothing
end


end # @muladd
