function calc_gradient_c_mean_square!(grad_c_mean_sq, u, mesh, equations::AcousticPerturbationEquations2D,
                                      dg::DG, cache)
  @unpack derivative_matrix = dg.basis

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      c_mean_sq_x = 0.0 # partial derivative of c_mean square in x direction on the ref. element
      for ii in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, ii, j, element)
        c_mean = u_node[6]
        c_mean_sq_x += derivative_matrix[i, ii] * c_mean^2
      end
      grad_c_mean_sq[1, i, j, element] = c_mean_sq_x * cache.elements.inverse_jacobian[element]

      c_mean_sq_y = 0.0 # partial derivative of c_mean square in y direction on the ref. element
      for jj in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, jj, element)
        c_mean = u_node[6]
        c_mean_sq_y += derivative_matrix[j, jj] * c_mean^2
      end
      grad_c_mean_sq[2, i, j, element] = c_mean_sq_y * cache.elements.inverse_jacobian[element]
    end
  end

  return nothing
end


function calc_acoustic_sources!(acoustic_source_terms, u_euler, u_ape, vorticity_mean,
                                source_region, weights, mesh, equations, dg::DG, cache)
  @unpack derivative_matrix = dg.basis
  @unpack node_coordinates = cache.elements

  acoustic_source_terms .= zero(eltype(acoustic_source_terms))

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      # Only calculate sources on nodes that lie within the acoustic source region
      if source_region(x)
        # Calculate vorticity
        dv2_dx = 0.0
        for ii in eachnode(dg)
          # TODO: Use get_node_vars instead?
          v2 = u_euler[3, ii, j, element] / u_euler[1, ii, j, element]
          dv2_dx += derivative_matrix[i, ii] * v2
        end

        dv1_dy = 0.0
        for jj in eachnode(dg)
          v1 = u_euler[2, i, jj, element] / u_euler[1, i, jj, element]
          dv1_dy += derivative_matrix[j, jj] * v1
        end

        vorticity = (dv2_dx - dv1_dy) * cache.elements.inverse_jacobian[element]

        prim_euler = cons2prim(get_node_vars(u_euler, equations, dg, i, j, element), equations)
        v1 = prim_euler[2]
        v2 = prim_euler[3]
        v1_mean = u_ape[4, i, j, element]
        v2_mean = u_ape[5, i, j, element]

        vorticity_prime = vorticity - vorticity_mean[i, j, element]
        v1_prime = v1 - v1_mean
        v2_prime = v2 - v2_mean

        acoustic_source_terms[1, i, j, element] -= -vorticity_prime * v2_mean - vorticity_mean[i, j, element] * v2_prime
        acoustic_source_terms[2, i, j, element] -=  vorticity_prime * v1_mean + vorticity_mean[i, j, element] * v1_prime

        # Apply acoustic source weighting function
        acoustic_source_terms[1, i, j, element] *= weights(x)
        acoustic_source_terms[2, i, j, element] *= weights(x)
      end
    end
  end

  return nothing
end


#####
##### Functions for calculating analytical sources for the co-rotating vortex pair elixir
#####
# Analytical flow solution
function velocity(x, t, vortex_pair)
  @unpack r0, rc, c0, circulation = vortex_pair

  omega = circulation / (4 * pi * r0^2)
  si, co = sincos(omega * t)
  b = SVector(r0 * co, r0 * si)
  z_plus = x - b
  z_minus = x + b

  r_plus = norm(z_plus)
  r_minus = norm(z_minus)
  theta_plus = atan(z_plus[2], z_plus[1])
  theta_minus = atan(z_minus[2], z_minus[1])
  si_plus, co_plus = sincos(theta_plus)
  si_minus, co_minus = sincos(theta_minus)

  v1 = -circulation/(2 * pi) * ( r_plus /(rc^2 + r_plus^2)  * si_plus +
                                r_minus/(rc^2 + r_minus^2) * si_minus)
  v2 = circulation/(2 * pi) * ( r_plus /(rc^2 + r_plus^2)  * co_plus +
                               r_minus/(rc^2 + r_minus^2) * co_minus )

  return SVector(v1, v2)
end

function vorticity(x, t, vortex_pair)
  J = ForwardDiff.jacobian(x -> velocity(x, t, vortex_pair), x)
  return J[2, 1] - J[1, 2]
end

# Cross product of the form [0, 0, w] × [v..., 0]
# Third component of such product is always zero, only the first two components are returned
function crossproduct(w::Real, v::AbstractVector)
  return SVector(-w * v[2], w * v[1])
end

# Acoustic source term based on the linearized Lamb vector of the analytical flow solution
function acoustic_source_term(u, x, t, vorticity_mean, vortex_pair)
  v_mean = SVector(u[4], u[5])
  #v_prime = SVector(u[1], u[2])
  v = velocity(x, t, vortex_pair)
  v_prime = v - v_mean

  w_mean = vorticity_mean
  w = vorticity(x, t, vortex_pair)
  w_prime = w - w_mean

  q_m = -crossproduct(w_prime, v_mean) - crossproduct(w_mean, v_prime)
  z = zero(eltype(u))

  return SVector(q_m[1], q_m[2], z, z, z, z, z)
end

function calc_analytical_acoustic_sources!(acoustic_source_terms, u_ape, t, vorticity_mean, vortex_pair,
                                           acoustic_source_radius, mesh, equations, dg::DG, cache)
  @unpack node_coordinates = cache.elements

  acoustic_source_terms .= zero(eltype(acoustic_source_terms))

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x = get_node_coords(node_coordinates, equations, dg, i, j, element)
      # Only calculate sources on nodes that lie within the acoustic source radius
      if sum(x.^2) <= acoustic_source_radius^2
        u_ape_node = get_node_vars(u_ape, equations, dg, i, j, element)
        s = acoustic_source_term(u_ape_node, x, t, vorticity_mean[i, j, element], vortex_pair)

        acoustic_source_terms[1, i, j, element] = s[1]
        acoustic_source_terms[2, i, j, element] = s[2]
      end
    end
  end

  return nothing
end