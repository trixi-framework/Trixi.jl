# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""

Adaptive filter for DGMulti to ensure the well-posedness of the entropy
projection step.
Modify the input u to be the filtered solution û, such that the
entropy-projected conservative variables u(Πv(û)) satisfies some desired bound.
"""
# TODO: only support 1D GaussSBP. Multidimension and general DGMulti in future PRs.
function adaptive_filter!(u, filter,
                          mesh::DGMultiMesh{1},
                          equations::Union{AbstractCompressibleEulerEquations,CompressibleNavierStokesDiffusion2D},
                          dg::DGMulti{1}, cache)

  rd = dg.basis
  @unpack Vq = rd
  @unpack invVDM = filter.ops
  @unpack u_modal_coeffs = filter.cache
  @unpack u_values, entropy_var_values = cache

  # TODO: redundant operations with local_filtered_entropy_projection!
  apply_to_each_field(mul_by!(Vq), u_values, u)
  apply_to_each_field(mul_by!(invVDM), u_modal_coeffs, u)

  # TODO: redundant operations with local_filtered_entropy_projection!
  # TODO: only need the last entropy variable to determine the local bound
  # transform quadrature values to entropy variables
  cons2entropy!(entropy_var_values, u_values, equations)

  @threaded for e in eachelement(mesh, dg, cache)
    local_bound = calc_local_bound(filter, e, equations, cache)
    θ = calc_local_filter_factor!(cache, u, e, local_bound, filter, mesh, equations, dg)
    calc_local_filtered_cons_values!(filter, view(u, :, e), θ, e, mesh, equations, dg)
  end
end

"""

On element e, calculate and return the local density, internal energy and last
entropy variable relaxed bound using adaptive filter's relaxation factors.
"""
function calc_local_bound(filter, e,
                          equations::Union{AbstractCompressibleEulerEquations,CompressibleNavierStokesDiffusion2D},
                          cache)

  @unpack u_values, entropy_var_values = cache
  η = get_relaxation_factor_cons_var(filter)
  ζ = get_relaxation_factor_entropy_var(filter)

  u_values_e           = view(u_values, :, e)
  entropy_var_values_e = view(entropy_var_values, :, e)
  rho    = Base.Generator(u->density(u, equations), u_values_e)
  rhoe   = Base.Generator(u->energy_internal(u, equations), u_values_e)
  v_last = Base.Generator(v->v[end], entropy_var_values_e)

  return CompressibleFlowBound(ζ*maximum(v_last),
                               max((1.0-η)*minimum(rho), eps()) , (1.0+η)*maximum(rho),
                               max((1.0-η)*minimum(rhoe), eps()), (1.0+η)*maximum(rhoe))

end

"""

calculate and return the filter factor θ on element e, such that the filtered
entropy projected variables u(Πv(û(θ))) satisfy local_bound
"""
function calc_local_filter_factor!(cache, u, e, local_bound, filter::SecondOrderExponentialAdaptiveFilter, mesh, equations, dg)
  cond(θ) = calc_and_check_local_filtered_values!(cache, u, θ, e, local_bound, filter, mesh, equations, dg)
  return bisection_bound(cond, -log(eps()), 0.0)
end

"""

On element e, calculate filtered entropy projected variables u(Πv(û(θ))) and
check whether the filtered variables satisfy the local_bound.
Return boolean indicates whether the local_bound is satisfied.
"""
function calc_and_check_local_filtered_values!(cache, u, θ, e, local_bound, filter, mesh, equations, dg)

  try
    calc_local_filtered_values!(cache, view(u, :, e), θ, e, filter, mesh, equations, dg)
    return check_local_bound(e, local_bound, equations, dg, cache) 
  catch err
    # Catch negativity during entropy projection, else throw other errors
    isa(err, DomainError) ? false : throw(err)
  end

end

"""

On element e, calculate filtered entropy projected variables u(Πv(û(θ)))
"""
function calc_local_filtered_values!(cache, u_e, θ, e, filter, mesh, equations, dg)

  calc_local_filtered_cons_values!(filter, u_e, θ, e, mesh, equations, dg)
  local_entropy_projection!(cache, u_e, e, mesh, equations, dg)

end

"""

On element e, calculate filtered conservative variables û(θ) and set u_e = û(θ)
"""
function calc_local_filtered_cons_values!(filter, u_e, θ, e, mesh, equations, dg)

  rd = dg.basis
  @unpack VDM = rd
  @unpack local_u_modal_coeffs_threaded = filter.cache

  apply_local_filter!(filter, θ, e, mesh, dg)   # Update filter value into u_modal_coeffs_threaded
  apply_to_each_field(mul_by!(VDM), u_e, local_u_modal_coeffs_threaded[Threads.threadid()])

end

# TODO: Only support 1D. Dispatch on dimension in future PRs
"""

On element e, calculate modal coefficients of filtered conservative variables
û(θ) and put it in a temporary cache
"""
function apply_local_filter!(filter::SecondOrderExponentialAdaptiveFilter, θ, e, mesh::DGMultiMesh{1}, dg::DGMulti{1})
  
  @unpack u_modal_coeffs, local_u_modal_coeffs_threaded = filter.cache

  for i in each_mode(mesh, dg)
    local_u_modal_coeffs_threaded[Threads.threadid()][i] = exp(-θ*(i-1)*(i-1))*u_modal_coeffs[i, e]
  end

end

"""

Check whether entropy projected variables u(Πv(û)) stored in cache satisfy the
local_bound
"""
function check_local_bound(e, local_bound, equations, dg, cache)

  @unpack entropy_projected_u_values, entropy_var_values = cache

  rd = dg.basis
  entropy_var_face_values_e    = view(entropy_var_values, :, e)
  entropy_projected_u_values_e = view(entropy_projected_u_values, :, e)

  rho     = Base.Generator(u->density(u, equations), entropy_projected_u_values_e)
  rhoe    = Base.Generator(u->energy_internal(u, equations), entropy_projected_u_values_e)
  vf_last = Base.Generator(v->v[end], entropy_var_face_values_e)

  return (mapreduce((x -> x >= local_bound.ρmin)    , &, rho    )
       && mapreduce((x -> x <= local_bound.ρmax)    , &, rho    )
       && mapreduce((x -> x >= local_bound.ρemin)   , &, rhoe   )
       && mapreduce((x -> x <= local_bound.ρemax)   , &, rhoe   )
       && mapreduce((x -> x <= local_bound.vlastmax), &, vf_last))

end

# TODO: REFACTOR. Not so obvious on how to use Roots.jl in this case 
#####################
###   Utilities   ###
#####################

"""
Bisection algorithm to find θ closest to θ_invalid that cond(θ) == true.
Assuming cond(θ_valid) == true, and cond is a continuous function in the
interval between θ_valid and θ_invalid
"""
function bisection_bound(cond, θ_valid, θ_invalid)

  if cond(θ_invalid)
    return θ_invalid
  else
    maxit = 20      # Maximum number of nonlinear iterations, TODO: hide from users?
    for iter = 1:maxit
      θ_new = .5*(θ_valid+θ_invalid)
      cond(θ_new) ? θ_valid = θ_new : θ_invalid = θ_new
    end
    return θ_valid
  end

end

end # @muladd
