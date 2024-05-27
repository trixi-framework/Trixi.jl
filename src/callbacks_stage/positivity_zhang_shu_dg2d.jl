# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function limiter_zhang_shu!(u, threshold::Real, variable,
                            mesh::AbstractMesh{2}, equations, dg::DGSEM, cache)
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    @threaded for element in eachelement(dg, cache)
        # determine minimum value
        value_min = typemax(eltype(u))
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            value_min = min(value_min, variable(u_node, equations))
        end

        # detect if limiting is necessary
        value_min < threshold || continue

        # compute mean value
        u_mean = zero(get_node_vars(u, equations, dg, 1, 1, element))
        total_volume = zero(eltype(u))
        for j in eachnode(dg), i in eachnode(dg)
            volume_jacobian = abs(inv(get_inverse_jacobian(inverse_jacobian, mesh,
                                                           i, j, element)))
            u_node = get_node_vars(u, equations, dg, i, j, element)
            u_mean += u_node * weights[i] * weights[j] * volume_jacobian
            total_volume += weights[i] * weights[j] * volume_jacobian
        end
        # normalize with the total volume
        u_mean = u_mean / total_volume

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                           equations, dg, i, j, element)
        end
    end

    return nothing
end
function perform_idp_correction!(u, dt, mesh::TreeMesh2D, equations::JinXinCompressibleEulerEquations2D, dg, cache)

            # relaxation parameter
            eps = equations.eps_relaxation
            dt_ = dt
            factor =1.0/ (eps + dt_)
            eq_relax = equations.equations_relaxation

 @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            # compute compressible Euler fluxes
            u_cons = SVector(u_node[1], u_node[2], u_node[3], u_node[4])
            vu = flux(u_cons,1,eq_relax)
            wu = flux(u_cons,2,eq_relax)
            # compute relaxation terms
            du1 = u_node[1]
            du2 = u_node[2]
            du3 = u_node[3]
            du4 = u_node[4]
            du5 = factor * (eps * u_node[5] + dt_ * vu[1])
            du6 = factor * (eps * u_node[6] + dt_ * vu[2])
            du7 = factor * (eps * u_node[7] + dt_ * vu[3])
            du8 = factor * (eps * u_node[8] + dt_ * vu[4])
            du9 = factor * (eps * u_node[9] + dt_ * wu[1])
            du10= factor * (eps * u_node[10]+ dt_ * wu[2])
            du11= factor * (eps * u_node[11]+ dt_ * wu[3])
            du12= factor * (eps * u_node[12]+ dt_ * wu[4])
            new_u = SVector(du1, du2, du3, du4, du5, du6, du7, du8, du9, du10, du11, du12)
            set_node_vars!(u, new_u, equations, dg, i, j, element)
        end
    end

    return nothing
end
end # @muladd
