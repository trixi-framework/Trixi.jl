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
function perform_idp_correction!(u, dt, mesh::TreeMesh2D, equations::JinXinEquations, dg, cache)

            # relaxation parameter
            eps = equations.eps_relaxation
            dt_ = dt
            factor =1.0/ (eps + dt_)
            eq_relax = equations.equations_base

            # prepare local storage for projection
            @unpack interpolate_N_to_M, project_M_to_N, filter_modal_to_N = dg.basis
            nnodes_,nnodes_projection = size(project_M_to_N)
            nVars = nvariables(eq_relax)
            RealT = real(dg)
            u_N = zeros(RealT, nVars, nnodes_, nnodes_)
            w_N = zeros(RealT, nVars, nnodes_, nnodes_)
            f_N = zeros(RealT, nVars, nnodes_, nnodes_)
            g_N = zeros(RealT, nVars, nnodes_, nnodes_)
            u_M = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
            w_M_raw = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
            w_M = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
            f_M = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
            g_M = zeros(RealT, nVars, nnodes_projection, nnodes_projection)

            tmp_MxM = zeros(RealT, nVars, nnodes_projection, nnodes_projection)
            tmp_MxN = zeros(RealT, nVars, nnodes_projection, nnodes_)
            tmp_NxM = zeros(RealT, nVars, nnodes_, nnodes_projection)

 #@threaded for element in eachelement(dg, cache)
  for element in eachelement(dg, cache)

        # get element u_N
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            for v in eachvariable(eq_relax)
                u_N[v,i,j] = u_node[v]
            end
        end
        # bring elemtn u_N to grid (M+1)x(M+1)
        multiply_dimensionwise!(u_M,interpolate_N_to_M,u_N,tmp_MxN)
        
        # compute nodal values of entropy variables w on the M grid
        for j in 1:nnodes_projection, i in 1:nnodes_projection
            u_cons = get_node_vars(u_M, eq_relax, dg, i, j)
            w_ij   = cons2entropy(u_cons,eq_relax) 
            set_node_vars!(w_M_raw,w_ij,eq_relax,dg,i,j)
        end
        # compute projection of w with M values down to N
        multiply_dimensionwise!(w_M,filter_modal_to_N,w_M_raw,tmp_MxM)

        #multiply_dimensionwise!(w_N,project_M_to_N,w_M)
        #multiply_dimensionwise!(w_M,interpolate_N_to_M,w_N)


        # compute nodal values of conservative f,g on the M grid
        for j in 1:nnodes_projection, i in 1:nnodes_projection
            w_ij = get_node_vars(w_M, eq_relax, dg, i, j)
            u_cons = entropy2cons(w_ij, eq_relax)
            f_cons = flux(u_cons,1,eq_relax)
            set_node_vars!(f_M,f_cons,eq_relax,dg,i,j)
            g_cons = flux(u_cons,2,eq_relax)
            set_node_vars!(g_M,g_cons,eq_relax,dg,i,j)
        end
        # compute projection of f with M values down to N, same for g
        multiply_dimensionwise!(f_N,project_M_to_N,f_M,tmp_NxM)
        multiply_dimensionwise!(g_N,project_M_to_N,g_M,tmp_NxM)
        #@assert nnodes_projection == nnodes(dg) 
        #for j in 1:nnodes_projection, i in 1:nnodes_projection
        #    u_cons = get_node_vars(u_N, eq_relax, dg, i, j)
        #    f_cons = flux(u_cons,1,eq_relax)
        #    set_node_vars!(f_N,f_cons,eq_relax,dg,i,j)
        #    g_cons = flux(u_cons,2,eq_relax)
        #    set_node_vars!(g_N,g_cons,eq_relax,dg,i,j)
        #end

        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            # compute compressible Euler fluxes
            vu = get_node_vars(f_N,eq_relax,dg,i,j)
            wu = get_node_vars(g_N,eq_relax,dg,i,j)
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
