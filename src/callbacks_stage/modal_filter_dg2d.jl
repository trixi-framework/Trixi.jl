# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function apply_modal_filter!(u_filtered, u, cons2filter, filter2cons, filter_matrix,
                             mesh::TreeMesh{2}, equations, dg, cache)
    nnodes_ = nnodes(dg)
    nvars = nvariables(equations)
    RealT = eltype(u)

    u_element = zeros(RealT, nvars, nnodes_, nnodes_)
    u_element_filtered = zeros(RealT, nvars, nnodes_, nnodes_)
    tmp_NxN = zeros(RealT, nvars, nnodes_, nnodes_)

    @threaded for element in eachelement(dg, cache)
        # convert u to filter variables
        for j in eachnode(dg), i in eachnode(dg)
            u_node_cons = get_node_vars(u, equations, dg, i, j, element)
            u_node_filter = cons2filter(u_node_cons, equations)
            for v in eachvariable(equations)
                u_element[v, i, j] = u_node_filter[v]
            end
        end

        # Apply modal filter
        multiply_dimensionwise!(u_element_filtered, filter_matrix, u_element, tmp_NxN)

        # compute nodal values of conservative variables from the projected entropy variables
        for j in eachnode(dg), i in eachnode(dg)
            u_node_filter = get_node_vars(u_element_filtered, equations, dg, i, j)
            u_node_cons = filter2cons(u_node_filter, equations)
            set_node_vars!(u_filtered, u_node_cons, equations, dg, i, j, element)
        end
    end
end
end # @muladd
