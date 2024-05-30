# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function apply_modal_filter!(u_filtered, u, cons2filter, filter2cons, filter_matrix,
                             mesh::TreeMesh{2}, equations, dg, cache,
                             u_element_filtered_threaded, tmp_threaded)
    # Unpack temporary arrays for each thread
    u_element_filtered = u_element_filtered_threaded[Threads.threadid()]
    tmp = tmp_threaded[Threads.threadid()]

    @threaded for element in eachelement(dg, cache)
        # convert u to filter variables
        for j in eachnode(dg), i in eachnode(dg)
            u_node_cons = get_node_vars(u, equations, dg, i, j, element)
            u_node_filter = cons2filter(u_node_cons, equations)
            set_node_vars!(u_filtered, u_node_filter, equations, dg, i, j, element)
        end

        # Apply modal filter
        multiply_dimensionwise!(u_element_filtered, filter_matrix, view(u_filtered, :, :, :, element), tmp)

        # compute nodal values of conservative variables from the projected entropy variables
        for j in eachnode(dg), i in eachnode(dg)
            u_node_filter = get_node_vars(u_element_filtered, equations, dg, i, j)
            u_node_cons = filter2cons(u_node_filter, equations)
            set_node_vars!(u_filtered, u_node_cons, equations, dg, i, j, element)
        end
    end
end

# Convenience version that creates all required temporary arrays in a thread-safe manner
function apply_modal_filter!(u_filtered, u, cons2filter, filter2cons, filter_matrix,
                             mesh::TreeMesh{2}, equations, dg, cache)
      nnodes_ = nnodes(dg)
      nvars = nvariables(equations)
      RealT = eltype(u)
  
      A3 = Array{uEltype, 3}
      u_element_filtered_threaded = A3[A3(undef, nvars, nnodes_, nnodes_)
                                       for _ in 1:Threads.nthreads()]
      tmp_threaded = A3[A3(undef, nvars, nnodes_, nnodes_) for _ in 1:Threads.nthreads()]

      apply_modal_filter!(u_filtered, u, cons2filter, filter2cons, filter_matrix,
                          mesh::TreeMesh{2}, equations, dg, cache,
                          u_element_filtered_threaded, tmp_threaded)
end

# Convenience version that stores output in the same array as the input
function apply_modal_filter!(u, cons2filter, filter2cons, filter_matrix,
                             mesh::TreeMesh{2}, equations, dg, cache, args...)
    apply_modal_filter!(u, u, cons2filter, filter2cons, filter_matrix,
                        mesh::TreeMesh{2}, equations, dg, cache, args...)
end

end # @muladd