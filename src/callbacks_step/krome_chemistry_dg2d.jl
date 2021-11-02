# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


  function apply_krome_chemistry!(u, dt, mesh, chemistry_terms, equations::CompressibleEulerMultichemistryEquations2D, dg::DG, cache)

    @threaded for element in eachelement(dg, cache)
      for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)
        update = chemistry_terms(u_node, dt, equations)
        add_to_node_vars!(u, update, equations, dg, i, j, element)
      end
    end

    return nothing
  end


  end # @muladd 