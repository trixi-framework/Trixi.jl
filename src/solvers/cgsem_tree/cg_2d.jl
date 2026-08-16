# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# everything related to a CG semidiscretization in 2D,
# currently limited to Lobatto-Legendre nodes

# This method is called when a SemidiscretizationHyperbolic is constructed.
# In contrast to the DGSEM, the cache contains neither interface solution values
# nor boundaries nor mortars since the elements are coupled by a direct stiffness
# summation, which only requires the connectivity of the elements.
function create_cache(mesh::TreeMesh{2}, equations,
                      cg::CGSEM, RealT, uEltype)
    # Get cells for which an element needs to be created (i.e. all leaf cells)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations, cg.basis, RealT, uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements, cg)

    # Container cache
    cache = (; elements, interfaces)

    return cache
end

# Project the initial condition onto the space of continuous, piecewise
# polynomial functions
function compute_coefficients!(backend::Nothing, u, func, t, mesh::TreeMesh{2},
                               equations, cg::CGSEM, cache)
    @unpack node_coordinates = cache.elements
    node_indices = CartesianIndices(ntuple(_ -> nnodes(cg), ndims(mesh)))
    @threaded for element in eachelement(cg, cache)
        compute_coefficients_per_element!(u, func, t, equations, cg, node_coordinates,
                                          element, node_indices)
    end

    apply_direct_stiffness_summation!(u, mesh, equations, cg, cache)

    return nothing
end

# The CGSEM uses the same element-local flux differencing volume integral as the
# DGSEM. Since the numerical solution is continuous, the surface contributions of
# two neighboring elements cancel each other, i.e., there is no surface integral.
# Instead, the element-local contributions of the shared degrees of freedom are
# combined by a direct stiffness summation at the very end.
function rhs_hyperbolic!(backend::Nothing,
                         du, u, t,
                         mesh::TreeMesh{2},
                         equations,
                         boundary_conditions, source_terms::Source,
                         cg::CGSEM, cache) where {Source}
    # Reset du
    @trixi_timeit_ext backend timer() "reset ∂u/∂t" begin
        set_zero!(du, cg, cache)
    end

    # Calculate volume integral
    @trixi_timeit_ext backend timer() "volume integral" begin
        calc_volume_integral!(backend, du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              cg.volume_integral, cg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit_ext backend timer() "Jacobian" begin
        apply_jacobian!(backend, du, mesh, equations, cg, cache)
    end

    # Calculate source terms
    @trixi_timeit_ext backend timer() "source terms" begin
        calc_sources!(backend, du, u, t, source_terms, equations, cg, cache)
    end

    # Combine the element-local contributions of the shared degrees of freedom
    @trixi_timeit_ext backend timer() "direct stiffness summation" begin
        apply_direct_stiffness_summation!(du, mesh, equations, cg, cache)
    end

    return nothing
end

# Since the mesh is conforming, all elements sharing a degree of freedom have the
# same quadrature weights and the same Jacobian. Thus, dividing the assembled
# contributions by the assembled (diagonal) mass matrix reduces to taking the
# arithmetic mean of the element-local values.
#
# The interfaces in x- and y-direction are swept one after another. This also
# yields the correct mean value at the degrees of freedom shared by four elements:
# After the sweep in x-direction, the two values entering the sweep in y-direction
# are the means of the two values below and above the shared node, respectively.
function apply_direct_stiffness_summation!(u, mesh::TreeMesh{2}, equations,
                                           cg::CGSEM, cache)
    @unpack neighbor_ids, orientations = cache.interfaces

    # Sweep the interfaces in x-direction. Since every element has exactly one
    # interface in negative and one interface in positive x-direction, no two
    # interfaces of this sweep touch the same degrees of freedom.
    @threaded for interface in eachinterface(cg, cache)
        if orientations[interface] == 1
            left_element = neighbor_ids[1, interface]
            right_element = neighbor_ids[2, interface]

            for j in eachnode(cg), v in eachvariable(equations)
                u_mean = 0.5f0 * (u[v, nnodes(cg), j, left_element] +
                          u[v, 1, j, right_element])
                u[v, nnodes(cg), j, left_element] = u_mean
                u[v, 1, j, right_element] = u_mean
            end
        end
    end

    # Sweep the interfaces in y-direction
    @threaded for interface in eachinterface(cg, cache)
        if orientations[interface] == 2
            left_element = neighbor_ids[1, interface]
            right_element = neighbor_ids[2, interface]

            for i in eachnode(cg), v in eachvariable(equations)
                u_mean = 0.5f0 * (u[v, i, nnodes(cg), left_element] +
                          u[v, i, 1, right_element])
                u[v, i, nnodes(cg), left_element] = u_mean
                u[v, i, 1, right_element] = u_mean
            end
        end
    end

    return nothing
end
end # @muladd
