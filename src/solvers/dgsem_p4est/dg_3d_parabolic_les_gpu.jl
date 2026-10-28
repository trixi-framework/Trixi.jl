# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# KernelAbstractions.jl version of `calc_parabolic_fluxes!` for LES equations, see
# `dg_3d_parabolic_les.jl`. Each thread computes the coefficients of the subgrid-scale model
# of its element, which only have access to the inverse Jacobian.
function calc_parabolic_fluxes!(backend::Backend, flux_parabolic, gradients,
                                u_transformed, mesh::P4estMesh{3},
                                equations_parabolic::AbstractLESDiffusion{3},
                                dg::DG, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack inverse_jacobian = cache.elements
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(u_transformed, mesh, equations_parabolic, dg, cache)
        for gradient in gradients
            check_axes(gradient, mesh, equations_parabolic, dg, cache)
        end
        for flux_parabolic_i in flux_parabolic
            check_axes(flux_parabolic_i, mesh, equations_parabolic, dg, cache)
        end
        check_axes(cache.elements, equations_parabolic, dg, cache)
    end
    kernel! = calc_parabolic_fluxes_les_KAkernel!(backend)
    kernel!(flux_parabolic, gradients, u_transformed, typeof(mesh),
            equations_parabolic, dg, inverse_jacobian,
            ndrange = (element_node_ndrange(mesh, dg)..., nelements(dg, cache)))
    return nothing
end

@kernel inbounds=true function calc_parabolic_fluxes_les_KAkernel!(flux_parabolic,
                                                                   gradients,
                                                                   u_transformed,
                                                                   MeshT::Type{<:P4estMesh{3}},
                                                                   equations_parabolic,
                                                                   dg, inverse_jacobian)
    i, j, k, element = @index(Global, NTuple)
    flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic
    gradients_x, gradients_y, gradients_z = gradients
    subgrid_model_ = subgrid_model(equations_parabolic)

    coefficients = calc_subgrid_model_coefficients(u_transformed, gradients, element,
                                                   subgrid_model_, MeshT,
                                                   equations_parabolic, dg,
                                                   (; elements = (; inverse_jacobian)))

    u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, j, k, element)
    gradients_1_node = get_node_vars(gradients_x, equations_parabolic, dg,
                                     i, j, k, element)
    gradients_2_node = get_node_vars(gradients_y, equations_parabolic, dg,
                                     i, j, k, element)
    gradients_3_node = get_node_vars(gradients_z, equations_parabolic, dg,
                                     i, j, k, element)
    gradients_node = (gradients_1_node, gradients_2_node, gradients_3_node)

    mu_t = eddy_viscosity(u_node, gradients_node, coefficients, subgrid_model_,
                          equations_parabolic)
    diffusivities_node = diffusivities(u_node, mu_t, equations_parabolic)

    flux_parabolic_node_x = flux(u_node, gradients_node, 1, diffusivities_node,
                                 equations_parabolic)
    flux_parabolic_node_y = flux(u_node, gradients_node, 2, diffusivities_node,
                                 equations_parabolic)
    flux_parabolic_node_z = flux(u_node, gradients_node, 3, diffusivities_node,
                                 equations_parabolic)
    set_node_vars!(flux_parabolic_x, flux_parabolic_node_x, equations_parabolic,
                   dg, i, j, k, element)
    set_node_vars!(flux_parabolic_y, flux_parabolic_node_y, equations_parabolic,
                   dg, i, j, k, element)
    set_node_vars!(flux_parabolic_z, flux_parabolic_node_z, equations_parabolic,
                   dg, i, j, k, element)
end
end # @muladd
