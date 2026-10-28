# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Subgrid-scale models do not need additional arrays by default
function create_cache_subgrid_model(subgrid_model::AbstractSubgridModel,
                                    mesh::P4estMesh{3},
                                    equations_parabolic::AbstractLESDiffusion{3},
                                    dg::DGSEM, cache, uEltype)
    return (;)
end

# Parabolic fluxes of LES equations with an arbitrary subgrid-scale model, which provides the
# eddy viscosity `mu_t` at each node, see `AbstractLESDiffusion` and `AbstractSubgridModel`
function calc_parabolic_fluxes!(flux_parabolic,
                                gradients, u_transformed,
                                mesh::P4estMesh{3},
                                equations_parabolic::AbstractLESDiffusion{3},
                                dg::DG, cache)
    gradients_x, gradients_y, gradients_z = gradients
    flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic # output arrays

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(u_transformed, mesh, equations_parabolic, dg, cache)
        check_axes(gradients_x, mesh, equations_parabolic, dg, cache)
        check_axes(gradients_y, mesh, equations_parabolic, dg, cache)
        check_axes(gradients_z, mesh, equations_parabolic, dg, cache)
        check_axes(flux_parabolic_x, mesh, equations_parabolic, dg, cache)
        check_axes(flux_parabolic_y, mesh, equations_parabolic, dg, cache)
        check_axes(flux_parabolic_z, mesh, equations_parabolic, dg, cache)
    end

    subgrid_model_ = subgrid_model(equations_parabolic)

    @threaded for element in eachelement(dg, cache)
        # Coefficients of the subgrid-scale model in this element
        @inbounds coefficients = calc_subgrid_model_coefficients(u_transformed,
                                                                 gradients,
                                                                 element,
                                                                 subgrid_model_,
                                                                 mesh,
                                                                 equations_parabolic,
                                                                 dg, cache)

        @inbounds for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            # Get solution and gradients
            u_node = get_node_vars(u_transformed, equations_parabolic, dg,
                                   i, j, k, element)
            gradients_1_node = get_node_vars(gradients_x, equations_parabolic, dg,
                                             i, j, k, element)
            gradients_2_node = get_node_vars(gradients_y, equations_parabolic, dg,
                                             i, j, k, element)
            gradients_3_node = get_node_vars(gradients_z, equations_parabolic, dg,
                                             i, j, k, element)
            gradients_node = (gradients_1_node, gradients_2_node, gradients_3_node)

            # Diffusivities, e.g., viscosity and thermal conductivity, including the eddy
            # viscosity `mu_t`
            mu_t = eddy_viscosity(u_node, gradients_node, coefficients, subgrid_model_,
                                  equations_parabolic)
            diffusivities_node = diffusivities(u_node, mu_t, equations_parabolic)

            # Calculate parabolic flux and store each component for later use
            flux_parabolic_node_x = flux(u_node, gradients_node, 1, diffusivities_node,
                                         equations_parabolic)
            flux_parabolic_node_y = flux(u_node, gradients_node, 2, diffusivities_node,
                                         equations_parabolic)
            flux_parabolic_node_z = flux(u_node, gradients_node, 3, diffusivities_node,
                                         equations_parabolic)
            set_node_vars!(flux_parabolic_x, flux_parabolic_node_x,
                           equations_parabolic, dg,
                           i, j, k, element)
            set_node_vars!(flux_parabolic_y, flux_parabolic_node_y,
                           equations_parabolic, dg,
                           i, j, k, element)
            set_node_vars!(flux_parabolic_z, flux_parabolic_node_z,
                           equations_parabolic, dg,
                           i, j, k, element)
        end
    end

    return nothing
end
end # @muladd
