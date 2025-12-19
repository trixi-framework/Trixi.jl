@muladd begin
function create_cache(mesh, artificial_viscosity::EntropyCorrectionArtificialViscosity, dg::DG, cache, RealT, uEltype)
    
    coefficients = zeros(real(dg), nelements(dg, cache))
    cache = (; coefficients)
    return cache
end

function calc_volume_entropy_residual(du, u, element, mesh::TreeMesh{2}, equations, dg)
    # calculate volume integral
    volume_integral_du_entropy = zero(real(dg))
    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)
        du_node = get_node_vars(du, equations, dg, i, j, element)
        weight_ij = dg.basis.weights[i] * dg.basis.weights[j]
        volume_integral_du_entropy = volume_integral_du_entropy + 
            dot(cons2entropy(u_node, equations), du_node) * weight_ij
    end
    
    # calculate surface integral
    surface_integral_entropy_potential = zero(real(dg))
    for ii in eachnode(dg)
        # x direction
        u_left = get_node_vars(u, equations, dg, 1, ii, element)
        u_right = get_node_vars(u, equations, dg, nnodes(dg), ii, element)
        surface_integral_entropy_potential = surface_integral_entropy_potential + 
            dg.basis.weights[ii] * (entropy_potential(u_right, SVector(1.f0, 0.f0), equations) +
                                    entropy_potential(u_left, SVector(-1.f0, 0.f0), equations))

        # y direction
        u_left = get_node_vars(u, equations, dg, ii, 1, element)
        u_right = get_node_vars(u, equations, dg, ii, nnodes(dg), element)
        surface_integral_entropy_potential = surface_integral_entropy_potential + 
            dg.basis.weights[ii] * (entropy_potential(u_right, SVector(0.f0, 1.f0), equations) + 
                                    entropy_potential(u_left, SVector(0.f0, -1.f0), equations))
    end
    return volume_integral_du_entropy + surface_integral_entropy_potential
end

function rhs_artificial_viscosity!(du, u, t, mesh::TreeMesh{2}, 
                                   equations, equations_parabolic, equations_artificial_viscosity, 
                                   boundary_conditions, boundary_conditions_parabolic, source_terms::Source,
                                   dg::DG, solver_parabolic, cache, cache_parabolic) where {Source}

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # calculate entropy residual
    entropy_residual = cache.artificial_viscosity.coefficients # reuse storage
    @threaded for element in eachelement(dg, cache)
        entropy_residual[element] = calc_volume_entropy_residual(du, u, element, mesh, equations, dg)
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, u, mesh, equations, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
                             dg.surface_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, u, mesh, equations, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # # Prolong solution to mortars
    # @trixi_timeit timer() "prolong2mortars" begin
    #     # prolong2mortars!(cache, u, mesh, equations, dg.mortar, dg)
    #     prolong_entropy_projection_2_mortars!(cache, u, mesh, equations, dg.mortar, dg)
    # end

    # # Calculate mortar fluxes
    # @trixi_timeit timer() "mortar flux" begin
    #     calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
    #                       have_nonconservative_terms(equations), equations,
    #                       dg.mortar, dg.surface_integral, dg, cache)
    # end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    # @trixi_timeit timer() "transform variables" begin
    #     (; u_transformed, flux_viscous, gradients) = cache_parabolic.viscous_container
    #     transform_variables!(u_transformed, u, mesh, equations_artificial_viscosity, dg,
    #                          solver_parabolic, cache)
    # end

    @trixi_timeit timer() "calculate viscous fluxes" begin
        (; u_transformed, flux_viscous, gradients) = cache_parabolic.viscous_container
        calc_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh,
                             equations_artificial_viscosity, dg, cache)
    end

    # --- calculate AV denominator by dotting `flux_viscous` and `gradients`
    @threaded for element in eachelement(dg, cache)

        # calculate volume integral
        element_viscous_dissipation = zero(real(dg))
        for j in eachnode(dg), i in eachnode(dg)
            flux_viscous_x_node = get_node_vars(flux_viscous[1], equations, dg, i, j, element)
            flux_viscous_y_node = get_node_vars(flux_viscous[2], equations, dg, i, j, element)
            gradients_x_node = get_node_vars(gradients[1], equations, dg, i, j, element)
            gradients_y_node = get_node_vars(gradients[2], equations, dg, i, j, element)
            viscous_dissipation_x = dot(flux_viscous_x_node, gradients_x_node)
            viscous_dissipation_y = dot(flux_viscous_y_node, gradients_y_node)

            volume_jacobian_ = volume_jacobian(element, mesh, cache)
            weight_ij = dg.basis.weights[i] * dg.basis.weights[j]
            element_viscous_dissipation = element_viscous_dissipation + 
                (viscous_dissipation_x + viscous_dissipation_y) * weight_ij * volume_jacobian_
        end

        # Scale viscous flux by ecav coefficient. 
        # Note: we usually use "-min(0, entropy_residual)" to define the ECAV coefficient, but we 
        # flip the sign to account for the fact that viscous terms are negated by convention in Trixi.jl. 
        ecav_coefficient = regularized_ratio(min(0, entropy_residual[element]), element_viscous_dissipation)
        # ecav_coefficient *= equations_artificial_viscosity.diffusivity # optional extra scaling of AV
        cache.artificial_viscosity.coefficients[element] = -ecav_coefficient # save output
        for j in eachnode(dg), i in eachnode(dg)
            flux_viscous_x_node = get_node_vars(flux_viscous[1], equations, dg, i, j, element)
            flux_viscous_y_node = get_node_vars(flux_viscous[2], equations, dg, i, j, element)
            set_node_vars!(flux_viscous[1], ecav_coefficient * flux_viscous_x_node, equations, dg, i, j, element)
            set_node_vars!(flux_viscous[2], ecav_coefficient * flux_viscous_y_node, equations, dg, i, j, element)
        end
    end

    @trixi_timeit timer() "calc divergence" calc_divergence!(du, flux_viscous, u, mesh, 
                                                             equations_parabolic, boundary_conditions_parabolic, # TODO: hacky pass in parabolic equations 
                                                             #  equations_artificial_viscosity, BoundaryConditionDoNothing(), 
                                                             dg, solver_parabolic, cache, t)

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, equations, dg, cache)
    end

    return nothing
end

function rhs_combined!(du, u, t, mesh::TreeMesh{2}, 
                       equations, equations_parabolic, equations_artificial_viscosity, 
                       boundary_conditions, boundary_conditions_parabolic, source_terms::Source,
                       dg::DG, parabolic_scheme, cache, cache_parabolic) where {Source}

    @unpack viscous_container = cache_parabolic
    @unpack u_transformed, gradients, flux_viscous = viscous_container

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # ========= hyperbolic part ============

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # calculate entropy residual
    entropy_residual = cache.artificial_viscosity.coefficients # reuse storage
    @threaded for element in eachelement(dg, cache)
        entropy_residual[element] = calc_volume_entropy_residual(du, u, element, mesh, equations, dg)
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, u, mesh, equations, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
                             dg.surface_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, u, mesh, equations, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    # ==== shared parabolic terms ====

    # Convert conservative variables to a form more suitable for viscous flux calculations
    @trixi_timeit timer() "transform variables" begin
        transform_variables!(u_transformed, u, mesh, equations_parabolic,
                             dg, parabolic_scheme, cache)
    end

    # Compute the gradients of the transformed variables
    @trixi_timeit timer() "calculate gradient" begin
        calc_gradient!(gradients, u_transformed, t, mesh,
                       equations_parabolic, boundary_conditions_parabolic,
                       dg, parabolic_scheme, cache)
    end

    # ========= AV specific part ============

    @trixi_timeit timer() "calculate AV viscous fluxes" begin
        (; u_transformed, flux_viscous, gradients) = cache_parabolic.viscous_container
        calc_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh,
                             equations_artificial_viscosity, dg, cache)
    end

    # --- calculate ECAV denominator by dotting `flux_viscous` and `gradients`
    @threaded for element in eachelement(dg, cache)

        # calculate volume integral
        element_viscous_dissipation = zero(real(dg))
        for j in eachnode(dg), i in eachnode(dg)
            flux_viscous_x_node = get_node_vars(flux_viscous[1], equations, dg, i, j, element)
            flux_viscous_y_node = get_node_vars(flux_viscous[2], equations, dg, i, j, element)
            gradients_x_node = get_node_vars(gradients[1], equations, dg, i, j, element)
            gradients_y_node = get_node_vars(gradients[2], equations, dg, i, j, element)
            viscous_dissipation_x = dot(flux_viscous_x_node, gradients_x_node)
            viscous_dissipation_y = dot(flux_viscous_y_node, gradients_y_node)

            volume_jacobian_ = volume_jacobian(element, mesh, cache)
            weight_ij = dg.basis.weights[i] * dg.basis.weights[j]
            element_viscous_dissipation = element_viscous_dissipation + 
                (viscous_dissipation_x + viscous_dissipation_y) * weight_ij * volume_jacobian_
        end

        # Scale viscous flux by ecav coefficient. 
        # Note: we usually use "-min(0, entropy_residual)" to define the ECAV coefficient, but we 
        # flip the sign to account for the fact that viscous terms are negated by convention in Trixi.jl. 
        ecav_coefficient = regularized_ratio(min(0, entropy_residual[element]), element_viscous_dissipation)
        # ecav_coefficient *= equations_artificial_viscosity.diffusivity # optional extra scaling of AV
        cache.artificial_viscosity.coefficients[element] = -ecav_coefficient # save output
        for j in eachnode(dg), i in eachnode(dg)
            flux_viscous_x_node = get_node_vars(flux_viscous[1], equations, dg, i, j, element)
            flux_viscous_y_node = get_node_vars(flux_viscous[2], equations, dg, i, j, element)
            set_node_vars!(flux_viscous[1], ecav_coefficient * flux_viscous_x_node, equations, dg, i, j, element)
            set_node_vars!(flux_viscous[2], ecav_coefficient * flux_viscous_y_node, equations, dg, i, j, element)
        end
    end

    # # TODO: accumulate into flux_viscous instead
    # # accumulate the AV term
    # @trixi_timeit timer() "calc AV divergence" calc_divergence!(du, flux_viscous, u, mesh, 
    #                                                             equations_artificial_viscosity, 
    #                                                             boundaryConditionDoNothing(), 
    #                                                             dg, parabolic_scheme, cache, t)


    # ======== physical parabolic part ==========
        
    # accumulate physical viscous fluxes    
    @trixi_timeit timer() "calculate viscous fluxes" begin
        accum_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh,
                              equations_parabolic, dg, cache)
    end

    # TODO: fix BCs for equations_artificial_viscosity
    @trixi_timeit timer() "calc divergence" calc_divergence!(du, flux_viscous, u, mesh, 
                                                             equations_parabolic, 
                                                             boundary_conditions_parabolic, 
                                                             dg, parabolic_scheme, cache, t)
   
    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, equations, dg, cache)
    end

    return nothing
end

function accum_viscous_fluxes!(flux_viscous,
                               gradients, u_transformed,
                               mesh::Union{TreeMesh{2}, P4estMesh{2}},
                               equations_parabolic::AbstractEquationsParabolic,
                               dg::DG, cache)
    gradients_x, gradients_y = gradients
    flux_viscous_x, flux_viscous_y = flux_viscous # output arrays

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            # Get solution and gradients
            u_node = get_node_vars(u_transformed, equations_parabolic, dg,
                                   i, j, element)
            gradients_1_node = get_node_vars(gradients_x, equations_parabolic, dg,
                                             i, j, element)
            gradients_2_node = get_node_vars(gradients_y, equations_parabolic, dg,
                                             i, j, element)

            # Calculate viscous flux and store each component for later use
            flux_viscous_node_x = flux(u_node, (gradients_1_node, gradients_2_node), 1,
                                                             equations_parabolic)
            flux_viscous_node_y = flux(u_node, (gradients_1_node, gradients_2_node), 2,
                                                             equations_parabolic)

            # flip sign for Trixi's parabolic convention
            add_to_node_vars!(flux_viscous_x, -flux_viscous_node_x, equations_parabolic, dg,
                           i, j, element)
            add_to_node_vars!(flux_viscous_y, -flux_viscous_node_y, equations_parabolic, dg,
                           i, j, element)
        end
    end

    return nothing
end

end # @muladd
