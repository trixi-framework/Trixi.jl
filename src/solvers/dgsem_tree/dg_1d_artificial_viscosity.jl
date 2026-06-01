@muladd begin
    function calc_volume_entropy_residual(du, u, element, mesh::TreeMesh{1}, equations, dg,
                                          cache)

        # calculate volume integral
        volume_integral_du_entropy = zero(real(dg))
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            du_node = get_node_vars(du, equations, dg, i, element)
            weight_i = dg.basis.weights[i]

            # calc integral(-dv/dx_i * f(u)) -> missing factor of J
            volume_integral_du_entropy = volume_integral_du_entropy +
                                         dot(cons2entropy(u_node, equations), du_node) *
                                         weight_i
        end

        # calculate surface integral
        surface_integral_entropy_potential = zero(real(dg))
        # x direction
        u_left = get_node_vars(u, equations, dg, 1, element)
        u_right = get_node_vars(u, equations, dg, nnodes(dg), element)
        surface_integral_entropy_potential = (entropy_potential(u_right,
                                                SVector(1.0f0),
                                                                equations) +
                                                entropy_potential(u_left,
                                                                SVector(-1.0f0),
                                                                equations))

        # by default, the volume_integral contribution to du does not scale by any geometric terms
        # For TreeMesh, these geometric terms are ds/dx = 0 and dr/dx * J = 0.5 * h. Thus, to calculate 
        # the volume integral over the physical element, we need to scale by the 1D Jacobian. Similarly,
        # the surface integrals should be scaled by the 1D Jacobian as well. 
        jacobian_1d = inv(cache.elements.inverse_jacobian[element]) # O(h) 
        return (volume_integral_du_entropy + surface_integral_entropy_potential) #*
               #jacobian_1d
    end

    function calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual,
                                     equations, mesh::TreeMesh{1}, dg, cache)
        @threaded for element in eachelement(dg, cache)
            volume_jacobian_ = volume_jacobian(element, mesh, cache)

            # calculate viscous dissipation (ECAV denominator)
            element_viscous_dissipation = zero(real(dg))
            for i in eachnode(dg)
                flux_parabolic_x_node = get_node_vars(flux_parabolic, equations, dg, i,
                                                      element)
                gradients_x_node = get_node_vars(gradients, equations, dg, i, element)
                viscous_dissipation_x = dot(flux_parabolic_x_node, gradients_x_node)

                weight_i = dg.basis.weights[i]
                element_viscous_dissipation = element_viscous_dissipation +
                                              (viscous_dissipation_x) * weight_i * volume_jacobian_
            end

            # Scale viscous flux by ecav coefficient.
            # Note: we usually use "-min(0, entropy_residual)" to define the ECAV coefficient, but we
            # flip the sign to account for the fact that viscous terms are negated by convention in Trixi.jl.
            ecav_coefficient = regularized_ratio(min(0, entropy_residual[element]),
                                                 element_viscous_dissipation)
            #ecav_coefficient = 0.0
            cache.artificial_viscosity.coefficients[element] = -ecav_coefficient # save output
            for i in eachnode(dg)
                flux_parabolic_x_node = get_node_vars(flux_parabolic, equations, dg, i,
                                                      element)
                set_node_vars!(flux_parabolic, ecav_coefficient * flux_parabolic_x_node,
                               equations, dg, i, element)
            end
        end
        push!(cache.artificial_viscosity.max_coeff, maximum(cache.artificial_viscosity.coefficients))
        return nothing
    end

    function rhs_artificial_viscosity!(du, u, t, mesh::TreeMesh{1},
                                       equations, equations_parabolic,
                                       equations_artificial_viscosity,
                                       boundary_conditions, boundary_conditions_parabolic,
                                       source_terms::Source,
                                       dg::DG, solver_parabolic, cache,
                                       cache_parabolic) where {Source}
        backend = trixi_backend(u)

        # Reset du
        @trixi_timeit_ext backend timer() "reset ∂u/∂t" begin
            reset_du!(du, dg, cache)
        end

        # Calculate volume integral
        @trixi_timeit_ext backend timer() "volume integral" begin
            calc_volume_integral!(backend, du, u, mesh,
                                  have_nonconservative_terms(equations), equations,
                                  dg.volume_integral, dg, cache)
        end

        # calculate entropy residual
        entropy_residual = cache.artificial_viscosity.coefficients # reuse storage
        @threaded for element in eachelement(dg, cache)
            entropy_residual[element] = calc_volume_entropy_residual(du, u, element, mesh,
                                                                     equations, dg, cache)
        end

        # Prolong solution to interfaces
        @trixi_timeit_ext backend timer() "prolong2interfaces" begin
            prolong2interfaces!(backend, cache, u, mesh, equations, dg)
        end

        # Calculate interface fluxes
        @trixi_timeit_ext backend timer() "interface flux" begin
            calc_interface_flux!(backend, cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
        end

        # Prolong solution to boundaries
        @trixi_timeit_ext backend timer() "prolong2boundaries" begin
            prolong2boundaries!(cache, u, mesh, equations, dg)
        end

        # Calculate boundary fluxes
        @trixi_timeit_ext backend timer() "boundary flux" begin
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
        @trixi_timeit_ext backend timer() "surface integral" begin
            calc_surface_integral!(backend, du, u, mesh, equations,
                                   dg.surface_integral, dg, cache)
        end

        # @trixi_timeit timer() "transform variables" begin
        #     (; u_transformed, flux_parabolic, gradients) = cache_parabolic.parabolic_container
        #     transform_variables!(u_transformed, u, mesh, equations_artificial_viscosity, dg,
        #                          solver_parabolic, cache)
        # end

        @trixi_timeit timer() "calculate parabolic fluxes" begin
            (; u_transformed, flux_parabolic, gradients) = cache_parabolic.parabolic_container
            calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                   equations_artificial_viscosity, dg, cache)
        end

        calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual, equations,
                                mesh,
                                dg, cache)

        @trixi_timeit timer() "calc divergence" calc_divergence!(du, flux_parabolic, u,
                                                                 mesh,
                                                                 equations_parabolic,
                                                                 boundary_conditions_parabolic, # TODO: hacky pass in parabolic equations
                                                                 #  equations_artificial_viscosity, BoundaryConditionDoNothing(), 
                                                                 dg, solver_parabolic,
                                                                 cache, t)

        # Apply Jacobian from mapping to reference element
        @trixi_timeit_ext backend timer() "Jacobian" begin
            apply_jacobian!(backend, du, mesh, equations, dg, cache)
        end

        # Calculate source terms
        @trixi_timeit_ext backend timer() "source terms" begin
            calc_sources!(du, u, t, source_terms, equations, dg, cache)
        end

        return nothing
    end

    function rhs_combined!(du, u, t, mesh::TreeMesh{1},
                           equations, equations_parabolic, equations_artificial_viscosity,
                           boundary_conditions, boundary_conditions_parabolic,
                           source_terms::Source,
                           dg::DG, parabolic_scheme, cache, cache_parabolic) where {Source}
        (; u_transformed, flux_parabolic, gradients) = cache_parabolic.parabolic_container
        backend = trixi_backend(u)
        # Reset du
        @trixi_timeit_ext backend timer() "reset ∂u/∂t" begin
            set_zero!(du, dg, cache)
        end

        # ========= hyperbolic part ============

        # Calculate volume integral
        @trixi_timeit_ext backend timer() "volume integral" begin
            calc_volume_integral!(backend, du, u, mesh,
                                  have_nonconservative_terms(equations), equations,
                                  dg.volume_integral, dg, cache)
        end

        # calculate entropy residual
        entropy_residual = cache.artificial_viscosity.coefficients # reuse storage
        @threaded for element in eachelement(dg, cache)
            entropy_residual[element] = calc_volume_entropy_residual(du, u, element, mesh,
                                                                     equations, dg, cache)
        end
        #push!(cache.artificial_viscosity.max_coeff, maximum(-min.(0.0, entropy_residual)))

        # Prolong solution to interfaces
        @trixi_timeit_ext backend timer() "prolong2interfaces" begin
            prolong2interfaces!(cache, u, mesh, equations, dg)
        end

        # Calculate interface fluxes
        @trixi_timeit_ext backend timer() "interface flux" begin
            calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
        end

        # Prolong solution to boundaries
        @trixi_timeit_ext backend timer() "prolong2boundaries" begin
            prolong2boundaries!(cache, u, mesh, equations, dg)
        end

        # Calculate boundary fluxes
        @trixi_timeit_ext backend timer() "boundary flux" begin
            calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                                dg.surface_integral, dg)
        end

        # Calculate surface integrals
        @trixi_timeit_ext backend timer() "surface integral" begin
            calc_surface_integral!(backend, du, u, mesh, equations,
                                   dg.surface_integral, dg, cache)
        end

        # ==== shared parabolic terms ====

        # Convert conservative variables to a form more suitable for viscous flux calculations
        @trixi_timeit timer() "transform variables" begin
            transform_variables!(u_transformed, u, mesh, equations_parabolic,
                                 dg, cache)
        end

        # Compute the gradients of the transformed variables
        @trixi_timeit timer() "calculate gradient" begin
            calc_gradient!(gradients, u_transformed, t, mesh,
                           equations_parabolic, boundary_conditions_parabolic,
                           dg, parabolic_scheme, cache)
        end
        #push!(cache.artificial_viscosity.max_coeff, maximum(u_transformed))

        # ========= AV specific part ============

        @trixi_timeit timer() "calculate AV parabolic fluxes" begin
            calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                   equations_artificial_viscosity, dg, cache)
        end

        calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual, equations,
                                mesh,
                                dg, cache)

        # # TODO: accumulate into flux_parabolic instead
        # # accumulate the AV term
        # @trixi_timeit timer() "calc AV divergence" calc_divergence!(du, flux_parabolic, u, mesh, 
        #                                                             equations_artificial_viscosity, 
        #                                                             boundaryConditionDoNothing(), 
        #                                                             dg, parabolic_scheme, cache, t)

        # ======== physical parabolic part ==========

        # accumulate physical viscous fluxes    
        @trixi_timeit timer() "calculate viscous fluxes" begin
            accum_viscous_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                  equations_parabolic, dg, cache)
        end

        # TODO: fix BCs for equations_artificial_viscosity
        @trixi_timeit timer() "calc divergence" calc_divergence!(du, flux_parabolic, u,
                                                                 mesh,
                                                                 equations_parabolic,
                                                                 boundary_conditions_parabolic,
                                                                 dg, parabolic_scheme,
                                                                 cache, t)

        # Apply Jacobian from mapping to reference element
        @trixi_timeit_ext backend timer() "Jacobian" begin
            apply_jacobian!(backend, du, mesh, equations, dg, cache)
        end

        # Calculate source terms
        @trixi_timeit_ext backend timer() "source terms" begin
            calc_sources!(backend, du, u, t, source_terms, equations, dg, cache)
        end

        return nothing
    end

    function accum_viscous_fluxes!(flux_viscous,
                                   gradients, u_transformed,
                                   mesh::Union{TreeMesh{1}, P4estMesh{1}},
                                   equations_parabolic::AbstractEquationsParabolic,
                                   dg::DG, cache)

        @threaded for element in eachelement(dg, cache)
            for i in eachnode(dg)
                # Get solution and gradients
                u_node = get_node_vars(u_transformed, equations_parabolic, dg,
                                       i, element)
                gradients_node = get_node_vars(gradients, equations_parabolic, dg,
                                                 i, element)
                # Calculate viscous flux and store each component for later use
                flux_viscous_node = flux(u_node, (gradients_node,), 1,
                                           equations_parabolic)
                # flip sign for Trixi's parabolic convention
                add_to_node_vars!(flux_viscous, -flux_viscous_node, equations_parabolic,
                                  dg,
                                  i, element)
            end
        end

        return nothing
    end
end # @muladd
