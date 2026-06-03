@muladd begin
    function calc_volume_entropy_residual(du, u, element, mesh::TreeMesh{3}, equations, dg,
                                          cache)

        # calculate volume integral
        volume_integral_du_entropy = zero(real(dg))
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            du_node = get_node_vars(du, equations, dg, i, j, k, element)
            weight_ijk = dg.basis.weights[i] * dg.basis.weights[j] * dg.basis.weights[k]

            # calc integral(-dv/dx_i * f(u)) -> missing factor of J
            volume_integral_du_entropy = volume_integral_du_entropy +
                                         dot(cons2entropy(u_node, equations), du_node) *
                                         weight_ijk
        end

        # calculate surface integral
        surface_integral_entropy_potential = zero(real(dg))
        for ii in eachnode(dg), jj in eachnode(dg)
            # x direction
            u_left = get_node_vars(u, equations, dg, 1, ii, jj, element)
            u_right = get_node_vars(u, equations, dg, nnodes(dg), ii, jj, element)
            surface_integral_entropy_potential = surface_integral_entropy_potential +
                                                 dg.basis.weights[ii] * dg.basis.weights[jj] * 
                                                 (entropy_potential(u_right,
                                                                    SVector(1.0f0, 0.0f0, 0.0f0),
                                                                    equations) +
                                                  entropy_potential(u_left,
                                                                    SVector(-1.0f0, 0.0f0, 0.0f0),
                                                                    equations))

            # y direction
            u_left = get_node_vars(u, equations, dg, ii, 1, jj, element)
            u_right = get_node_vars(u, equations, dg, ii, nnodes(dg), jj, element)
            surface_integral_entropy_potential = surface_integral_entropy_potential +
                                                 dg.basis.weights[ii] * dg.basis.weights[jj] * 
                                                 (entropy_potential(u_right,
                                                                    SVector(0.0f0, 1.0f0, 0.0f0),
                                                                    equations) +
                                                  entropy_potential(u_left,
                                                                    SVector(0.0f0, -1.0f0, 0.0f0),
                                                                    equations))
            # z direction 
            u_left = get_node_vars(u, equations, dg, ii, jj, 1, element)
            u_right = get_node_vars(u, equations, dg, ii, jj, nnodes(dg), element)
            surface_integral_entropy_potential = surface_integral_entropy_potential +
                                                 dg.basis.weights[ii] * dg.basis.weights[jj] *
                                                 (entropy_potential(u_right,
                                                                    SVector(0.0f0, 0.0f0, 1.0f0),
                                                                    equations) +
                                                  entropy_potential(u_left,
                                                                    SVector(0.0f0, 0.0f0, -1.0f0),
                                                                    equations))
        end

        # by default, the volume_integral contribution to du does not scale by any geometric terms
        # For TreeMesh, these geometric terms are ds/dx = 0 and dr/dx * J = 0.5 * h. Thus, to calculate 
        # the volume integral over the physical element, we need to scale by the 1D Jacobian. Similarly,
        # the surface integrals should be scaled by the 1D Jacobian as well. 
        jacobian_1d = inv(cache.elements.inverse_jacobian[element])# O(h) 
        return (volume_integral_du_entropy + surface_integral_entropy_potential) *
               jacobian_1d * jacobian_1d;
        #cache.artificial_viscosity.norm_residuals[end] += res * res;
        #return res
    end

    function calc_ecav_svv_coefficients!(flux_parabolic, gradients, 
            filtered_parabolic, filtered_gradients, tmp_gradient, tmp_parabolic,
                entropy_residual, equations, mesh::TreeMesh{3}, dg, cache) 
                
        flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic
        gradients_x, gradients_y, gradients_z = gradients

        tmp_gradient_x, tmp_gradient_y, tmp_gradient_z = tmp_gradient
        tmp_parabolic_x, tmp_parabolic_y, tmp_parabolic_z = tmp_parabolic

        filtered_gradients_x, filtered_gradients_y, 
            filtered_gradients_z = filtered_gradients

        filtered_flux_parabolic_x, filtered_flux_parabolic_y, 
            filtered_flux_parabolic_z = filtered_parabolic

        (; F1D) = cache
        @threaded for element in eachelement(dg, cache)
            volume_jacobian_ = volume_jacobian(element, mesh, cache)

            # calculate viscous dissipation (ECAV denominator)
            element_viscous_dissipation = zero(real(dg))
            svv_element_viscous_dissipation = zero(real(dg))
            # allocate memory to store svv viscous flux per node wise since
            # we don't want to allocate another flux_viscous type of object.
            # we will accumulate the svv fluxes into flux_viscous to save memory.
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                flux_viscous_x_node = get_node_vars(flux_parabolic_x, equations, dg, i, j, k, 
                                                    element)
                flux_viscous_y_node = get_node_vars(flux_parabolic_y, equations, dg, i, j, k, 
                                                    element)
                flux_viscous_z_node = get_node_vars(flux_parabolic_z, equations, dg, i, j, k, 
                                                    element)
                gradients_x_node = get_node_vars(gradients_x, equations, dg, i, j, k, element)
                gradients_y_node = get_node_vars(gradients_y, equations, dg, i, j, k, element)
                gradients_z_node = get_node_vars(gradients_z, equations, dg, i, j, k, element)
                
                viscous_dissipation_x = dot(flux_viscous_x_node, gradients_x_node)
                viscous_dissipation_y = dot(flux_viscous_y_node, gradients_y_node)
                viscous_dissipation_z = dot(flux_viscous_z_node, gradients_z_node)

                weight_ijk = dg.basis.weights[i] * dg.basis.weights[j] * dg.basis.weights[k]
                element_viscous_dissipation = element_viscous_dissipation +
                                              (viscous_dissipation_x +
                                               viscous_dissipation_y + 
                                                viscous_dissipation_z) * weight_ijk *
                                              volume_jacobian_
            end
            filtered_gradient_node_x = similar(get_node_vars(gradients_x, equations, dg, 1, 1, 1, 1));
            filtered_flux_parabolic_node_x = similar(filtered_gradient_node_x)
            filtered_gradient_node_y = similar(filtered_gradient_node_x)
            filtered_flux_parabolic_node_y = similar(filtered_gradient_node_x)
            filtered_gradient_node_z = similar(filtered_gradient_node_x);
            filtered_flux_parabolic_node_z = similar(filtered_gradient_node_x);

            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                fill!(filtered_gradient_node_x, zero(real(dg)))
                fill!(filtered_flux_parabolic_node_x, zero(real(dg)))
                fill!(filtered_gradient_node_y, zero(real(dg)))
                fill!(filtered_flux_parabolic_node_y, zero(real(dg)))
                fill!(filtered_gradient_node_z, zero(real(dg)))
                fill!(filtered_flux_parabolic_node_z, zero(real(dg)))
                for ii in eachnode(dg)
                    Fval = F1D[i, ii]
                    tmp_gradient_x_node = get_node_vars(gradients_x, equations, dg, ii, j, k, element)
                    tmp_parabolic_x_node = get_node_vars(flux_parabolic_x, equations, dg, ii, j, k, element)

                    tmp_gradient_y_node = get_node_vars(gradients_y, equations, dg, ii, j, k, element)
                    tmp_parabolic_y_node = get_node_vars(flux_parabolic_y, equations, dg, ii, j, k, element)

                    tmp_gradient_z_node = get_node_vars(gradients_z, equations, dg, ii, j, k, element)
                    tmp_parabolic_z_node = get_node_vars(flux_parabolic_z, equations, dg, ii, j, k, element)
                    @simd for v in 1:5
                        filtered_gradient_node_x[v] += Fval * tmp_gradient_x_node[v]
                        filtered_flux_parabolic_node_x[v] += Fval * tmp_parabolic_x_node[v]

                        filtered_gradient_node_y[v] += Fval * tmp_gradient_y_node[v]
                        filtered_flux_parabolic_node_y[v] += Fval * tmp_parabolic_y_node[v]

                        filtered_gradient_node_z[v] += Fval * tmp_gradient_z_node[v]
                        filtered_flux_parabolic_node_z[v] += Fval * tmp_parabolic_z_node[v]
                    end
                end
                set_node_vars!(filtered_gradients_x, filtered_gradient_node_x, equations, dg, i, j, k, element);
                set_node_vars!(filtered_flux_parabolic_x, filtered_flux_parabolic_node_x, equations, dg, i, j, k, element);

                set_node_vars!(filtered_gradients_y, filtered_gradient_node_y, equations, dg, i, j, k, element);
                set_node_vars!(filtered_flux_parabolic_y, filtered_flux_parabolic_node_y, equations, dg, i, j, k, element);

                set_node_vars!(filtered_gradients_z, filtered_gradient_node_z, equations, dg, i, j, k, element);
                set_node_vars!(filtered_flux_parabolic_z, filtered_flux_parabolic_node_z, equations, dg, i, j, k, element);
            end

            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                fill!(filtered_gradient_node_x, zero(real(dg)))
                fill!(filtered_flux_parabolic_node_x, zero(real(dg)))
                fill!(filtered_gradient_node_y, zero(real(dg)))
                fill!(filtered_flux_parabolic_node_y, zero(real(dg)))
                fill!(filtered_gradient_node_z, zero(real(dg)))
                fill!(filtered_flux_parabolic_node_z, zero(real(dg)))
                for jj in eachnode(dg)
                    Fval = F1D[j, jj]
                    tmp_gradient_x_node = get_node_vars(filtered_gradients_x, equations, dg, i, jj, k, element)
                    tmp_parabolic_x_node = get_node_vars(filtered_flux_parabolic_x, equations, dg, i, jj, k, element)

                    tmp_gradient_y_node = get_node_vars(filtered_gradients_y, equations, dg, i, jj, k, element)
                    tmp_parabolic_y_node = get_node_vars(filtered_flux_parabolic_y, equations, dg, i, jj, k, element)

                    tmp_gradient_z_node = get_node_vars(filtered_gradients_z, equations, dg, i, jj, k, element)
                    tmp_parabolic_z_node = get_node_vars(filtered_flux_parabolic_z, equations, dg, i, jj, k, element)
                    @simd for v in 1:5
                        filtered_gradient_node_x[v] += Fval * tmp_gradient_x_node[v]
                        filtered_flux_parabolic_node_x[v] += Fval * tmp_parabolic_x_node[v]

                        filtered_gradient_node_y[v] += Fval * tmp_gradient_y_node[v]
                        filtered_flux_parabolic_node_y[v] += Fval * tmp_parabolic_y_node[v]

                        filtered_gradient_node_z[v] += Fval * tmp_gradient_z_node[v]
                        filtered_flux_parabolic_node_z[v] += Fval * tmp_parabolic_z_node[v]
                    end
                end
                set_node_vars!(tmp_gradient_x, filtered_gradient_node_x, equations, dg, i, j, k, element);
                set_node_vars!(tmp_parabolic_x, filtered_flux_parabolic_node_x, equations, dg, i, j, k, element);

                set_node_vars!(tmp_gradient_y, filtered_gradient_node_y, equations, dg, i, j, k, element);
                set_node_vars!(tmp_parabolic_y, filtered_flux_parabolic_node_y, equations, dg, i, j, k, element);

                set_node_vars!(tmp_gradient_z, filtered_gradient_node_z, equations, dg, i, j, k, element);
                set_node_vars!(tmp_parabolic_z, filtered_flux_parabolic_node_z, equations, dg, i, j, k, element);
            end

            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                fill!(filtered_gradient_node_x, zero(real(dg)))
                fill!(filtered_flux_parabolic_node_x, zero(real(dg)))
                fill!(filtered_gradient_node_y, zero(real(dg)))
                fill!(filtered_flux_parabolic_node_y, zero(real(dg)))
                fill!(filtered_gradient_node_z, zero(real(dg)))
                fill!(filtered_flux_parabolic_node_z, zero(real(dg)))
                for kk in eachnode(dg)
                    Fval = F1D[k, kk]
                    tmp_gradient_x_node = get_node_vars(tmp_gradient_x, equations, dg, i, j, kk, element)
                    tmp_parabolic_x_node = get_node_vars(tmp_parabolic_x, equations, dg, i, j, kk, element)

                    tmp_gradient_y_node = get_node_vars(tmp_gradient_y, equations, dg, i, j, kk, element)
                    tmp_parabolic_y_node = get_node_vars(tmp_parabolic_y, equations, dg, i, j, kk, element)

                    tmp_gradient_z_node = get_node_vars(tmp_gradient_z, equations, dg, i, j, kk, element)
                    tmp_parabolic_z_node = get_node_vars(tmp_parabolic_z, equations, dg, i, j, kk, element)
                    @simd for v in eachvariable(equations)
                        filtered_gradient_node_x[v] += Fval * tmp_gradient_x_node[v]
                        filtered_flux_parabolic_node_x[v] += Fval * tmp_parabolic_x_node[v]

                        filtered_gradient_node_y[v] += Fval * tmp_gradient_y_node[v]
                        filtered_flux_parabolic_node_y[v] += Fval * tmp_parabolic_y_node[v]

                        filtered_gradient_node_z[v] += Fval * tmp_gradient_z_node[v]
                        filtered_flux_parabolic_node_z[v] += Fval * tmp_parabolic_z_node[v]
                    end
                end 
                set_node_vars!(filtered_gradients_x, filtered_gradient_node_x, equations, dg, i, j, k, element);
                set_node_vars!(filtered_flux_parabolic_x, filtered_flux_parabolic_node_x, equations, dg, i, j, k, element);

                set_node_vars!(filtered_gradients_y, filtered_gradient_node_y, equations, dg, i, j, k, element);
                set_node_vars!(filtered_flux_parabolic_y, filtered_flux_parabolic_node_y, equations, dg, i, j, k, element);

                set_node_vars!(filtered_gradients_z, filtered_gradient_node_z, equations, dg, i, j, k, element);
                set_node_vars!(filtered_flux_parabolic_z, filtered_flux_parabolic_node_z, equations, dg, i, j, k, element);
            end


            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                filtered_flux_viscous_x_node = get_node_vars(filtered_flux_parabolic_x, equations, dg, i, j, k, 
                                                    element)
                filtered_flux_viscous_y_node = get_node_vars(filtered_flux_parabolic_y, equations, dg, i, j, k, 
                                                    element)
                filtered_flux_viscous_z_node = get_node_vars(filtered_flux_parabolic_z, equations, dg, i, j, k, 
                                                    element)
                filtered_gradients_x_node = get_node_vars(filtered_gradients_x, equations, dg, i, j, k, element)
                filtered_gradients_y_node = get_node_vars(filtered_gradients_y, equations, dg, i, j, k, element)
                filtered_gradients_z_node = get_node_vars(filtered_gradients_z, equations, dg, i, j, k, element)
                
                svv_viscous_dissipation_x = dot(filtered_flux_viscous_x_node, filtered_gradients_x_node)
                svv_viscous_dissipation_y = dot(filtered_flux_viscous_y_node, filtered_gradients_y_node)
                svv_viscous_dissipation_z = dot(filtered_flux_viscous_z_node, filtered_gradients_z_node)

                weight_ijk = dg.basis.weights[i] * dg.basis.weights[j] * dg.basis.weights[k]
                svv_element_viscous_dissipation = svv_element_viscous_dissipation +
                                              (svv_viscous_dissipation_x +
                                               svv_viscous_dissipation_y + 
                                                svv_viscous_dissipation_z) * weight_ijk *
                                              volume_jacobian_
            end
            # Scale viscous flux by ecav coefficient.
            # Note: we usually use "-min(0, entropy_residual)" to define the ECAV coefficient, but we
            # flip the sign to account for the fact that viscous terms are negated by convention in Trixi.jl.
            w = 1.0
            num = min(0, entropy_residual[element])
            denom = element_viscous_dissipation * element_viscous_dissipation + 
                    w * svv_element_viscous_dissipation * svv_element_viscous_dissipation
            ecav_coefficient = (num * element_viscous_dissipation) / (denom + 1e-12)
            svv_coefficient = (w* num * svv_element_viscous_dissipation) /(denom + 1e-12)

            cache.artificial_viscosity.coefficients[element] = -ecav_coefficient # save output
            cache.artificial_viscosity.svv_coefficients[element] = -svv_coefficient # save output
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)

                filtered_flux_parabolic_x_node = get_node_vars(filtered_flux_parabolic_x, equations, dg, i, j, k, element)
                filtered_flux_parabolic_y_node = get_node_vars(filtered_flux_parabolic_y, equations, dg, i, j, k, element)
                filtered_flux_parabolic_z_node = get_node_vars(filtered_flux_parabolic_z, equations, dg, i, j, k, element)

                multiply_multiply_add_to_node_vars!(flux_parabolic_x, ecav_coefficient, svv_coefficient, filtered_flux_parabolic_x_node, 
                                        equations, dg, i, j, k, element)
                multiply_multiply_add_to_node_vars!(flux_parabolic_y, ecav_coefficient, svv_coefficient, filtered_flux_parabolic_y_node, 
                                        equations, dg, i, j, k, element)
                multiply_multiply_add_to_node_vars!(flux_parabolic_z, ecav_coefficient, svv_coefficient, filtered_flux_parabolic_z_node, 
                                        equations, dg, i, j, k, element)
            end
        end
        return nothing
    end

    function calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual,
                                     equations, mesh::TreeMesh{3}, dg, cache)
        flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic
        gradients_x, gradients_y, gradients_z = gradients

        @threaded for element in eachelement(dg, cache)
            volume_jacobian_ = volume_jacobian(element, mesh, cache)

            # calculate viscous dissipation (ECAV denominator)
            element_viscous_dissipation = zero(real(dg))

            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                flux_viscous_x_node = get_node_vars(flux_parabolic_x, equations, dg, i, j, k, 
                                                    element)
                flux_viscous_y_node = get_node_vars(flux_parabolic_y, equations, dg, i, j, k, 
                                                    element)
                flux_viscous_z_node = get_node_vars(flux_parabolic_z, equations, dg, i, j, k, 
                                                    element)
                gradients_x_node = get_node_vars(gradients_x, equations, dg, i, j, k, element)
                gradients_y_node = get_node_vars(gradients_y, equations, dg, i, j, k, element)
                gradients_z_node = get_node_vars(gradients_z, equations, dg, i, j, k, element)
                viscous_dissipation_x = dot(flux_viscous_x_node, gradients_x_node)
                viscous_dissipation_y = dot(flux_viscous_y_node, gradients_y_node)
                viscous_dissipation_z = dot(flux_viscous_z_node, gradients_z_node)

                weight_ijk = dg.basis.weights[i] * dg.basis.weights[j] * dg.basis.weights[k]
                element_viscous_dissipation = element_viscous_dissipation +
                                              (viscous_dissipation_x +
                                               viscous_dissipation_y + 
                                                viscous_dissipation_z) * weight_ijk *
                                              volume_jacobian_
            end
            # Scale viscous flux by ecav coefficient.
            # Note: we usually use "-min(0, entropy_residual)" to define the ECAV coefficient, but we
            # flip the sign to account for the fact that viscous terms are negated by convention in Trixi.jl.
            ecav_coefficient = regularized_ratio(min(0, entropy_residual[element]),
                                                 element_viscous_dissipation)
            #ecav_coefficient = 0.0;   
            #cache.artificial_viscosity.norm_coefficients[end] += ecav_coefficient * ecav_coefficient        
            cache.artificial_viscosity.coefficients[element] = -ecav_coefficient # save output
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                # flux_viscous_x_node = get_node_vars(flux_parabolic_x, equations, dg, i, j, k, 
                #                                     element)
                # flux_viscous_y_node = get_node_vars(flux_parabolic_y, equations, dg, i, j, k, 
                #                                     element)
                # flux_viscous_z_node = get_node_vars(flux_parabolic_z, equations, dg, i, j, k, 
                #                                     element)
                # set_node_vars!(flux_parabolic_x, ecav_coefficient * flux_viscous_x_node,
                #                equations, dg, i, j, k, element)
                # set_node_vars!(flux_parabolic_y, ecav_coefficient * flux_viscous_y_node,
                #                equations, dg, i, j, k, element)
                # set_node_vars!(flux_parabolic_z, ecav_coefficient * flux_viscous_z_node,
                #                equations, dg, i, j, k, element)
                multiply_to_node_vars!(flux_parabolic_x, ecav_coefficient, equations, dg, i, j, k, 
                                                    element)
                multiply_to_node_vars!(flux_parabolic_y, ecav_coefficient, equations, dg, i, j, k, 
                                                    element)
                multiply_to_node_vars!(flux_parabolic_z, ecav_coefficient, equations, dg, i, j, k, 
                                                    element)
            end
        end
        #cache.artificial_viscosity.norm_coefficients[end] = sqrt(cache.artificial_viscosity.norm_coefficients[end])
        #push!(cache.artificial_viscosity.max_coeff, maximum(cache.artificial_viscosity.coefficients))
        return nothing
    end
 
    function calc_ecav_ducros_coefficients!(u, sensor, 
            flux_parabolic, gradients, entropy_residual,
                    equations, equations_parabolic,
                    mesh::TreeMesh{3}, dg, cache)
        #push!(cache.artificial_viscosity.norm_coefficients, 0.0)
        flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic
        gradients_x, gradients_y, gradients_z = gradients

        set_zero!(sensor, dg, cache)
        derivative_matrix = dg.basis.derivative_matrix

        @threaded for element in eachelement(dg, cache)
            volume_jacobian_ = volume_jacobian(element, mesh, cache)

            # calculate viscous dissipation (ECAV denominator)
            element_viscous_dissipation = zero(real(dg))
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                flux_viscous_x_node = get_node_vars(flux_parabolic_x, equations, dg, i, j, k, 
                                                    element)
                flux_viscous_y_node = get_node_vars(flux_parabolic_y, equations, dg, i, j, k, 
                                                    element)
                flux_viscous_z_node = get_node_vars(flux_parabolic_z, equations, dg, i, j, k, 
                                                    element)
                gradients_x_node = get_node_vars(gradients_x, equations, dg, i, j, k, element)
                gradients_y_node = get_node_vars(gradients_y, equations, dg, i, j, k, element)
                gradients_z_node = get_node_vars(gradients_z, equations, dg, i, j, k, element)
                viscous_dissipation_x = dot(flux_viscous_x_node, gradients_x_node)
                viscous_dissipation_y = dot(flux_viscous_y_node, gradients_y_node)
                viscous_dissipation_z = dot(flux_viscous_z_node, gradients_z_node)


                weight_ijk = dg.basis.weights[i] * dg.basis.weights[j] * dg.basis.weights[k]
                element_viscous_dissipation = element_viscous_dissipation +
                                              (viscous_dissipation_x +
                                               viscous_dissipation_y + 
                                                viscous_dissipation_z) * weight_ijk *
                                              volume_jacobian_
                ### For Ducros sensor, added here for efficiency, for simplicity, should be in a
                ### separate function call

                divb = zero(eltype(u))
                for l in eachnode(dg)
                    u_ljk = get_node_vars(u, equations, dg, l, j, k, element)
                    u_ilk = get_node_vars(u, equations, dg, i, l, k, element)
                    u_ijl = get_node_vars(u, equations, dg, i, j, l, element)

                    v_ljk = velocity(u_ljk, equations)
                    v_ilk = velocity(u_ilk, equations)
                    v_ijl = velocity(u_ijl, equations)

                    divb += (derivative_matrix[i, l] * v_ljk[1] +
                            derivative_matrix[j, l] * v_ilk[2] +
                            derivative_matrix[k, l] * v_ijl[3])
                end
                divb *= cache.elements.inverse_jacobian[element]
                add_constant_to_vars!(sensor, divb * divb, equations, dg, i, j, k, element)
                u_node = get_node_vars(u, equations, dg, i, j, k, element)
                gradients_node = (gradients_x_node, gradients_y_node, gradients_z_node)
                v = norm(vorticity(u_node, gradients_node, equations_parabolic))
                ducros_to_node_vars!(sensor, v * v, equations, dg, i, j, k, element)
            end

            # Scale viscous flux by ecav coefficient.
            # Note: we usually use "-min(0, entropy_residual)" to define the ECAV coefficient, but we
            # flip the sign to account for the fact that viscous terms are negated by convention in Trixi.jl.
            ecav_coefficient = regularized_ratio(min(0, entropy_residual[element]),
                                                 element_viscous_dissipation)
            #ecav_coefficient = 0.0;   
            cache.artificial_viscosity.coefficients[element] = -ecav_coefficient # save output
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                ecav_coefficient = get_node_vars(sensor, equations, dg, i, j, k, element)[1] * ecav_coefficient
                multiply_to_node_vars!(flux_parabolic_x, ecav_coefficient, equations, dg, i, j, k, 
                                                    element)
                multiply_to_node_vars!(flux_parabolic_y, ecav_coefficient, equations, dg, i, j, k, 
                                                    element)
                multiply_to_node_vars!(flux_parabolic_z, ecav_coefficient, equations, dg, i, j, k, 
                                                    element)
            end
        end
        #cache.artificial_viscosity.norm_coefficients[end] = sqrt(cache.artificial_viscosity.norm_coefficients[end])
        return nothing
    end

    function rhs_artificial_viscosity!(du, u, t, mesh::TreeMesh{3},
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
        #@show cache.artificial_viscosity.norm_residuals
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

        calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual, equations, mesh,
                                dg, cache)

        @trixi_timeit timer() "calc divergence" calc_divergence!(du, flux_parabolic, u, mesh,
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

    function rhs_combined!(du, u, t, mesh::TreeMesh{3},
                           equations, equations_parabolic, equations_artificial_viscosity,
                           boundary_conditions, boundary_conditions_parabolic,
                           source_terms::Source,
                           dg::DG, parabolic_scheme, cache, cache_parabolic) where {Source}             
        (; u_transformed, flux_parabolic, gradients, filtered_gradients,
            filtered_parabolic, tmp_gradient, tmp_parabolic) = cache_parabolic.parabolic_container
        (; ecav_choice) = cache
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

        #cache.artificial_viscosity.norm_residuals[end] = sqrt(cache.artificial_viscosity.norm_residuals[end])

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
            prolong2boundaries!(backend, cache, u, mesh, equations, dg)
        end

        # Calculate boundary fluxes
        @trixi_timeit_ext backend timer() "boundary flux" begin
            calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
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

        # ========= AV specific part ============

        @trixi_timeit timer() "calculate AV viscous fluxes" begin
            calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                 equations_artificial_viscosity, dg, cache)
        end
        if (ecav_choice == :ecav)
            calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual, equations, mesh,
                                    dg, cache)
        elseif (ecav_choice == :ecav_svv)
            calc_ecav_svv_coefficients!(flux_parabolic, gradients, 
                filtered_parabolic, filtered_gradients, tmp_gradient, tmp_parabolic,
                entropy_residual, equations, mesh, dg, cache)
        elseif (ecav_choice == :ecav_ducros)
            (; sensor) = cache.artificial_viscosity
            calc_ecav_ducros_coefficients!(u, sensor, 
                flux_parabolic, gradients, entropy_residual, equations, 
                equations_parabolic, mesh, dg, cache)
        end
       
        # # TODO: accumulate into flux_viscous instead
        # # accumulate the AV term
        # @trixi_timeit timer() "calc AV divergence" calc_divergence!(du, flux_viscous, u, mesh, 
        #                                                             equations_artificial_viscosity, 
        #                                                             boundary_conditions_parabolic, # TODO: check right thing to do here
        #                                                             # BoundaryConditionDoNothing(), 
        #                                                             dg, parabolic_scheme, cache, t)

        # ======== physical parabolic part ==========

        # accumulate physical viscous fluxes    
        @trixi_timeit timer() "calculate viscous fluxes" begin
            accum_viscous_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                  equations_parabolic, dg, cache)
        end        

        # TODO: fix BCs for equations_artificial_viscosity
        @trixi_timeit timer() "calc divergence" calc_divergence!(du, flux_parabolic, u, mesh,
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

    function accum_viscous_fluxes!(flux_parabolic,
                                   gradients, u_transformed,
                                   mesh::Union{TreeMesh{3}, P4estMesh{3}},
                                   equations_parabolic::AbstractEquationsParabolic,
                                   dg::DG, cache)
        gradients_1, gradients_2, gradients_3 = gradients
        flux_parabolic_1, flux_parabolic_2, flux_parabolic_3 = flux_parabolic # output arrays

        @threaded for element in eachelement(dg, cache)
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                # Get solution and gradients
                u_node = get_node_vars(u_transformed, equations_parabolic, dg,
                                       i, j, k, element)
                gradients_1_node = get_node_vars(gradients_1, equations_parabolic, dg,
                                                 i, j, k, element)
                gradients_2_node = get_node_vars(gradients_2, equations_parabolic, dg,
                                                 i, j, k, element)
                gradients_3_node = get_node_vars(gradients_3, equations_parabolic, dg,
                                                 i, j, k, element)

                # Calculate viscous flux and store each component for later use
                flux_viscous_node_x = flux(u_node, (gradients_1_node, gradients_2_node, gradients_3_node), 1,
                                           equations_parabolic)
                flux_viscous_node_y = flux(u_node, (gradients_1_node, gradients_2_node, gradients_3_node), 2,
                                           equations_parabolic)
                flux_viscous_node_z = flux(u_node, (gradients_1_node, gradients_2_node, gradients_3_node), 3,
                                           equations_parabolic)

                # flip sign for Trixi's parabolic convention
                add_to_node_vars!(flux_parabolic_1, -flux_viscous_node_x, equations_parabolic,
                                  dg,
                                  i, j, k, element)
                add_to_node_vars!(flux_parabolic_2, -flux_viscous_node_y, equations_parabolic,
                                  dg,
                                  i, j, k, element)
                add_to_node_vars!(flux_parabolic_3, -flux_viscous_node_z, equations_parabolic,
                                  dg,
                                  i, j, k, element)
            end
        end

        return nothing
    end
end # @muladd