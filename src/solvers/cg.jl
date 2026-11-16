# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Continuous Galerkin spectral element methods (CGSEM) build on the DGSEM kernels
# and containers of the `P4estMesh`. The `CGSEM` type is defined first since the
# gather-scatter functions dispatch on it.
include("cgsem/cgsem.jl")
include("cgsem/assembly.jl")

# Hyperbolic right-hand side of CGSEM on a conforming `P4estMesh{2}`, cf. the DGSEM
# version in `dgsem_tree/dg_2d.jl`. The differences are:
# - Interface and mortar terms are skipped; continuity is enforced by the assembly.
# - The interior entries of `surface_flux_values` are zero (see
#   `create_cache_hyperbolic`), so the surface integral adds only the weakly imposed
#   boundary fluxes.
# - The source terms are added before the assembly, see `calc_sources!` below.
# - `assemble_and_apply_jacobian!` replaces `apply_jacobian!`.
function rhs_hyperbolic!(backend::Nothing, du, u, t, mesh::P4estMesh{2}, equations,
                         boundary_conditions, source_terms::Source, solver::CGSEM,
                         cache) where {Source}
    (; dg) = solver

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" begin
        set_zero!(du, dg, cache)
    end

    # Calculate volume integral. The DGSEM kernels include the element-boundary term
    # -M^{-1} B f(u) of the weak form, which cancels between neighboring elements for
    # continuous u once the residuals are summed in the assembly.
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(backend, du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(backend, cache, u, mesh, equations, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate surface integrals. Only the boundary faces contribute, since the
    # interior entries of `surface_flux_values` are zero.
    # TODO: Revisit this soon. The DGSEM kernel `calc_surface_integral!` loops over all
    # faces of all elements, so most of its work adds zeros at interior faces. We reuse
    # it for now since it keeps the option of adding surface terms at interior faces
    # later (e.g., for stabilization). Without such terms, a surface integral over the
    # boundary faces only would avoid the unnecessary work.
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(backend, du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    # Calculate source terms before assembly/Jacobian application
    @trixi_timeit timer() "source terms" begin
        calc_sources!(backend, du, u, t, source_terms, equations, solver, cache)
    end

    # Sum the residuals of all copies of each node and apply the inverse of the
    # assembled lumped mass matrix
    @trixi_timeit timer() "assembly" begin
        assemble_and_apply_jacobian!(du, cache.assembly, mesh, equations, solver, cache)
    end

    return nothing
end

# Source terms of CGSEM. DGSEM adds them after `apply_jacobian!`, i.e., to the final
# time derivative. CGSEM adds them to the element-local residuals before the assembly,
# which scales the residuals by `-1 / sum_k J_k` (see `assemble_and_apply_jacobian!`),
# so the source terms are multiplied by `-J` here. The assembled time derivative then
# contains the Jacobian-weighted average of the source terms of all copies of a node,
# which is the same for all copies even though their node coordinates (and thus the
# source terms) differ by roundoff.
function calc_sources!(backend::Nothing, du, u, t, source_terms::Nothing,
                       equations::AbstractEquations{2}, solver::CGSEM, cache)
    return nothing
end

function calc_sources!(backend::Nothing, du, u, t, source_terms,
                       equations::AbstractEquations{2}, solver::CGSEM, cache)
    (; dg) = solver
    (; node_coordinates, inverse_jacobian) = cache.elements

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(u, Val(2), equations, dg, cache)
        check_axes(du, Val(2), equations, dg, cache)
        check_axes(cache.elements, equations, dg, cache)
    end

    @threaded for element in eachelement(dg, cache)
        @inbounds begin
            for j in eachnode(dg), i in eachnode(dg)
                u_local = get_node_vars(u, equations, dg, i, j, element)
                x_local = get_node_coords(node_coordinates, equations, dg,
                                          i, j, element)
                source_local = source_terms(u_local, x_local, t, equations)
                factor = -inv(inverse_jacobian[i, j, element])
                multiply_add_to_node_vars!(du, factor, source_local, equations, dg,
                                           i, j, element)
            end
        end
    end

    return nothing
end

###############################################################################
# Parabolic terms

# Parabolic right-hand side of CGSEM on a conforming `P4estMesh{2}`, cf. the DGSEM
# version in `dgsem_p4est/dg_2d_parabolic.jl`. The differences are:
# - The gradients are element-local strong form derivatives of the continuous solution
#   plus a lift of the boundary data on the boundary faces, see `calc_gradient_cgsem!`.
#   They are not assembled.
# - Interface and mortar terms are skipped. As in `rhs_hyperbolic!`, the interior
#   entries of `surface_flux_values` are zero, so the surface integrals add only the
#   boundary terms.
# - The parabolic source terms are added before the assembly.
# - `assemble_and_apply_jacobian_parabolic!` replaces `apply_jacobian_parabolic!`.
function rhs_parabolic!(backend::Nothing, du, u, t, mesh::P4estMesh{2},
                        equations_parabolic::AbstractEquationsParabolic,
                        boundary_conditions_parabolic, source_terms_parabolic,
                        solver::CGSEM,
                        parabolic_scheme::ParabolicFormulationBassiRebay1,
                        cache, cache_parabolic)
    (; dg) = solver
    (; u_transformed, gradients, flux_parabolic) = cache_parabolic.parabolic_container

    # Convert conservative variables to the gradient variables
    @trixi_timeit timer() "transform variables" begin
        transform_variables!(u_transformed, u, mesh, equations_parabolic, dg, cache)
    end

    # Compute the element-local gradients of the gradient variables
    @trixi_timeit timer() "calculate gradient" begin
        calc_gradient_cgsem!(gradients, u_transformed, t, mesh, equations_parabolic,
                             boundary_conditions_parabolic, dg, cache)
    end

    # Compute and store the parabolic fluxes
    @trixi_timeit timer() "calculate parabolic fluxes" begin
        calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                               equations_parabolic, dg, cache)
    end

    # The remainder computes the divergence of the parabolic fluxes. As in DGSEM, the
    # boundary container in `cache` stores the normal parabolic fluxes at the boundary.

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" set_zero!(du, dg, cache)

    # Calculate the weak form volume integral of the parabolic fluxes. Summed over the
    # copies of a node in the assembly, this is the continuous Galerkin weak form,
    # which has no interior face terms.
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, flux_parabolic, mesh, equations_parabolic, dg, cache)
    end

    # Prolong the normal parabolic fluxes to the boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, flux_parabolic, mesh, equations_parabolic, dg)
    end

    # Calculate boundary fluxes, which contain the Neumann data
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_divergence!(cache, t, boundary_conditions_parabolic, mesh,
                                       equations_parabolic, dg.surface_integral, dg)
    end

    # Calculate surface integrals (boundary faces only). This reuses the DGSEM kernel,
    # which adds zeros at the interior faces; see the TODO in `rhs_hyperbolic!`.
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(nothing, du, u, mesh, equations_parabolic,
                               dg.surface_integral, dg, cache)
    end

    # Calculate parabolic source terms before the assembly
    @trixi_timeit timer() "source terms parabolic" begin
        calc_sources_parabolic_cgsem!(du, u, gradients, t, source_terms_parabolic,
                                      equations_parabolic, dg, cache)
    end

    # Sum the residuals of all copies of each node and apply the inverse of the
    # assembled lumped mass matrix
    @trixi_timeit timer() "assembly" begin
        assemble_and_apply_jacobian_parabolic!(du, cache.assembly, mesh,
                                               equations_parabolic, dg, cache)
    end

    return nothing
end

# The parabolic formulation of DGSEM only determines the interface fluxes, which CGSEM
# does not compute. Thus, other formulations such as `ParabolicFormulationLocalDG` would
# silently behave like BR1 and are rejected.
function rhs_parabolic!(backend::Nothing, du, u, t, mesh::P4estMesh{2},
                        equations_parabolic::AbstractEquationsParabolic,
                        boundary_conditions_parabolic, source_terms_parabolic,
                        solver::CGSEM, parabolic_scheme, cache, cache_parabolic)
    throw(ArgumentError("CGSEM supports only `ParabolicFormulationBassiRebay1`, " *
                        "got `$(nameof(typeof(parabolic_scheme)))`"))
end

# Gradients of the gradient variables for CGSEM. In DGSEM, the weak form volume term
# `calc_volume_integral_gradient!` contains the boundary term -M^{-1} B u, and
# `calc_surface_integral_gradient!` adds M^{-1} B u^* with the numerical flux u^*.
# Together, this is the strong form derivative plus the lift M^{-1} B (u^* - u). Since
# u is continuous, u^* = u at the interior faces, so only the strong form derivative
# remains there. At the boundary faces, u^* is the boundary value of the boundary
# condition, which imposes Dirichlet data weakly (for Neumann conditions, u^* = u).
function calc_gradient_cgsem!(gradients, u_transformed, t, mesh::P4estMesh{2},
                              equations_parabolic, boundary_conditions_parabolic,
                              dg::DGSEM, cache)
    # Reset gradients
    @trixi_timeit timer() "reset gradients" begin
        reset_gradients!(gradients, dg, cache)
    end

    # Calculate the element-local strong form derivatives
    @trixi_timeit timer() "volume integral" begin
        calc_gradient_strong!(gradients, u_transformed, mesh, equations_parabolic, dg,
                              cache)
    end

    # Prolong the gradient variables to the boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(nothing, cache, u_transformed, mesh, equations_parabolic,
                            dg)
    end

    # Calculate the jumps u^* - u of the gradient variables at the boundaries
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_gradient!(cache, t, boundary_conditions_parabolic, mesh,
                                     equations_parabolic, dg.surface_integral, dg)
        calc_boundary_jump_gradient!(cache, mesh, equations_parabolic, dg)
    end

    # Add the lift of the boundary jumps. The interior entries of `surface_flux_values`
    # are zero, so only the boundary faces contribute.
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral_gradient!(gradients, mesh, equations_parabolic, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(gradients, mesh, equations_parabolic, dg, cache)
    end

    return nothing
end

# Element-local strong form derivatives of `u_transformed`, multiplied by the Jacobian
# (which is divided out by `apply_jacobian_parabolic!`). This is DGSEM's
# `calc_volume_integral_gradient!` with the derivative matrix D in place of the weak
# form derivative matrix `derivative_hat` = D - M^{-1} B.
function calc_gradient_strong!(gradients, u_transformed, mesh::P4estMesh{2},
                               equations_parabolic::AbstractEquationsParabolic,
                               dg::DGSEM, cache)
    (; derivative_matrix) = dg.basis
    (; contravariant_vectors) = cache.elements
    gradients_x, gradients_y = gradients

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(u_transformed, mesh, equations_parabolic, dg, cache)
        check_axes(gradients_x, mesh, equations_parabolic, dg, cache)
        check_axes(gradients_y, mesh, equations_parabolic, dg, cache)
        check_axes(cache.elements, equations_parabolic, dg, cache)
    end

    @threaded for element in eachelement(dg, cache)
        @inbounds begin
            # Derivatives with respect to the reference coordinates
            for j in eachnode(dg), i in eachnode(dg)
                u_node = get_node_vars(u_transformed, equations_parabolic, dg,
                                       i, j, element)

                for ii in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_x, derivative_matrix[ii, i],
                                               u_node, equations_parabolic, dg,
                                               ii, j, element)
                end

                for jj in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_y, derivative_matrix[jj, j],
                                               u_node, equations_parabolic, dg,
                                               i, jj, element)
                end
            end

            # Transform them node by node to physical gradients (times the Jacobian)
            # using the contravariant vectors, as in `calc_volume_integral_gradient!`
            for j in eachnode(dg), i in eachnode(dg)
                Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors,
                                                      i, j, element)
                Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors,
                                                      i, j, element)

                gradients_reference_1 = get_node_vars(gradients_x, equations_parabolic,
                                                      dg, i, j, element)
                gradients_reference_2 = get_node_vars(gradients_y, equations_parabolic,
                                                      dg, i, j, element)

                gradient_x_node = Ja11 * gradients_reference_1 +
                                  Ja21 * gradients_reference_2
                gradient_y_node = Ja12 * gradients_reference_1 +
                                  Ja22 * gradients_reference_2

                set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg,
                               i, j, element)
                set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg,
                               i, j, element)
            end
        end
    end

    return nothing
end

# `calc_boundary_flux_gradient!` stores the numerical fluxes u^* of the gradient
# variables at the boundaries in `surface_flux_values`. Subtract the boundary values u
# of the gradient variables, so that `calc_surface_integral_gradient!` adds the lift of
# the jumps u^* - u instead of the weak form surface term of DGSEM.
function calc_boundary_jump_gradient!(cache, mesh::P4estMesh{2},
                                      equations_parabolic::AbstractEquationsParabolic,
                                      dg::DGSEM)
    (; boundaries) = cache
    (; surface_flux_values) = cache.elements

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(boundaries, equations_parabolic, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh, equations_parabolic,
                                       dg, cache)
    end

    @threaded for boundary in eachboundary(dg, cache)
        @inbounds begin
            element = boundaries.neighbor_ids[boundary]
            direction = indices2direction(boundaries.node_indices[boundary])

            # `surface_flux_values` and `boundaries.u` use the same node ordering
            for i in eachnode(dg), v in eachvariable(equations_parabolic)
                surface_flux_values[v, i, direction, element] -= boundaries.u[v, i,
                                                                              boundary]
            end
        end
    end

    return nothing
end

# Parabolic source terms of CGSEM. As for `calc_sources_cgsem!`, they are added to the
# element-local residuals before the assembly, which scales the residuals by
# `1 / sum_k J_k` (see `assemble_and_apply_jacobian_parabolic!`), so the source terms
# are multiplied by `J` here.
function calc_sources_parabolic_cgsem!(du, u, gradients, t,
                                       source_terms_parabolic::Nothing,
                                       equations_parabolic::AbstractEquations{2},
                                       dg::DGSEM, cache)
    return nothing
end

function calc_sources_parabolic_cgsem!(du, u, gradients, t, source_terms_parabolic,
                                       equations_parabolic::AbstractEquations{2},
                                       dg::DGSEM, cache)
    (; node_coordinates, inverse_jacobian) = cache.elements
    gradients_x, gradients_y = gradients

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(u, Val(2), equations_parabolic, dg, cache)
        check_axes(du, Val(2), equations_parabolic, dg, cache)
        check_axes(gradients_x, Val(2), equations_parabolic, dg, cache)
        check_axes(gradients_y, Val(2), equations_parabolic, dg, cache)
        check_axes(cache.elements, equations_parabolic, dg, cache)
    end

    @threaded for element in eachelement(dg, cache)
        @inbounds begin
            for j in eachnode(dg), i in eachnode(dg)
                u_local = get_node_vars(u, equations_parabolic, dg, i, j, element)
                gradients_x_local = get_node_vars(gradients_x, equations_parabolic, dg,
                                                  i, j, element)
                gradients_y_local = get_node_vars(gradients_y, equations_parabolic, dg,
                                                  i, j, element)
                gradients_local = (gradients_x_local, gradients_y_local)
                x_local = get_node_coords(node_coordinates, equations_parabolic, dg,
                                          i, j, element)
                source_local = source_terms_parabolic(u_local, gradients_local, x_local,
                                                      t, equations_parabolic)
                factor = inv(inverse_jacobian[i, j, element])
                multiply_add_to_node_vars!(du, factor, source_local,
                                           equations_parabolic, dg, i, j, element)
            end
        end
    end

    return nothing
end
end # @muladd
