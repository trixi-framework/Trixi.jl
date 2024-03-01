# !!! warning "Experimental implementation (curvilinear FDSBP)"
#     This is an experimental feature and may change in future releases.

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# 2D unstructured cache
function create_cache(mesh::UnstructuredMesh2D, equations, dg::FDSBP, RealT, uEltype)
    elements = init_elements(mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(mesh, elements)

    boundaries = init_boundaries(mesh, elements)

    cache = (; elements, interfaces, boundaries)

    # Add specialized parts of the cache required to for efficient flux computations
    cache = (; cache...,
             create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)

    return cache
end

# TODO: FD; Upwind versions of surface / volume integral

# 2D volume integral contributions for `VolumeIntegralStrongForm`
# OBS! This is the standard (not de-aliased) form of the volume integral.
# So it is not provably stable for variable coefficients due to the the metric terms.
@inline function calc_volume_integral!(du, u,
                                       mesh::UnstructuredMesh2D,
                                       nonconservative_terms::False, equations,
                                       volume_integral::VolumeIntegralStrongForm,
                                       dg::FDSBP, cache)
    D = dg.basis # SBP derivative operator
    @unpack f_threaded = cache
    @unpack contravariant_vectors = cache.elements

    # SBP operators from SummationByPartsOperators.jl implement the basic interface
    # of matrix-vector multiplication. Thus, we pass an "array of structures",
    # packing all variables per node in an `SVector`.
    if nvariables(equations) == 1
        # `reinterpret(reshape, ...)` removes the leading dimension only if more
        # than one variable is used.
        u_vectors = reshape(reinterpret(SVector{nvariables(equations), eltype(u)}, u),
                            nnodes(dg), nnodes(dg), nelements(dg, cache))
        du_vectors = reshape(reinterpret(SVector{nvariables(equations), eltype(du)},
                                         du),
                             nnodes(dg), nnodes(dg), nelements(dg, cache))
    else
        u_vectors = reinterpret(reshape, SVector{nvariables(equations), eltype(u)}, u)
        du_vectors = reinterpret(reshape, SVector{nvariables(equations), eltype(du)},
                                 du)
    end

    # Use the tensor product structure to compute the discrete derivatives of
    # the contravariant fluxes line-by-line and add them to `du` for each element.
    @threaded for element in eachelement(dg, cache)
        f_element = f_threaded[Threads.threadid()]
        u_element = view(u_vectors, :, :, element)

        # x direction
        for j in eachnode(dg)
            for i in eachnode(dg)
                Ja1 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
                f_element[i, j] = flux(u_element[i, j], Ja1, equations)
            end
            mul!(view(du_vectors, :, j, element), D, view(f_element, :, j),
                 one(eltype(du)), one(eltype(du)))
        end

        # y direction
        for i in eachnode(dg)
            for j in eachnode(dg)
                Ja2 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
                f_element[i, j] = flux(u_element[i, j], Ja2, equations)
            end
            mul!(view(du_vectors, i, :, element), D, view(f_element, i, :),
                 one(eltype(du)), one(eltype(du)))
        end
    end

    return nothing
end

# 2D volume integral contributions for `VolumeIntegralUpwind`.
# Note that the plus / minus notation of the operators does not refer to the
# upwind / downwind directions of the fluxes.
# Instead, the plus / minus refers to the direction of the biasing within
# the finite difference stencils. Thus, the D^- operator acts on the positive
# part of the flux splitting f^+ and the D^+ operator acts on the negative part
# of the flux splitting f^-.
function calc_volume_integral!(du, u,
                               mesh::UnstructuredMesh2D,
                               nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralUpwind,
                               dg::FDSBP, cache)

    # ######################
    # # This is an attempt at the encapsulated version of the USBP operators. The standard way
    # # of doing this from Åland cannot be done because it is NOT true that
    # #   ftilde^\pm = Ja_1 f1^\pm + Ja_2 f2^\pm
    # # Thus, we explicitly separate the conservative and nonconserative forms of the mapped
    # # fluxes. Earlier experiments with the nonconservative form had issues with rotated elements
    # # which I suspect will still be the case here. This still appears to be an issue.
    # # The other issue is that the conservative form is not FSP and this already spoils any hope
    # # that this strategy would be FSP as well.
    # ######################

    # @unpack f_minus_plus_threaded, f_minus_threaded, f_plus_threaded = cache
    # @unpack splitting = volume_integral
    # @unpack contravariant_vectors, contravariant_vectors_plus, contravariant_vectors_minus = cache.elements

    # D_minus = Matrix(dg.basis.minus) # Upwind SBP D^- derivative operator
    # D_plus = Matrix(dg.basis.plus)   # Upwind SBP D^+ derivative operator

    # # Use the tensor product structure to compute the discrete derivatives of
    # # the fluxes line-by-line and add them to `du` for each element.
    # @threaded for element in eachelement(dg, cache)
    #     for j in eachnode(dg), i in eachnode(dg)
    #         u_node = get_node_vars(u, equations, dg, i, j, element)

    #         # Grab the contravariant vectors Ja^1 computed with D^+ and D^-
    #         Ja1_plus = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    #         Ja1_minus = get_contravariant_vector(1, contravariant_vectors, i, j, element)

    #         # Grab the contravariant vectors Ja^2 computed with D^+ and D^-
    #         Ja2_plus = get_contravariant_vector(2, contravariant_vectors, i, j, element)
    #         Ja2_minus = get_contravariant_vector(2, contravariant_vectors, i, j, element)

    #         # specialized FVS terms that use different combinations of the biased metric terms
    #         ftilde1_minus = splitting(u_node, Val{:minus}(), Ja1_plus, equations)
    #         ftilde1_plus = splitting(u_node, Val{:plus}(), Ja1_minus, equations)
    #         f1_minus, f1_plus = splitting(u_node, 1, equations)

    #         ftilde2_minus = splitting(u_node, Val{:minus}(), Ja2_plus, equations)
    #         ftilde2_plus = splitting(u_node, Val{:plus}(), Ja2_minus, equations)
    #         f2_minus, f2_plus = splitting(u_node, 2, equations)

    #         # xi-direction
    #         for ii in eachnode(dg)
    #             # Term: 1/2 D^+ ftilde1_minus
    #             multiply_add_to_node_vars!(du, 0.5*D_plus[ii, i], ftilde1_minus, equations, dg, ii, j, element)
    #             # Term: 1/2 D^- ftilde1_plus
    #             multiply_add_to_node_vars!(du, 0.5*D_minus[ii, i], ftilde1_plus, equations, dg, ii, j, element)
    #             # Term: 1/2 Ja11^+ D^+ f1_minus
    #             multiply_add_to_node_vars!(du, 0.5*Ja1_plus[1]*D_plus[ii, i], f1_minus, equations, dg, ii, j, element)
    #             # Term: 1/2 Ja11^- D^- f1_plus
    #             multiply_add_to_node_vars!(du, 0.5*Ja1_minus[1]*D_minus[ii, i], f1_plus, equations, dg, ii, j, element)
    #             # Term: 1/2 Ja12^+ D^+ f2_minus
    #             multiply_add_to_node_vars!(du, 0.5*Ja1_plus[2]*D_plus[ii, i], f2_minus, equations, dg, ii, j, element)
    #             # Term: 1/2 Ja12^- D^- f2_plus
    #             multiply_add_to_node_vars!(du, 0.5*Ja1_minus[2]*D_minus[ii, i], f2_plus, equations, dg, ii, j, element)
    #         end

    #         # eta-direction
    #         for jj in eachnode(dg)
    #             # Term: 1/2 D^+ ftilde2_minus
    #             multiply_add_to_node_vars!(du, 0.5*D_plus[jj, j], ftilde2_minus, equations, dg, i, jj, element)
    #             # Term: 1/2 D^- ftilde2_plus
    #             multiply_add_to_node_vars!(du, 0.5*D_minus[jj, j], ftilde2_plus, equations, dg, i, jj, element)
    #             # Term: 1/2 Ja21^+ D^+ f1_minus
    #             multiply_add_to_node_vars!(du, 0.5*Ja2_plus[1]*D_plus[jj, j], f1_minus, equations, dg, i, jj, element)
    #             # Term: 1/2 Ja21^+ D^- f1_plus
    #             multiply_add_to_node_vars!(du, 0.5*Ja2_minus[1]*D_minus[jj, j], f1_plus, equations, dg, i, jj, element)
    #             # Term: 1/2 Ja22^+ D^+ f2_minus
    #             multiply_add_to_node_vars!(du, 0.5*Ja2_plus[2]*D_plus[jj, j], f2_minus, equations, dg, i, jj, element)
    #             # Term: 1/2 Ja22^+ D^- f2_plus
    #             multiply_add_to_node_vars!(du, 0.5*Ja2_minus[2]*D_minus[jj, j], f2_plus, equations, dg, i, jj, element)
    #         end
    #     end
    # end

    # VERSION THAT IS MORE OR LESS WORKING
    # Assume that
    # dg.basis isa SummationByPartsOperators.UpwindOperators
    D_minus = dg.basis.minus # Upwind SBP D^- derivative operator
    D_plus = dg.basis.plus   # Upwind SBP D^+ derivative operator
    D_central = dg.basis.central # Central operator 0.5 (D^+ + D^-)
    @unpack f_minus_plus_threaded, f_minus_threaded, f_plus_threaded = cache
    @unpack splitting = volume_integral
    @unpack contravariant_vectors, contravariant_vectors_plus, contravariant_vectors_minus = cache.elements

    # SBP operators from SummationByPartsOperators.jl implement the basic interface
    # of matrix-vector multiplication. Thus, we pass an "array of structures",
    # packing all variables per node in an `SVector`.
    if nvariables(equations) == 1
        # `reinterpret(reshape, ...)` removes the leading dimension only if more
        # than one variable is used.
        u_vectors = reshape(reinterpret(SVector{nvariables(equations), eltype(u)}, u),
                            nnodes(dg), nnodes(dg), nelements(dg, cache))
        du_vectors = reshape(reinterpret(SVector{nvariables(equations), eltype(du)},
                                         du),
                             nnodes(dg), nnodes(dg), nelements(dg, cache))
    else
        u_vectors = reinterpret(reshape, SVector{nvariables(equations), eltype(u)}, u)
        du_vectors = reinterpret(reshape, SVector{nvariables(equations), eltype(du)},
                                 du)
    end

    # Use the tensor product structure to compute the discrete derivatives of
    # the fluxes line-by-line and add them to `du` for each element.
    @threaded for element in eachelement(dg, cache)
        # f_minus_plus_element wraps the storage provided by f_minus_element and
        # f_plus_element such that we can use a single plain broadcasting below.
        # f_minus_element and f_plus_element are updated in broadcasting calls
        # of the form `@. f_minus_plus_element = ...`.
        f_minus_plus_element = f_minus_plus_threaded[Threads.threadid()]
        f_minus_element = f_minus_threaded[Threads.threadid()]
        f_plus_element = f_plus_threaded[Threads.threadid()]
        u_element = view(u_vectors, :, :, element)

        # x direction
        # @. f_minus_plus_element = splitting(u_element, 1, equations)
        for j in eachnode(dg), i in eachnode(dg)
            # contravariant vector and flux computed with central D matrix
            # OBS! converges for MMS on flipped mesh but not FSP
            Ja1 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
            f_minus_plus_element[i, j] = splitting(u_element[i, j], Ja1, equations)

            # # For testing, use biased metric terms and rotationally invariant splitting
            # # OBS! converges for MMS on flipped mesh but not FSP
            # Ja1 = get_contravariant_vector(1, contravariant_vectors_plus, i, j, element)
            # f_minus_element[i, j] = splitting(u_element[i, j], Val{:minus}(), Ja1, equations)

            # Ja1 = get_contravariant_vector(1, contravariant_vectors_minus, i, j, element)
            # f_plus_element[i, j] = splitting(u_element[i, j], Val{:plus}(), Ja1, equations)

            # # Testing, the f_minus stores the actually flux and f_plus stores the difference
            # # between f^- and f^+
            # Ja1 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
            # f_minus_element[i, j] = flux(u_element[i, j], Ja1, equations)
            # f_plus_element[i, j] = (splitting(u_element[i, j], Val{:minus}(), Ja1, equations)
            #                         - splitting(u_element[i, j], Val{:plus}(), Ja1, equations))
        end

        for j in eachnode(dg)
            # first instinct upwind differences
            mul!(view(du_vectors, :, j, element), D_minus, view(f_plus_element, :, j),
                 one(eltype(du)), one(eltype(du)))
            mul!(view(du_vectors, :, j, element), D_plus, view(f_minus_element, :, j),
                 one(eltype(du)), one(eltype(du)))

            # # use a combination of the central and diffusion matrices that comes from, e.g.,
            # #     D⁺ f⁻ + D⁻ f⁺ = D⁺ f⁻ + D⁻ (f - f⁻)
            # #                   = D⁺ (0.5 f + f⁻ - 0.5 f) + D⁻ (0.5 f - f⁻ + 0.5 f)
            # #                   = 0.5 (D⁺ + D⁻) f + 0.5 (D⁺ - D⁻) (2 f⁻ - f)
            # #                   = Dᶜ f + 0.5 (D⁺ - D⁻) (f⁻ - f⁺)
            # mul!(view(du_vectors, :, j, element), dg.basis.central, view(f_minus_element, :, j),
            #      one(eltype(du)), one(eltype(du)))
            # mul!(view(du_vectors, :, j, element), D_plus, view(f_plus_element, :, j),
            #      0.5*one(eltype(du)), one(eltype(du)))
            # mul!(view(du_vectors, :, j, element), D_minus, view(f_plus_element, :, j),
            #      -0.5*one(eltype(du)), one(eltype(du)))
        end

        # y direction
        # @. f_minus_plus_element = splitting(u_element, 2, equations)
        for j in eachnode(dg), i in eachnode(dg)
            # contravariant vector and flux computed with central D matrix (not FSP)
            Ja2 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
            f_minus_plus_element[i, j] = splitting(u_element[i, j], Ja2, equations)

            # # For testing, use biased metric terms and rotationally invariant splitting
            # Ja2 = get_contravariant_vector(2, contravariant_vectors_plus, i, j, element)
            # f_minus_element[i, j] = splitting(u_element[i, j], Val{:minus}(), Ja2, equations)

            # Ja2 = get_contravariant_vector(2, contravariant_vectors_minus, i, j, element)
            # f_plus_element[i, j] = splitting(u_element[i, j], Val{:plus}(), Ja2, equations)

            # # Testing, the f_minus stores the full flux and f_plus stores the difference
            # # between f^- and f^+
            # Ja2 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
            # f_minus_element[i, j] = flux(u_element[i, j], Ja2, equations)
            # f_plus_element[i, j] = (splitting(u_element[i, j], Val{:minus}(), Ja2, equations)
            #                         - splitting(u_element[i, j], Val{:plus}(), Ja2, equations))
        end

        for i in eachnode(dg)
            # first instinct upwind differences
            mul!(view(du_vectors, i, :, element), D_minus, view(f_plus_element, i, :),
                 one(eltype(du)), one(eltype(du)))
            mul!(view(du_vectors, i, :, element), D_plus, view(f_minus_element, i, :),
                 one(eltype(du)), one(eltype(du)))

            # # use a combination of the central and diffusion matrices
            # mul!(view(du_vectors, i, :, element), dg.basis.central, view(f_minus_element, i, :),
            #      one(eltype(du)), one(eltype(du)))
            # mul!(view(du_vectors, i, :, element), D_plus, view(f_plus_element, i, :),
            #      0.5*one(eltype(du)), one(eltype(du)))
            # mul!(view(du_vectors, i, :, element), D_minus, view(f_plus_element, i, :),
            #      -0.5*one(eltype(du)), one(eltype(du)))
        end
    end

    return nothing
end

# # TODO: Write a better comment here. Specialized computation of the interface
# # values using flux vector splitting in an arbitrary direction
# # TODO: OBS! `SurfaceIntegralUpwind` is buggy and possibly unnecessary.
# function calc_interface_flux!(surface_flux_values,
#                               mesh::UnstructuredMesh2D,
#                               nonconservative_terms::False, equations,
#                               surface_integral::SurfaceIntegralUpwind,
#                               dg::FDSBP, cache)
#     @unpack splitting = surface_integral
#     @unpack u, start_index, index_increment, element_ids, element_side_ids = cache.interfaces
#     @unpack normal_directions, normal_directions_plus, normal_directions_minus = cache.elements

#     @threaded for interface in eachinterface(dg, cache)
#         # Get neighboring elements
#         primary_element = element_ids[1, interface]
#         secondary_element = element_ids[2, interface]

#         # Get the local side id on which to compute the flux
#         primary_side = element_side_ids[1, interface]
#         secondary_side = element_side_ids[2, interface]

#         # initial index for the coordinate system on the secondary element
#         secondary_index = start_index[interface]

#         # loop through the primary element coordinate system and compute the interface coupling
#         for primary_index in eachnode(dg)
#             # Pull the primary and secondary states from the boundary u values
#             u_ll = get_one_sided_surface_node_vars(u, equations, dg, 1, primary_index,
#                                                    interface)
#             u_rr = get_one_sided_surface_node_vars(u, equations, dg, 2, secondary_index,
#                                                    interface)

#             # pull the outward pointing (normal) directional vectors
#             #   Note! this assumes a conforming approximation, more must be done in terms of the normals
#             #         for hanging nodes and other non-conforming approximation spaces
#             outward_direction = get_surface_normal(normal_directions, primary_index,
#                                                    primary_side,
#                                                    primary_element)
#             # outward_direction_plus = get_surface_normal(normal_directions_plus,
#             #                                             primary_index,
#             #                                             primary_side,
#             #                                             primary_element)
#             # outward_direction_minus = get_surface_normal(normal_directions_minus,
#             #                                              primary_index,
#             #                                              primary_side,
#             #                                              primary_element)

#             # Compute the upwind coupling terms where right-traveling
#             # information comes from the left and left-traveling information
#             # comes from the right
#             flux_minus_rr = splitting(u_rr, Val{:minus}(), outward_direction, equations)
#             flux_plus_ll = splitting(u_ll, Val{:plus}(), outward_direction, equations)
#             # flux_minus_rr = splitting(u_rr, Val{:minus}(), outward_direction_plus, equations)
#             # flux_plus_ll = splitting(u_ll, Val{:plus}(), outward_direction_minus, equations)

#             # Copy upwind fluxes back to primary/secondary element storage
#             # Note the sign change for the normal flux in the secondary element!
#             # TODO: do we need this sign flip on the neighbour?
#             # OBS! if we switch the sign flip here then the `SurfaceIntegralUpwind` also needs adjusted
#             for v in eachvariable(equations)
#                 surface_flux_values[v, primary_index, primary_side, primary_element] = flux_minus_rr[v]
#                 surface_flux_values[v, secondary_index, secondary_side, secondary_element] = -flux_plus_ll[v]
#             end

#             # increment the index of the coordinate system in the secondary element
#             secondary_index += index_increment[interface]
#         end
#     end

#     return nothing
# end

# Note! The local side numbering for the unstructured quadrilateral element implementation differs
#       from the structured TreeMesh or StructuredMesh local side numbering:
#
#      TreeMesh/StructuredMesh sides   versus   UnstructuredMesh sides
#                  4                                  3
#          -----------------                  -----------------
#          |               |                  |               |
#          | ^ eta         |                  | ^ eta         |
#        1 | |             | 2              4 | |             | 2
#          | |             |                  | |             |
#          | ---> xi       |                  | ---> xi       |
#          -----------------                  -----------------
#                  3                                  1
# Therefore, we require a different surface integral routine here despite their similar structure.
# Also, the normal directions are already outward pointing for `UnstructuredMesh2D` so all the
# surface contributions are added.
function calc_surface_integral!(du, u, mesh::UnstructuredMesh2D,
                                equations, surface_integral::SurfaceIntegralStrongForm,
                                dg::FDSBP, cache)
    inv_weight_left = inv(left_boundary_weight(dg.basis))
    inv_weight_right = inv(right_boundary_weight(dg.basis))
    @unpack normal_directions, surface_flux_values = cache.elements

    @threaded for element in eachelement(dg, cache)
        for l in eachnode(dg)
            # surface at -x
            u_node = get_node_vars(u, equations, dg, 1, l, element)
            # compute internal flux in normal direction on side 4
            # outward_direction = get_node_coords(normal_directions, equations, dg, l, 4,
            #                                     element)
            # TODO: can probably replace the above with the simpler (same goes with the other instances below)
            outward_direction = get_surface_normal(normal_directions, l, 4, element)
            f_node = flux(u_node, outward_direction, equations)
            f_num = get_node_vars(surface_flux_values, equations, dg, l, 4, element)
            multiply_add_to_node_vars!(du, inv_weight_left, (f_num - f_node),
                                       equations, dg, 1, l, element)

            # surface at +x
            u_node = get_node_vars(u, equations, dg, nnodes(dg), l, element)
            # compute internal flux in normal direction on side 2
            # outward_direction = get_node_coords(normal_directions, equations, dg, l, 2,
            #                                     element)
            outward_direction = get_surface_normal(normal_directions, l, 2, element)
            f_node = flux(u_node, outward_direction, equations)
            f_num = get_node_vars(surface_flux_values, equations, dg, l, 2, element)
            multiply_add_to_node_vars!(du, inv_weight_right, (f_num - f_node),
                                       equations, dg, nnodes(dg), l, element)

            # surface at -y
            u_node = get_node_vars(u, equations, dg, l, 1, element)
            # compute internal flux in normal direction on side 1
            # outward_direction = get_node_coords(normal_directions, equations, dg, l, 1,
            #                                     element)
            outward_direction = get_surface_normal(normal_directions, l, 1, element)
            f_node = flux(u_node, outward_direction, equations)
            f_num = get_node_vars(surface_flux_values, equations, dg, l, 1, element)
            multiply_add_to_node_vars!(du, inv_weight_left, (f_num - f_node),
                                       equations, dg, l, 1, element)

            # surface at +y
            u_node = get_node_vars(u, equations, dg, l, nnodes(dg), element)
            # compute internal flux in normal direction on side 3
            # outward_direction = get_node_coords(normal_directions, equations, dg, l, 3,
            #                                     element)
            outward_direction = get_surface_normal(normal_directions, l, 3, element)
            f_node = flux(u_node, outward_direction, equations)
            f_num = get_node_vars(surface_flux_values, equations, dg, l, 3, element)
            multiply_add_to_node_vars!(du, inv_weight_right, (f_num - f_node),
                                       equations, dg, l, nnodes(dg), element)
        end
    end

    return nothing
end

# # Implementation of fully upwind SATs. The surface flux values are pre-computed
# # in the specialized `calc_interface_flux` routine. These SATs are still of
# # a strong form penalty type, except that the interior flux at a particular
# # side of the element are computed in the upwind direction.
# # TODO: OBS! `SurfaceIntegralUpwind` is buggy and possibly unnecessary.
# function calc_surface_integral!(du, u, mesh::UnstructuredMesh2D,
#                                 equations, surface_integral::SurfaceIntegralUpwind,
#                                 dg::FDSBP, cache)
#     inv_weight_left = inv(left_boundary_weight(dg.basis))
#     inv_weight_right = inv(right_boundary_weight(dg.basis))
#     @unpack normal_directions, normal_directions_plus, normal_directions_minus, surface_flux_values = cache.elements
#     @unpack splitting = surface_integral

#     @threaded for element in eachelement(dg, cache)
#         for l in eachnode(dg)
#             # surface at -x
#             u_node = get_node_vars(u, equations, dg, 1, l, element)
#             # compute internal flux in normal direction on side 4
#             # outward_direction = get_node_coords(normal_directions, equations, dg, l, 4,
#             #                                     element)
#             outward_direction = get_surface_normal(normal_directions, l, 4, element)
#             f_node = splitting(u_node, Val{:plus}(), outward_direction, equations)
#             f_num = get_node_vars(surface_flux_values, equations, dg, l, 4, element)
#             multiply_add_to_node_vars!(du, inv_weight_left, (f_num - f_node),
#                                        equations, dg, 1, l, element)

#             # surface at +x
#             u_node = get_node_vars(u, equations, dg, nnodes(dg), l, element)
#             # compute internal flux in normal direction on side 2
#             # outward_direction = get_node_coords(normal_directions, equations, dg, l, 2,
#             #                                     element)
#             outward_direction = get_surface_normal(normal_directions, l, 2, element)
#             f_node = splitting(u_node, Val{:minus}(), outward_direction, equations)
#             f_num = get_node_vars(surface_flux_values, equations, dg, l, 2, element)
#             multiply_add_to_node_vars!(du, inv_weight_right, (f_num - f_node),
#                                        equations, dg, nnodes(dg), l, element)

#             # surface at -y
#             u_node = get_node_vars(u, equations, dg, l, 1, element)
#             # compute internal flux in normal direction on side 1
#             # outward_direction = get_node_coords(normal_directions, equations, dg, l, 1,
#             #                                     element)
#             outward_direction = get_surface_normal(normal_directions, l, 1, element)
#             f_node = splitting(u_node, Val{:plus}(), outward_direction, equations)
#             f_num = get_node_vars(surface_flux_values, equations, dg, l, 1, element)
#             multiply_add_to_node_vars!(du, inv_weight_left, -(f_num - f_node),
#                                        equations, dg, l, 1, element)

#             # surface at +y
#             u_node = get_node_vars(u, equations, dg, l, nnodes(dg), element)
#             # compute internal flux in normal direction on side 3
#             # outward_direction = get_node_coords(normal_directions, equations, dg, l, 3,
#             #                                     element)
#             outward_direction = get_surface_normal(normal_directions, l, 3, element)
#             f_node = splitting(u_node, Val{:minus}(), outward_direction, equations)
#             f_num = get_node_vars(surface_flux_values, equations, dg, l, 3, element)
#             multiply_add_to_node_vars!(du, inv_weight_right, (f_num - f_node),
#                                        equations, dg, l, nnodes(dg), element)
#         end
#     end

#     return nothing
# end

# AnalysisCallback
function integrate_via_indices(func::Func, u,
                               mesh::UnstructuredMesh2D, equations,
                               dg::FDSBP, cache, args...; normalize = true) where {Func}
    # TODO: FD. This is rather inefficient right now and allocates...
    weights = diag(SummationByPartsOperators.mass_matrix(dg.basis))

    # Initialize integral with zeros of the right shape
    integral = zero(func(u, 1, 1, 1, equations, dg, args...))
    total_volume = zero(real(mesh))

    # Use quadrature to numerically integrate over entire domain
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, element]))
            integral += volume_jacobian * weights[i] * weights[j] *
                        func(u, i, j, element, equations, dg, args...)
            total_volume += volume_jacobian * weights[i] * weights[j]
        end
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume
    end

    return integral
end

function calc_error_norms(func, u, t, analyzer,
                          mesh::UnstructuredMesh2D, equations, initial_condition,
                          dg::FDSBP, cache, cache_analysis)
    # TODO: FD. This is rather inefficient right now and allocates...
    weights = diag(SummationByPartsOperators.mass_matrix(dg.basis))
    @unpack node_coordinates, inverse_jacobian = cache.elements

    # Set up data structures
    l2_error = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
    linf_error = copy(l2_error)
    total_volume = zero(real(mesh))

    # Iterate over all elements for error calculations
    for element in eachelement(dg, cache)
        for j in eachnode(analyzer), i in eachnode(analyzer)
            volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, element]))
            u_exact = initial_condition(get_node_coords(node_coordinates, equations, dg,
                                                        i, j, element), t, equations)
            diff = func(u_exact, equations) -
                   func(get_node_vars(u, equations, dg, i, j, element), equations)
            l2_error += diff .^ 2 * (weights[i] * weights[j] * volume_jacobian)
            linf_error = @. max(linf_error, abs(diff))
            total_volume += weights[i] * weights[j] * volume_jacobian
        end
    end

    # For L2 error, divide by total volume
    l2_error = @. sqrt(l2_error / total_volume)

    return l2_error, linf_error
end
end # @muladd
