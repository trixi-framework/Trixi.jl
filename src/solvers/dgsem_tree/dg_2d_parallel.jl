# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# everything related to a DG semidiscretization in 2D using MPI,
# currently limited to Lobatto-Legendre nodes

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache_hyperbolic(mesh::TreeMeshParallel{2}, equations,
                                 dg::DG, RealT, ::Type{uEltype}) where {uEltype <: Real}
    # Get cells for which an element needs to be created (i.e. all leaf cells)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    mpi_interfaces = init_mpi_interfaces(leaf_cell_ids, mesh, elements)

    boundaries = init_boundaries(leaf_cell_ids, mesh, elements, dg.basis)

    mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    mpi_mortars = init_mpi_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    mpi_cache = init_mpi_cache(mesh, elements, mpi_interfaces, mpi_mortars,
                               nvariables(equations), nnodes(dg), uEltype)

    # Container cache
    cache = (; elements, interfaces, mpi_interfaces, boundaries, mortars,
             mpi_mortars, mpi_cache)

    # Add Volume-Integral cache
    cache = (; cache...,
             create_cache_all_volume_integrals(mesh, equations, dg.volume_integral,
                                               dg, cache, uEltype)...)
    # Add Mortar cache
    cache = (; cache..., create_cache_mortar(mesh, equations, dg.mortar, uEltype)...)

    return cache
end

# This function is valid for all MPI-parallelized non-conforming mesh types, i.e.,
# the 2D `TreeMesh`es as well as the 2D/3D `P4estMesh`es and `T8codeMesh`es.
function rhs_hyperbolic!(backend::Nothing,
                         du, u, t,
                         mesh::Union{TreeMeshParallel{2},
                                     P4estMeshParallel{2}, T8codeMeshParallel{2},
                                     P4estMeshParallel{3}, T8codeMeshParallel{3}},
                         equations,
                         boundary_conditions, source_terms::Source,
                         dg::DG, cache) where {Source}
    # Start to receive MPI data
    @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache.mpi_cache)

    # Prolong solution to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces" begin
        prolong2mpiinterfaces!(backend, cache, u, mesh, equations,
                               dg.surface_integral, dg)
    end

    # Prolong solution to MPI mortars
    @trixi_timeit timer() "prolong2mpimortars" begin
        prolong2mpimortars!(backend, cache, u, mesh, equations,
                            dg.mortar, dg)
    end

    # Start to send MPI data
    @trixi_timeit timer() "start MPI send" begin
        start_mpi_send!(backend, cache.mpi_cache, mesh, equations, dg, cache)
    end

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" set_zero!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(backend, du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(backend, cache, u, mesh, equations, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(backend, cache.elements.surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
                             dg.surface_integral, dg, cache)
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

    # Prolong solution to mortars
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u, mesh, equations,
                         dg.mortar, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                          have_nonconservative_terms(equations), equations,
                          dg.mortar, dg.surface_integral, dg, cache)
    end

    # Finish to receive MPI data
    @trixi_timeit timer() "finish MPI receive" begin
        finish_mpi_receive!(backend, cache.mpi_cache, mesh, equations, dg, cache)
    end

    # Calculate MPI interface fluxes
    @trixi_timeit timer() "MPI interface flux" begin
        calc_mpi_interface_flux!(backend, cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
    end

    # Calculate MPI mortar fluxes
    @trixi_timeit timer() "MPI mortar flux" begin
        calc_mpi_mortar_flux!(backend, cache.elements.surface_flux_values, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.mortar, dg.surface_integral, dg, cache)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(backend, du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" apply_jacobian!(backend, du, mesh, equations, dg,
                                                     cache)

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(backend, du, u, t, source_terms, equations, dg, cache)
    end

    # Finish to send MPI data
    @trixi_timeit timer() "finish MPI send" finish_mpi_send!(cache.mpi_cache)

    return nothing
end

function prolong2mpiinterfaces!(backend::Nothing, cache, u,
                                mesh::TreeMeshParallel{2},
                                equations, surface_integral, dg::DG)
    @unpack mpi_interfaces = cache

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(u, mesh, equations, dg, cache)
        check_axes(mpi_interfaces, equations, dg, cache)
    end

    @threaded for interface in eachmpiinterface(dg, cache)
        @inbounds begin
            local_element = mpi_interfaces.local_neighbor_ids[interface]

            if mpi_interfaces.orientations[interface] == 1 # interface in x-direction
                if mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                    for j in eachnode(dg), v in eachvariable(equations)
                        mpi_interfaces.u[2, v, j, interface] = u[v, 1, j, local_element]
                    end
                else # local element in negative direction
                    for j in eachnode(dg), v in eachvariable(equations)
                        mpi_interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j,
                                                                 local_element]
                    end
                end
            else # interface in y-direction
                if mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                    for i in eachnode(dg), v in eachvariable(equations)
                        mpi_interfaces.u[2, v, i, interface] = u[v, i, 1, local_element]
                    end
                else # local element in negative direction
                    for i in eachnode(dg), v in eachvariable(equations)
                        mpi_interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg),
                                                                 local_element]
                    end
                end
            end
        end
    end

    return nothing
end

function prolong2mpimortars!(backend::Nothing, cache, u,
                             mesh::TreeMeshParallel{2}, equations,
                             mortar_l2::LobattoLegendreMortarL2,
                             dg::DGSEM)
    @unpack mpi_mortars = cache
    @unpack u_upper, u_lower = mpi_mortars

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(u, mesh, equations, dg, cache)
        check_axes(mpi_mortars, equations, dg, cache)
    end

    @threaded for mortar in eachmpimortar(dg, cache)
        @inbounds begin
            local_neighbor_ids = mpi_mortars.local_neighbor_ids[mortar]
            local_neighbor_positions = mpi_mortars.local_neighbor_positions[mortar]

            for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
                if position in (1, 2) # Current element is small
                    # Copy solution small to small
                    if mpi_mortars.large_sides[mortar] == 1 # -> small elements on right side
                        if mpi_mortars.orientations[mortar] == 1
                            # L2 mortars in x-direction
                            if position == 1
                                for l in eachnode(dg)
                                    for v in eachvariable(equations)
                                        u_lower[2, v, l, mortar] = u[v, 1, l, element]
                                    end
                                end
                            else # position == 2
                                for l in eachnode(dg)
                                    for v in eachvariable(equations)
                                        u_upper[2, v, l, mortar] = u[v, 1, l, element]
                                    end
                                end
                            end
                        else
                            # L2 mortars in y-direction
                            if position == 1
                                for l in eachnode(dg)
                                    for v in eachvariable(equations)
                                        u_lower[2, v, l, mortar] = u[v, l, 1, element]
                                    end
                                end
                            else # position == 2
                                for l in eachnode(dg)
                                    for v in eachvariable(equations)
                                        u_upper[2, v, l, mortar] = u[v, l, 1, element]
                                    end
                                end
                            end
                        end
                    else # large_sides[mortar] == 2 -> small elements on left side
                        if mpi_mortars.orientations[mortar] == 1
                            # L2 mortars in x-direction
                            if position == 1
                                for l in eachnode(dg)
                                    for v in eachvariable(equations)
                                        u_lower[1, v, l, mortar] = u[v, nnodes(dg), l,
                                                                     element]
                                    end
                                end
                            else # position == 2
                                for l in eachnode(dg)
                                    for v in eachvariable(equations)
                                        u_upper[1, v, l, mortar] = u[v, nnodes(dg), l,
                                                                     element]
                                    end
                                end
                            end
                        else
                            # L2 mortars in y-direction
                            if position == 1
                                for l in eachnode(dg)
                                    for v in eachvariable(equations)
                                        u_lower[1, v, l, mortar] = u[v, l, nnodes(dg),
                                                                     element]
                                    end
                                end
                            else # position == 2
                                for l in eachnode(dg)
                                    for v in eachvariable(equations)
                                        u_upper[1, v, l, mortar] = u[v, l, nnodes(dg),
                                                                     element]
                                    end
                                end
                            end
                        end
                    end
                else # position == 3 -> current element is large
                    # Interpolate large element face data to small interface locations
                    if mpi_mortars.large_sides[mortar] == 1 # -> large element on left side
                        leftright = 1
                        if mpi_mortars.orientations[mortar] == 1
                            # L2 mortars in x-direction
                            u_large = view(u, :, nnodes(dg), :, element)
                            element_solutions_to_mortars!(mpi_mortars, mortar_l2,
                                                          leftright, mortar, u_large)
                        else
                            # L2 mortars in y-direction
                            u_large = view(u, :, :, nnodes(dg), element)
                            element_solutions_to_mortars!(mpi_mortars, mortar_l2,
                                                          leftright, mortar, u_large)
                        end
                    else # large_sides[mortar] == 2 -> large element on right side
                        leftright = 2
                        if mpi_mortars.orientations[mortar] == 1
                            # L2 mortars in x-direction
                            u_large = view(u, :, 1, :, element)
                            element_solutions_to_mortars!(mpi_mortars, mortar_l2,
                                                          leftright, mortar, u_large)
                        else
                            # L2 mortars in y-direction
                            u_large = view(u, :, :, 1, element)
                            element_solutions_to_mortars!(mpi_mortars, mortar_l2,
                                                          leftright, mortar, u_large)
                        end
                    end
                end
            end
        end
    end

    return nothing
end

function calc_mpi_interface_flux!(backend::Nothing, surface_flux_values,
                                  mesh::TreeMeshParallel{2},
                                  have_nonconservative_terms::False, equations,
                                  surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u, local_neighbor_ids, orientations, remote_sides = cache.mpi_interfaces

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(cache.mpi_interfaces, equations, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh, equations, dg, cache)
    end

    @threaded for interface in eachmpiinterface(dg, cache)
        @inbounds begin
            # Get local neighboring element
            element = local_neighbor_ids[interface]

            # Determine interface direction with respect to element:
            if orientations[interface] == 1 # interface in x-direction
                if remote_sides[interface] == 1 # local element in positive direction
                    direction = 1
                else # local element in negative direction
                    direction = 2
                end
            else # interface in y-direction
                if remote_sides[interface] == 1 # local element in positive direction
                    direction = 3
                else # local element in negative direction
                    direction = 4
                end
            end

            for i in eachnode(dg)
                # Call pointwise Riemann solver
                u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
                flux = surface_flux(u_ll, u_rr, orientations[interface], equations)

                # Copy flux to local element storage
                for v in eachvariable(equations)
                    surface_flux_values[v, i, direction, element] = flux[v]
                end
            end
        end
    end

    return nothing
end

function calc_mpi_mortar_flux!(backend::Nothing, surface_flux_values,
                               mesh::TreeMeshParallel{2},
                               have_nonconservative_terms::False, equations,
                               mortar_l2::LobattoLegendreMortarL2,
                               surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u_lower, u_upper, orientations = cache.mpi_mortars
    @unpack fstar_primary_upper_threaded, fstar_primary_lower_threaded, fstar_secondary_upper_threaded, fstar_secondary_lower_threaded = cache

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(cache.mpi_mortars, equations, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh, equations, dg, cache)
    end

    @threaded for mortar in eachmpimortar(dg, cache)
        @inbounds begin
            # Choose thread-specific pre-allocated container
            fstar_primary_upper = fstar_primary_upper_threaded[Threads.threadid()]
            fstar_primary_lower = fstar_primary_lower_threaded[Threads.threadid()]
            fstar_secondary_upper = fstar_secondary_upper_threaded[Threads.threadid()]
            fstar_secondary_lower = fstar_secondary_lower_threaded[Threads.threadid()]

            # Because `have_nonconservative_terms` is `False` the primary and secondary fluxes
            # are identical. So, we could possibly save on computation and just pass two copies later.
            orientation = orientations[mortar]
            calc_fstar!(fstar_primary_upper, equations, surface_flux, dg,
                        u_upper, mortar, orientation)
            calc_fstar!(fstar_primary_lower, equations, surface_flux, dg,
                        u_lower, mortar, orientation)
            calc_fstar!(fstar_secondary_upper, equations, surface_flux, dg,
                        u_upper, mortar, orientation)
            calc_fstar!(fstar_secondary_lower, equations, surface_flux, dg,
                        u_lower, mortar, orientation)

            mpi_mortar_fluxes_to_elements!(surface_flux_values,
                                           mesh, equations, mortar_l2, dg, cache,
                                           mortar,
                                           fstar_primary_upper, fstar_primary_lower,
                                           fstar_secondary_upper, fstar_secondary_lower)
        end
    end

    return nothing
end

Base.@propagate_inbounds function mpi_mortar_fluxes_to_elements!(surface_flux_values,
                                                                 mesh::TreeMeshParallel{2},
                                                                 equations,
                                                                 mortar_l2::LobattoLegendreMortarL2,
                                                                 dg::DGSEM, cache,
                                                                 mortar,
                                                                 fstar_primary_upper,
                                                                 fstar_primary_lower,
                                                                 fstar_secondary_upper,
                                                                 fstar_secondary_lower)
    local_neighbor_ids = cache.mpi_mortars.local_neighbor_ids[mortar]
    local_neighbor_positions = cache.mpi_mortars.local_neighbor_positions[mortar]

    for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
        if position in (1, 2) # Current element is small
            # Copy flux small to small
            if cache.mpi_mortars.large_sides[mortar] == 1 # -> small elements on right side
                if cache.mpi_mortars.orientations[mortar] == 1
                    # L2 mortars in x-direction
                    direction = 1
                else
                    # L2 mortars in y-direction
                    direction = 3
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if cache.mpi_mortars.orientations[mortar] == 1
                    # L2 mortars in x-direction
                    direction = 2
                else
                    # L2 mortars in y-direction
                    direction = 4
                end
            end

            if position == 1
                surface_flux_values[:, :, direction, element] .= fstar_primary_lower
            elseif position == 2
                surface_flux_values[:, :, direction, element] .= fstar_primary_upper
            end
        else # position == 3 -> current element is large
            # Project small fluxes to large element
            if cache.mpi_mortars.large_sides[mortar] == 1 # -> large element on left side
                if cache.mpi_mortars.orientations[mortar] == 1
                    # L2 mortars in x-direction
                    direction = 2
                else
                    # L2 mortars in y-direction
                    direction = 4
                end
            else # large_sides[mortar] == 2 -> large element on right side
                if cache.mpi_mortars.orientations[mortar] == 1
                    # L2 mortars in x-direction
                    direction = 1
                else
                    # L2 mortars in y-direction
                    direction = 3
                end
            end

            multiply_dimensionwise!(view(surface_flux_values, :, :, direction, element),
                                    mortar_l2.reverse_upper, fstar_secondary_upper,
                                    mortar_l2.reverse_lower, fstar_secondary_lower)
        end
    end

    return nothing
end
end # @muladd
