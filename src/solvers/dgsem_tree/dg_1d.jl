# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# everything related to a DG semidiscretization in 1D,
# currently limited to Lobatto-Legendre nodes

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::TreeMesh{1}, equations,
                      dg::DG, RealT, uEltype)
    # Get cells for which an element needs to be created (i.e. all leaf cells)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    cache = (; elements, interfaces, boundaries)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache...,
             create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)

    return cache
end

# The methods below are specialized on the volume integral type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::Union{TreeMesh{1}, StructuredMesh{1}, P4estMesh{1}},
                      equations,
                      volume_integral::VolumeIntegralFluxDifferencing, dg::DG, uEltype)
    NamedTuple()
end

function create_cache(mesh::Union{TreeMesh{1}, StructuredMesh{1}, P4estMesh{1}},
                      equations,
                      volume_integral::VolumeIntegralShockCapturingHG, dg::DG, uEltype)
    element_ids_dg = Int[]
    element_ids_dgfv = Int[]

    cache = create_cache(mesh, equations,
                         VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                         dg, uEltype)

    A2dp1_x = Array{uEltype, 2}
    fstar1_L_threaded = A2dp1_x[A2dp1_x(undef, nvariables(equations), nnodes(dg) + 1)
                                for _ in 1:Threads.nthreads()]
    fstar1_R_threaded = A2dp1_x[A2dp1_x(undef, nvariables(equations), nnodes(dg) + 1)
                                for _ in 1:Threads.nthreads()]

    return (; cache..., element_ids_dg, element_ids_dgfv, fstar1_L_threaded,
            fstar1_R_threaded)
end

function create_cache(mesh::Union{TreeMesh{1}, StructuredMesh{1}, P4estMesh{1}},
                      equations,
                      volume_integral::VolumeIntegralPureLGLFiniteVolume, dg::DG,
                      uEltype)
    A2dp1_x = Array{uEltype, 2}
    fstar1_L_threaded = A2dp1_x[A2dp1_x(undef, nvariables(equations), nnodes(dg) + 1)
                                for _ in 1:Threads.nthreads()]
    fstar1_R_threaded = A2dp1_x[A2dp1_x(undef, nvariables(equations), nnodes(dg) + 1)
                                for _ in 1:Threads.nthreads()]

    return (; fstar1_L_threaded, fstar1_R_threaded)
end

# TODO: Taal discuss/refactor timer, allowing users to pass a custom timer?

function rhs!(du, u, t,
              mesh::TreeMesh{1}, equations,
              boundary_conditions, source_terms::Source,
              dg::DG, cache) where {Source}
    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, u, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
                             dg.surface_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, u, mesh, equations,
                            dg.surface_integral, dg)
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

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, equations, dg, cache)
    end

    return nothing
end

function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
    @threaded for element in eachelement(dg, cache)
        weak_form_kernel!(du, u, element, mesh,
                          nonconservative_terms, equations,
                          dg, cache)
    end

    return nothing
end

#=
`weak_form_kernel!` is only implemented for conserved terms as
non-conservative terms should always be discretized in conjunction with a flux-splitting scheme,
see `flux_differencing_kernel!`.
This treatment is required to achieve, e.g., entropy-stability or well-balancedness.
See also https://github.com/trixi-framework/Trixi.jl/issues/1671#issuecomment-1765644064
=#
@inline function weak_form_kernel!(du, u,
                                   element, mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                   nonconservative_terms::False, equations,
                                   dg::DGSEM, cache, alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    @unpack derivative_dhat = dg.basis

    for i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, element)

        flux1 = flux(u_node, 1, equations)
        for ii in eachnode(dg)
            multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], flux1,
                                       equations, dg, ii, element)
        end
    end

    return nothing
end

function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGSEM, cache)
    @threaded for element in eachelement(dg, cache)
        flux_differencing_kernel!(du, u, element, mesh, nonconservative_terms,
                                  equations,
                                  volume_integral.volume_flux, dg, cache)
    end
end

@inline function flux_differencing_kernel!(du, u,
                                           element,
                                           mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                           nonconservative_terms::False, equations,
                                           volume_flux, dg::DGSEM, cache, alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    @unpack derivative_split = dg.basis

    # Calculate volume integral in one element
    for i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, element)
            flux1 = volume_flux(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(du, alpha * derivative_split[i, ii], flux1,
                                       equations, dg, i, element)
            multiply_add_to_node_vars!(du, alpha * derivative_split[ii, i], flux1,
                                       equations, dg, ii, element)
        end
    end
end

@inline function flux_differencing_kernel!(du, u,
                                           element,
                                           mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                           nonconservative_terms::True, equations,
                                           volume_flux, dg::DGSEM, cache, alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    @unpack derivative_split = dg.basis
    symmetric_flux, nonconservative_flux = volume_flux

    # Apply the symmetric flux as usual
    flux_differencing_kernel!(du, u, element, mesh, False(), equations, symmetric_flux,
                              dg, cache, alpha)

    # Calculate the remaining volume terms using the nonsymmetric generalized flux
    for i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, element)

        # The diagonal terms are zero since the diagonal of `derivative_split`
        # is zero. We ignore this for now.

        # x direction
        integral_contribution = zero(u_node)
        for ii in eachnode(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, element)
            noncons_flux1 = nonconservative_flux(u_node, u_node_ii, 1, equations)
            integral_contribution = integral_contribution +
                                    derivative_split[i, ii] * noncons_flux1
        end

        # The factor 0.5 cancels the factor 2 in the flux differencing form
        multiply_add_to_node_vars!(du, alpha * 0.5f0, integral_contribution, equations,
                                   dg, i, element)
    end
end

# TODO: Taal dimension agnostic
function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHG,
                               dg::DGSEM, cache)
    @unpack element_ids_dg, element_ids_dgfv = cache
    @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral

    # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
    alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg,
                                                               cache)

    # Determine element ids for DG-only and blended DG-FV volume integral
    pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)

    # Loop over pure DG elements
    @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
        element = element_ids_dg[idx_element]
        flux_differencing_kernel!(du, u, element, mesh, nonconservative_terms,
                                  equations,
                                  volume_flux_dg, dg, cache)
    end

    # Loop over blended DG-FV elements
    @trixi_timeit timer() "blended DG-FV" @threaded for idx_element in eachindex(element_ids_dgfv)
        element = element_ids_dgfv[idx_element]
        alpha_element = alpha[element]

        # Calculate DG volume integral contribution
        flux_differencing_kernel!(du, u, element, mesh, nonconservative_terms,
                                  equations,
                                  volume_flux_dg, dg, cache, 1 - alpha_element)

        # Calculate FV volume integral contribution
        fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
                   dg, cache, element, alpha_element)
    end

    return nothing
end

# TODO: Taal dimension agnostic
function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralPureLGLFiniteVolume,
                               dg::DGSEM, cache)
    @unpack volume_flux_fv = volume_integral

    # Calculate LGL FV volume integral
    @threaded for element in eachelement(dg, cache)
        fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
                   dg, cache, element, true)
    end

    return nothing
end

@inline function fv_kernel!(du, u,
                            mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                            nonconservative_terms, equations,
                            volume_flux_fv, dg::DGSEM, cache, element, alpha = true)
    @unpack fstar1_L_threaded, fstar1_R_threaded = cache
    @unpack inverse_weights = dg.basis

    # Calculate FV two-point fluxes
    fstar1_L = fstar1_L_threaded[Threads.threadid()]
    fstar1_R = fstar1_R_threaded[Threads.threadid()]
    calcflux_fv!(fstar1_L, fstar1_R, u, mesh, nonconservative_terms, equations,
                 volume_flux_fv,
                 dg, element, cache)

    # Calculate FV volume integral contribution
    for i in eachnode(dg)
        for v in eachvariable(equations)
            du[v, i, element] += (alpha *
                                  (inverse_weights[i] *
                                   (fstar1_L[v, i + 1] - fstar1_R[v, i])))
        end
    end

    return nothing
end

@inline function calcflux_fv!(fstar1_L, fstar1_R, u::AbstractArray{<:Any, 3},
                              mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                              nonconservative_terms::False,
                              equations, volume_flux_fv, dg::DGSEM, element, cache)
    fstar1_L[:, 1] .= zero(eltype(fstar1_L))
    fstar1_L[:, nnodes(dg) + 1] .= zero(eltype(fstar1_L))
    fstar1_R[:, 1] .= zero(eltype(fstar1_R))
    fstar1_R[:, nnodes(dg) + 1] .= zero(eltype(fstar1_R))

    for i in 2:nnodes(dg)
        u_ll = get_node_vars(u, equations, dg, i - 1, element)
        u_rr = get_node_vars(u, equations, dg, i, element)
        flux = volume_flux_fv(u_ll, u_rr, 1, equations) # orientation 1: x direction
        set_node_vars!(fstar1_L, flux, equations, dg, i)
        set_node_vars!(fstar1_R, flux, equations, dg, i)
    end

    return nothing
end

@inline function calcflux_fv!(fstar1_L, fstar1_R, u::AbstractArray{<:Any, 3},
                              mesh::TreeMesh{1},
                              nonconservative_terms::True,
                              equations, volume_flux_fv, dg::DGSEM, element, cache)
    volume_flux, nonconservative_flux = volume_flux_fv

    fstar1_L[:, 1] .= zero(eltype(fstar1_L))
    fstar1_L[:, nnodes(dg) + 1] .= zero(eltype(fstar1_L))
    fstar1_R[:, 1] .= zero(eltype(fstar1_R))
    fstar1_R[:, nnodes(dg) + 1] .= zero(eltype(fstar1_R))

    for i in 2:nnodes(dg)
        u_ll = get_node_vars(u, equations, dg, i - 1, element)
        u_rr = get_node_vars(u, equations, dg, i, element)

        # Compute conservative part
        f1 = volume_flux(u_ll, u_rr, 1, equations) # orientation 1: x direction

        # Compute nonconservative part
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        f1_L = f1 + 0.5f0 * nonconservative_flux(u_ll, u_rr, 1, equations)
        f1_R = f1 + 0.5f0 * nonconservative_flux(u_rr, u_ll, 1, equations)

        # Copy to temporary storage
        set_node_vars!(fstar1_L, f1_L, equations, dg, i)
        set_node_vars!(fstar1_R, f1_R, equations, dg, i)
    end

    return nothing
end

# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache, u,
                             mesh::TreeMesh{1}, equations, surface_integral, dg::DG)
    @unpack interfaces = cache
    @unpack neighbor_ids = interfaces
    interfaces_u = interfaces.u

    @threaded for interface in eachinterface(dg, cache)
        left_element = neighbor_ids[1, interface]
        right_element = neighbor_ids[2, interface]

        # interface in x-direction
        for v in eachvariable(equations)
            interfaces_u[1, v, interface] = u[v, nnodes(dg), left_element]
            interfaces_u[2, v, interface] = u[v, 1, right_element]
        end
    end

    return nothing
end

function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{1},
                              nonconservative_terms::False, equations,
                              surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u, neighbor_ids, orientations = cache.interfaces

    @threaded for interface in eachinterface(dg, cache)
        # Get neighboring elements
        left_id = neighbor_ids[1, interface]
        right_id = neighbor_ids[2, interface]

        # Determine interface direction with respect to elements:
        # orientation = 1: left -> 2, right -> 1
        left_direction = 2 * orientations[interface]
        right_direction = 2 * orientations[interface] - 1

        # Call pointwise Riemann solver
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, interface)
        flux = surface_flux(u_ll, u_rr, orientations[interface], equations)

        # Copy flux to left and right element storage
        for v in eachvariable(equations)
            surface_flux_values[v, left_direction, left_id] = flux[v]
            surface_flux_values[v, right_direction, right_id] = flux[v]
        end
    end
end

function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{1},
                              nonconservative_terms::True, equations,
                              surface_integral, dg::DG, cache)
    surface_flux, nonconservative_flux = surface_integral.surface_flux
    @unpack u, neighbor_ids, orientations = cache.interfaces

    @threaded for interface in eachinterface(dg, cache)
        # Get neighboring elements
        left_id = neighbor_ids[1, interface]
        right_id = neighbor_ids[2, interface]

        # Determine interface direction with respect to elements:
        # orientation = 1: left -> 2, right -> 1
        # orientation = 2: left -> 4, right -> 3
        left_direction = 2 * orientations[interface]
        right_direction = 2 * orientations[interface] - 1

        # Call pointwise Riemann solver
        orientation = orientations[interface]
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, interface)
        flux = surface_flux(u_ll, u_rr, orientation, equations)

        # Compute both nonconservative fluxes
        noncons_left = nonconservative_flux(u_ll, u_rr, orientation, equations)
        noncons_right = nonconservative_flux(u_rr, u_ll, orientation, equations)

        # Copy flux to left and right element storage
        for v in eachvariable(equations)
            # Note the factor 0.5 necessary for the nonconservative fluxes based on
            # the interpretation of global SBP operators coupled discontinuously via
            # central fluxes/SATs
            surface_flux_values[v, left_direction, left_id] = flux[v] +
                                                              0.5f0 * noncons_left[v]
            surface_flux_values[v, right_direction, right_id] = flux[v] +
                                                                0.5f0 * noncons_right[v]
        end
    end

    return nothing
end

function prolong2boundaries!(cache, u,
                             mesh::TreeMesh{1}, equations, surface_integral, dg::DG)
    @unpack boundaries = cache
    @unpack neighbor_sides = boundaries

    @threaded for boundary in eachboundary(dg, cache)
        element = boundaries.neighbor_ids[boundary]

        # boundary in x-direction
        if neighbor_sides[boundary] == 1
            # element in -x direction of boundary
            for v in eachvariable(equations)
                boundaries.u[1, v, boundary] = u[v, nnodes(dg), element]
            end
        else # Element in +x direction of boundary
            for v in eachvariable(equations)
                boundaries.u[2, v, boundary] = u[v, 1, element]
            end
        end
    end

    return nothing
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::TreeMesh{1}, equations, surface_integral, dg::DG)
    @assert isempty(eachboundary(dg, cache))
end

function calc_boundary_flux!(cache, t, boundary_conditions::NamedTuple,
                             mesh::TreeMesh{1}, equations, surface_integral, dg::DG)
    @unpack surface_flux_values = cache.elements
    @unpack n_boundaries_per_direction = cache.boundaries

    # Calculate indices
    lasts = accumulate(+, n_boundaries_per_direction)
    firsts = lasts - n_boundaries_per_direction .+ 1

    # Calc boundary fluxes in each direction
    calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[1],
                                     have_nonconservative_terms(equations), equations,
                                     surface_integral, dg, cache,
                                     1, firsts[1], lasts[1])
    calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[2],
                                     have_nonconservative_terms(equations), equations,
                                     surface_integral, dg, cache,
                                     2, firsts[2], lasts[2])
end

function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any, 3},
                                          t,
                                          boundary_condition,
                                          nonconservative_terms::False, equations,
                                          surface_integral, dg::DG, cache,
                                          direction, first_boundary, last_boundary)
    @unpack surface_flux = surface_integral
    @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

    @threaded for boundary in first_boundary:last_boundary
        # Get neighboring element
        neighbor = neighbor_ids[boundary]

        # Get boundary flux
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, boundary)
        if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
            u_inner = u_ll
        else # Element is on the right, boundary on the left
            u_inner = u_rr
        end
        x = get_node_coords(node_coordinates, equations, dg, boundary)
        flux = boundary_condition(u_inner, orientations[boundary], direction, x, t,
                                  surface_flux,
                                  equations)

        # Copy flux to left and right element storage
        for v in eachvariable(equations)
            surface_flux_values[v, direction, neighbor] = flux[v]
        end
    end

    return nothing
end

function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any, 3},
                                          t,
                                          boundary_condition,
                                          nonconservative_terms::True, equations,
                                          surface_integral, dg::DG, cache,
                                          direction, first_boundary, last_boundary)
    surface_flux, nonconservative_flux = surface_integral.surface_flux
    @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

    @threaded for boundary in first_boundary:last_boundary
        # Get neighboring element
        neighbor = neighbor_ids[boundary]

        # Get boundary flux
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, boundary)
        if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
            u_inner = u_ll
        else # Element is on the right, boundary on the left
            u_inner = u_rr
        end
        x = get_node_coords(node_coordinates, equations, dg, boundary)
        flux = boundary_condition(u_inner, orientations[boundary], direction, x, t,
                                  surface_flux,
                                  equations)
        noncons_flux = boundary_condition(u_inner, orientations[boundary], direction, x,
                                          t, nonconservative_flux,
                                          equations)

        # Copy flux to left and right element storage
        for v in eachvariable(equations)
            surface_flux_values[v, direction, neighbor] = flux[v] +
                                                          0.5f0 * noncons_flux[v]
        end
    end

    return nothing
end

function calc_surface_integral!(du, u, mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                equations, surface_integral, dg::DGSEM, cache)
    @unpack boundary_interpolation = dg.basis
    @unpack surface_flux_values = cache.elements

    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # Access the factors only once before beginning the loop to increase performance.
    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    factor_1 = boundary_interpolation[1, 1]
    factor_2 = boundary_interpolation[nnodes(dg), 2]
    @threaded for element in eachelement(dg, cache)
        for v in eachvariable(equations)
            # surface at -x
            du[v, 1, element] = (du[v, 1, element] -
                                 surface_flux_values[v, 1, element] * factor_1)

            # surface at +x
            du[v, nnodes(dg), element] = (du[v, nnodes(dg), element] +
                                          surface_flux_values[v, 2, element] * factor_2)
        end
    end

    return nothing
end

function apply_jacobian!(du, mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                         equations, dg::DG, cache)
    @unpack inverse_jacobian = cache.elements

    @threaded for element in eachelement(dg, cache)
        factor = -inverse_jacobian[element]

        for i in eachnode(dg)
            for v in eachvariable(equations)
                du[v, i, element] *= factor
            end
        end
    end

    return nothing
end

# TODO: Taal dimension agnostic
function calc_sources!(du, u, t, source_terms::Nothing,
                       equations::AbstractEquations{1}, dg::DG, cache)
    return nothing
end

function calc_sources!(du, u, t, source_terms,
                       equations::AbstractEquations{1}, dg::DG, cache)
    @unpack node_coordinates = cache.elements

    @threaded for element in eachelement(dg, cache)
        for i in eachnode(dg)
            u_local = get_node_vars(u, equations, dg, i, element)
            x_local = get_node_coords(node_coordinates, equations, dg,
                                      i, element)
            du_local = source_terms(u_local, x_local, t, equations)
            add_to_node_vars!(du, du_local, equations, dg, i, element)
        end
    end

    return nothing
end
end # @muladd
