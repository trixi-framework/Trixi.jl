# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# BlockFV on a `P4estMesh` splits each `p4est` element into an
# `n_nodes × n_nodes` grid of FV cells. We get the physical corners of these FV
# cells by evaluating the mesh mapping - the same curved geometry DGSEM uses - at
# equidistant points in reference space (see `calc_fv_corner_coordinates`/
# `init_elements!` below).
#
# Past that, we don't use the curved geometry any further: each FV cell
# is just a straight-sided quadrilateral spanned by its 4 corners, with a constant
# volume and scaled face normals. So unlike DGSEM on
# `P4estMesh` (which keeps the full curved geometry via `contravariant_vectors`),
# curvature here is only resolved as well as this polygonal approximation allows.

# BlockFV does not need a separate `normal_directions` array in the interface
# container: face normals are looked up directly from `cache.normal_x`/`normal_y`
# (computed once in `create_cache`)

@inline trivial_interface_normals(::UniformFiniteVolumeBasis) = true

# Face normals are computed once in `create_cache` (`normal_x`, `normal_y`)

function init_normal_directions!(interfaces::P4estInterfaceContainer{2},
                                 basis::UniformFiniteVolumeBasis, elements)
    return nothing
end

#####################################################################

# Physical coordinates of the (n_nodes+1)×(n_nodes+1) cell corners per element,
# obtained by mapping the equidistant boundary nodes in [-1,1] through the mesh geometry.

function calc_fv_corner_coordinates(mesh::P4estMesh{2}, basis::UniformFiniteVolumeBasis)
    n_nodes = nnodes(basis)
    boundary_nodes = SVector{n_nodes + 1}(range(-1, 1, length = n_nodes + 1))
    corners = Array{real(mesh)}(undef, 2, n_nodes + 1, n_nodes + 1, ncells(mesh))
    calc_node_coordinates!(corners, mesh, boundary_nodes)
    return corners
end

# In `init_elements!` below, `inverse_jacobian` is set to the exact
# reciprocal of each FV cell's physical area, computed from its 4 corners.

function init_elements!(elements, mesh::P4estMesh{2}, basis::UniformFiniteVolumeBasis)
    @unpack node_coordinates, inverse_jacobian = elements
    calc_node_coordinates!(node_coordinates, mesh, basis.nodes)

    corners = calc_fv_corner_coordinates(mesh, basis)
    n_nodes = nnodes(basis)
    for element in 1:ncells(mesh)
        for j in 1:n_nodes, i in 1:n_nodes
            x1, y1 = corners[1, i, j, element], corners[2, i, j, element]
            x2, y2 = corners[1, i + 1, j, element], corners[2, i + 1, j, element]
            x3, y3 = corners[1, i + 1, j + 1, element],
                     corners[2, i + 1, j + 1, element]
            x4, y4 = corners[1, i, j + 1, element], corners[2, i, j + 1, element]
            # Area of a quadrilateral from its diagonals
            # d1 = (x3,y3) - (x1,y1) and d2 = (x4,y4) - (x2,y2):
            # area = (1/2) * (d1 x d2)
            # See e.g. https://en.wikipedia.org/wiki/Quadrilateral#Vector_formulas
            volume = 0.5f0 * ((x3 - x1) * (y4 - y2) - (x4 - x2) * (y3 - y1))
            inverse_jacobian[i, j, element] = inv(volume)
        end
    end

    return nothing
end

# BlockFV uses its basis as `dg.mortar` (see blockfv.jl constructor).
# The shared rhs_hyperbolic! (TreeMesh/P4estMesh) calls create_cache, prolong2mortars!, and
# calc_mortar_flux! with dg.mortar. BlockFV on P4estMesh needs none of these,
# since mortars are not yet supported.
function create_cache(mesh::P4estMesh{2}, ::Any, ::UniformFiniteVolumeBasis, ::Any)
    if count_required_surfaces(mesh).mortars > 0
        throw(ArgumentError("BlockFV on P4estMesh does not yet support non-conforming " *
                            "meshes with hanging nodes (mortars)"))
    end
    return NamedTuple()
end

prolong2mortars!(_, _, _, ::P4estMesh{2}, _, ::UniformFiniteVolumeBasis, ::BlockFV) = nothing
calc_mortar_flux!(_, _, ::P4estMesh{2}, _, _, ::UniformFiniteVolumeBasis, _, ::BlockFV, _) = nothing

# create_cache computes and stores for every FV cell face the
# scaled face-normal vector (`normal_x`/`normal_y`, scaled by the interface length)
# and the midpoint of the face (`midpoint_x`/`midpoint_y`)

function create_cache(mesh::P4estMesh{2}, equations,
                      volume_integral::VolumeIntegralFiniteVolume,
                      dg::BlockFV, cache_containers, uEltype)
    n_nodes = nnodes(dg)
    nv = nvariables(equations)

    MA_x = MArray{Tuple{nv, n_nodes + 1, n_nodes}, uEltype, 3,
                  nv * (n_nodes + 1) * n_nodes}
    fstar_x_threaded = [MA_x(undef) for _ in 1:Threads.maxthreadid()]

    MA_y = MArray{Tuple{nv, n_nodes, n_nodes + 1}, uEltype, 3,
                  nv * n_nodes * (n_nodes + 1)}
    fstar_y_threaded = [MA_y(undef) for _ in 1:Threads.maxthreadid()]

    for (fx, fy) in zip(fstar_x_threaded, fstar_y_threaded)
        fx[:, 1, :] .= zero(uEltype)
        fx[:, n_nodes + 1, :] .= zero(uEltype)
        fy[:, :, 1] .= zero(uEltype)
        fy[:, :, n_nodes + 1] .= zero(uEltype)
    end

    RealT = real(mesh)
    n_elements = nelements(dg, cache_containers)
    corners = calc_fv_corner_coordinates(mesh, dg.basis)

    normal_x = Array{RealT}(undef, 2, n_nodes + 1, n_nodes, n_elements)
    normal_y = Array{RealT}(undef, 2, n_nodes, n_nodes + 1, n_elements)
    midpoint_x = Array{RealT}(undef, 2, n_nodes + 1, n_nodes, n_elements)
    midpoint_y = Array{RealT}(undef, 2, n_nodes, n_nodes + 1, n_elements)
    for element in 1:n_elements
        for j in 1:n_nodes, i in 1:(n_nodes + 1)
            normal_x[1, i, j, element] = corners[2, i, j + 1, element] -
                                         corners[2, i, j, element]
            normal_x[2, i, j, element] = corners[1, i, j, element] -
                                         corners[1, i, j + 1, element]
            midpoint_x[1, i, j, element] = 0.5f0 * (corners[1, i, j, element] +
                                            corners[1, i, j + 1, element])
            midpoint_x[2, i, j, element] = 0.5f0 * (corners[2, i, j, element] +
                                            corners[2, i, j + 1, element])
        end
        for j in 1:(n_nodes + 1), i in 1:n_nodes
            normal_y[1, i, j, element] = corners[2, i, j, element] -
                                         corners[2, i + 1, j, element]
            normal_y[2, i, j, element] = corners[1, i + 1, j, element] -
                                         corners[1, i, j, element]
            midpoint_y[1, i, j, element] = 0.5f0 * (corners[1, i, j, element] +
                                            corners[1, i + 1, j, element])
            midpoint_y[2, i, j, element] = 0.5f0 * (corners[2, i, j, element] +
                                            corners[2, i + 1, j, element])
        end
    end

    return (; fstar_x_threaded, fstar_y_threaded, normal_x, normal_y, midpoint_x,
            midpoint_y)
end

#####################################################################

function prolong2interfaces!(backend::Nothing, cache, u,
                             mesh::P4estMesh{2},
                             equations, dg::BlockFV)
    @unpack interfaces = cache
    @unpack neighbor_ids, node_indices = cache.interfaces
    index_range = eachnode(dg)
    MeshT = typeof(mesh)

    @threaded for interface in eachinterface(dg, cache)
        prolong2interfaces_per_interface!(interfaces.u, u, interface,
                                          MeshT, equations,
                                          neighbor_ids, node_indices, index_range)
    end
    return nothing
end

@inline function get_fv_boundary_normal(direction, normal_x, normal_y, i, j, element,
                                        n_nodes)
    if direction == 1 # -x
        return SVector(-normal_x[1, 1, j, element], -normal_x[2, 1, j, element])
    elseif direction == 2 # +x
        return SVector(normal_x[1, n_nodes + 1, j, element],
                       normal_x[2, n_nodes + 1, j, element])
    elseif direction == 3 # -y
        return SVector(-normal_y[1, i, 1, element], -normal_y[2, i, 1, element])
    else # direction == 4, +y
        return SVector(normal_y[1, i, n_nodes + 1, element],
                       normal_y[2, i, n_nodes + 1, element])
    end
end

# Exact physical midpoint of an FV cell's face at an element boundary, taken
# from `midpoint_x`/`midpoint_y` computed in `create_cache`.
@inline function get_fv_boundary_midpoint(direction, midpoint_x, midpoint_y, i, j,
                                          element, n_nodes)
    if direction == 1 # -x
        return SVector(midpoint_x[1, 1, j, element], midpoint_x[2, 1, j, element])
    elseif direction == 2 # +x
        return SVector(midpoint_x[1, n_nodes + 1, j, element],
                       midpoint_x[2, n_nodes + 1, j, element])
    elseif direction == 3 # -y
        return SVector(midpoint_y[1, i, 1, element], midpoint_y[2, i, 1, element])
    else # direction == 4, +y
        return SVector(midpoint_y[1, i, n_nodes + 1, element],
                       midpoint_y[2, i, n_nodes + 1, element])
    end
end

#####################################################################

function calc_interface_flux!(backend::Nothing, surface_flux_values,
                              mesh::P4estMesh{2},
                              have_nonconservative_terms::False,
                              equations, surface_integral,
                              dg::BlockFV, cache)
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack normal_x, normal_y = cache
    index_range = eachnode(dg)
    index_end = last(index_range)
    n_nodes = nnodes(dg)
    MeshT = typeof(mesh)
    SolverT = typeof(dg)

    @threaded for interface in eachinterface(dg, cache)
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]
        primary_direction = indices2direction(primary_indices)

        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)

        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]
        secondary_direction = indices2direction(secondary_indices)

        # This index on the primary side will always run forward but
        # the secondary index might need to run backwards for flipped sides.
        if :i_backward in secondary_indices
            node_secondary = index_end
            node_secondary_step = -1
        else
            node_secondary = 1
            node_secondary_step = 1
        end

        i_primary = i_primary_start
        j_primary = j_primary_start
        for node in index_range
            normal_direction = get_fv_boundary_normal(primary_direction,
                                                      normal_x, normal_y,
                                                      i_primary, j_primary,
                                                      primary_element, n_nodes)

            calc_interface_flux!(surface_flux_values, MeshT, have_nonconservative_terms,
                                 equations, surface_integral, SolverT,
                                 cache.interfaces.u, interface,
                                 normal_direction, node,
                                 primary_direction, primary_element,
                                 node_secondary,
                                 secondary_direction, secondary_element)

            i_primary += i_primary_step
            j_primary += j_primary_step
            node_secondary += node_secondary_step
        end
    end

    return nothing
end

#####################################################################

@inline function calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                                     mesh::P4estMesh{2},
                                     have_nonconservative_terms::False, equations,
                                     surface_integral, dg::BlockFV, cache,
                                     i_index, j_index,
                                     node_index, direction_index, element_index,
                                     boundary_index)
    @unpack boundaries = cache
    @unpack normal_x, normal_y, midpoint_x, midpoint_y = cache
    @unpack surface_flux = surface_integral

    u_inner = get_node_vars(boundaries.u, equations, dg, node_index, boundary_index)

    n_nodes = nnodes(dg)
    normal_direction = get_fv_boundary_normal(direction_index, normal_x, normal_y,
                                              i_index, j_index, element_index, n_nodes)

    x = get_fv_boundary_midpoint(direction_index, midpoint_x, midpoint_y,
                                 i_index, j_index, element_index, n_nodes)

    flux_ = boundary_condition(u_inner, normal_direction, x, t, surface_flux, equations)

    for v in eachvariable(equations)
        surface_flux_values[v, node_index, direction_index, element_index] = flux_[v]
    end

    return nothing
end

#####################################################################

function calc_volume_integral!(backend::Nothing, du, u,
                               mesh::P4estMesh{2},
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFiniteVolume,
                               dg::BlockFV, cache)
    @unpack surface_flux = volume_integral
    @unpack fstar_x_threaded, fstar_y_threaded, normal_x, normal_y = cache

    @threaded for element in eachelement(dg, cache)
        fstar_x = fstar_x_threaded[Threads.threadid()]
        fstar_y = fstar_y_threaded[Threads.threadid()]

        # x-direction: internal interfaces at i = 2, ..., n_nodes (between cells i-1, i).
        # `normal` is the scaled face-normal vector (scaled by the interface length).
        for j in eachnode(dg)
            for i in 2:nnodes(dg)
                u_ll = get_node_vars(u, equations, dg, i - 1, j, element)
                u_rr = get_node_vars(u, equations, dg, i, j, element)
                normal = SVector(normal_x[1, i, j, element], normal_x[2, i, j, element])
                f = surface_flux(u_ll, u_rr, normal, equations)
                set_node_vars!(fstar_x, f, equations, dg, i, j)
            end
        end

        # y-direction: internal interfaces at j = 2, ..., n_nodes (between cells j-1, j).
        # `normal` is the scaled face-normal vector (scaled by the interface length).
        for j in 2:nnodes(dg)
            for i in eachnode(dg)
                u_ll = get_node_vars(u, equations, dg, i, j - 1, element)
                u_rr = get_node_vars(u, equations, dg, i, j, element)
                normal = SVector(normal_y[1, i, j, element], normal_y[2, i, j, element])
                f = surface_flux(u_ll, u_rr, normal, equations)
                set_node_vars!(fstar_y, f, equations, dg, i, j)
            end
        end

        # Apply flux differences
        for j in eachnode(dg)
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    # We require `du` to be set to zero before this operation.
                    # The numerical fluxes are computed using scaled normal
                    # directions (scaled by the interface length) and the
                    # division by the cell volume (Jacobian) is done later
                    # when evaluating the semidiscretization (`rhs_hyperbolic!`).
                    du[v, i, j, element] = (du[v, i, j, element] +
                                            (fstar_x[v, i + 1, j] - fstar_x[v, i, j]) +
                                            (fstar_y[v, i, j + 1] - fstar_y[v, i, j]))
                end
            end
        end
    end

    return nothing
end

#####################################################################

function calc_surface_integral!(backend::Nothing, du, u,
                                mesh::P4estMesh{2},
                                equations, surface_integral::SurfaceIntegralWeakForm,
                                dg::BlockFV, cache)
    @unpack surface_flux_values = cache.elements

    @threaded for element in eachelement(dg, cache)
        for l in eachnode(dg)
            for v in eachvariable(equations)
                # surface at -x
                du[v, 1, l, element] = du[v, 1, l, element] +
                                       surface_flux_values[v, l, 1, element]
                # surface at +x
                du[v, nnodes(dg), l, element] = du[v, nnodes(dg), l, element] +
                                                surface_flux_values[v, l, 2, element]
                # surface at -y
                du[v, l, 1, element] = du[v, l, 1, element] +
                                       surface_flux_values[v, l, 3, element]
                # surface at +y
                du[v, l, nnodes(dg), element] = du[v, l, nnodes(dg), element] +
                                                surface_flux_values[v, l, 4, element]
            end
        end
    end

    return nothing
end

#####################################################################
# max_dt: the generic version (src/callbacks_step/stepsize_dg2d.jl) reads
# `contravariant_vectors`, which BlockFV does not populate.
#  CFL condition: dt = |Ω_ij| / max_face_speed.

# The per-cell wave speed estimate for the CFL condition in `max_dt` below:
# how fast a signal can cross cell (i, j)'s faces, scaled by their length.
# We take the larger of the two opposing faces per direction (they can differ on a
# curved mesh) and add both directions together.
#
# On a Cartesian grid this matches `TreeMesh`'s estimate: each face
# normal points along a single axis with magnitude h, so `speed_x = h·λ1`,
# `speed_y = h·λ2`, giving `max_face_speed = h·(λ1+λ2)`. Divided by `cell_volume =
# h²` below, resulting in `dt = h / (λ1+λ2)` - the same CFL estimate as
# `TreeMesh`'s version in `blockfv_2d.jl`.
#
# We use `Base.max` to prevent silent failures, as `max` from `@fastmath` doesn't propagate
# `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
@inline function max_face_speed(normal_x, normal_y, i, j, element, lambda1, lambda2)
    nx_l, ny_l = normal_x[1, i, j, element], normal_x[2, i, j, element]
    nx_r, ny_r = normal_x[1, i + 1, j, element], normal_x[2, i + 1, j, element]
    speed_x = Base.max(abs(nx_l) * lambda1 + abs(ny_l) * lambda2,
                       abs(nx_r) * lambda1 + abs(ny_r) * lambda2)

    nx_b, ny_b = normal_y[1, i, j, element], normal_y[2, i, j, element]
    nx_t, ny_t = normal_y[1, i, j + 1, element], normal_y[2, i, j + 1, element]
    speed_y = Base.max(abs(nx_b) * lambda1 + abs(ny_b) * lambda2,
                       abs(nx_t) * lambda1 + abs(ny_t) * lambda2)

    return speed_x + speed_y
end

function max_dt(u, t, mesh::P4estMesh{2}, constant_speed::False, equations,
                dg::BlockFV, cache)
    @unpack inverse_jacobian = cache.elements
    @unpack normal_x, normal_y = cache
    max_scaled_speed = nextfloat(zero(t))

    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            lambda1, lambda2 = max_abs_speeds(u_node, equations)
            speed = max_face_speed(normal_x, normal_y, i, j, element, lambda1, lambda2)
            cell_volume = abs(inv(inverse_jacobian[i, j, element]))
            # Use `Base.max` to prevent silent failures, as `max` from `@fastmath`
            # doesn't propagate `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
            max_scaled_speed = Base.max(max_scaled_speed, speed / cell_volume)
        end
    end

    return inv(max_scaled_speed)
end

function max_dt(u, t, mesh::P4estMesh{2}, constant_speed::True, equations,
                dg::BlockFV, cache)
    @unpack inverse_jacobian = cache.elements
    @unpack normal_x, normal_y = cache
    max_scaled_speed = nextfloat(zero(t))
    lambda1, lambda2 = max_abs_speeds(equations)

    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            speed = max_face_speed(normal_x, normal_y, i, j, element, lambda1, lambda2)
            cell_volume = abs(inv(inverse_jacobian[i, j, element]))
            # Use `Base.max` to prevent silent failures, as `max` from `@fastmath`
            # doesn't propagate `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
            max_scaled_speed = Base.max(max_scaled_speed, speed / cell_volume)
        end
    end

    return inv(max_scaled_speed)
end

#####################################################################
# `inverse_jacobian` is already 1/|Ω_ij|, so no extra quadrature weights needed here.

function integrate_via_indices(func::Func, u,
                               mesh::P4estMesh{2}, equations,
                               dg::BlockFV, cache, args...;
                               normalize = true) where {Func}
    @unpack inverse_jacobian = cache.elements

    integral = zero(func(u, 1, 1, 1, equations, dg, args...))
    total_volume_ = zero(eltype(inverse_jacobian))

    @batch reduction=((+, integral), (+, total_volume_)) for element in eachelement(dg,
                                                                                    cache)
        for j in eachnode(dg)
            for i in eachnode(dg)
                cell_volume = abs(inv(inverse_jacobian[i, j, element]))
                integral += cell_volume * func(u, i, j, element, equations, dg, args...)
                total_volume_ += cell_volume
            end
        end
    end

    if normalize
        integral = integral / total_volume_
    end

    return integral
end

#####################################################################
# Discrete L2 and L∞ error norms in 2D
# Evaluates the exact solution at each FV cell center (no interpolation needed).

function calc_error_norms(func, u, t, analyzer,
                          mesh::P4estMesh{2}, equations, initial_condition,
                          dg::BlockFV, cache, cache_analysis)
    @unpack node_coordinates, inverse_jacobian = cache.elements

    l2_error = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
    linf_error = copy(l2_error)
    total_volume_ = zero(eltype(inverse_jacobian))

    for element in eachelement(dg, cache)
        for j in eachnode(dg)
            for i in eachnode(dg)
                cell_volume = abs(inv(inverse_jacobian[i, j, element]))
                x = get_node_coords(node_coordinates, equations, dg, i, j, element)
                u_exact = initial_condition(x, t, equations)
                diff = func(u_exact, equations) -
                       func(get_node_vars(u, equations, dg, i, j, element), equations)
                l2_error += diff .^ 2 * cell_volume
                linf_error = @. max(linf_error, abs(diff))
                total_volume_ += cell_volume
            end
        end
    end

    l2_error = @. sqrt(l2_error / total_volume_)
    return l2_error, linf_error
end
end # @muladd
