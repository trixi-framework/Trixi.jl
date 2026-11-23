# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Thread-local pads `pad[v, a]` with the pad index `a = i + nhalo` for the subcell `i`
# of the current element; the halo cells `a <= nhalo` and `a > n + nhalo` hold the
# subcells of the neighboring elements or the ghost subcells of physical boundaries.
# The neighbors of each element are looked up in `neighbor_elements[side, element]`
# (`side = 1` at -x, `2` at +x; `0` at a physical boundary) and the boundaries in
# `boundary_ids[side, element]`.
function create_cache_volume_integral(mesh::TreeMesh{1}, equations,
                                      volume_integral::VolumeIntegralMacroCellFV,
                                      dg::MacroCellFV, cache_containers, uEltype)
    n = nnodes(dg)
    g = nhalo(dg)
    # The halos are copied from the direct neighbors only
    g <= n ||
        throw(ArgumentError("the reconstruction needs $g halo subcells, which " *
                            "requires `n_cells >= $g`"))
    pad_threaded = [Matrix{uEltype}(undef, npadvariables(volume_integral, equations),
                                    n + 2 * g)
                    for _ in 1:Threads.maxthreadid()]

    @unpack elements, interfaces, boundaries = cache_containers
    n_elements = nelements(elements)
    neighbor_elements = zeros(Int, 2, n_elements)
    for interface in 1:ninterfaces(interfaces)
        left_element = interfaces.neighbor_ids[1, interface]
        right_element = interfaces.neighbor_ids[2, interface]
        neighbor_elements[2, left_element] = right_element
        neighbor_elements[1, right_element] = left_element
    end
    boundary_ids = zeros(Int, 2, n_elements)
    for boundary in 1:nboundaries(boundaries)
        element = boundaries.neighbor_ids[boundary]
        # Element on the left of the boundary: boundary at its +x face
        side = boundaries.neighbor_sides[boundary] == 1 ? 2 : 1
        boundary_ids[side, element] = boundary
    end

    return (; pad_threaded, neighbor_elements, boundary_ids)
end

function resize_cache_volume_integral!(cache, mesh::TreeMesh{1},
                                       volume_integral::VolumeIntegralMacroCellFV,
                                       dg::MacroCellFV, new_size)
    # Only check the size here; AMR is not supported yet.
    @assert size(cache.neighbor_elements, 2) == new_size
    return nothing
end

function rhs_hyperbolic!(backend::Nothing, du, u, t, mesh::TreeMesh{1}, equations,
                         boundary_conditions, source_terms::Source,
                         dg::MacroCellFV, cache) where {Source}
    # The boundary conditions at -x and +x (type-stable access in the element loop)
    boundary_conditions_sides = (boundary_condition_side(boundary_conditions, 1),
                                 boundary_condition_side(boundary_conditions, 2))

    @trixi_timeit timer() "subcell fluxes and divergence" begin
        @threaded for element in eachelement(dg, cache)
            pad = cache.pad_threaded[Threads.threadid()]
            calc_macrocell_element!(du, pad, u, t, element, boundary_conditions_sides,
                                    mesh, equations, dg.volume_integral, dg, cache)
        end
    end

    @trixi_timeit timer() "interface fluxes" begin
        @threaded for element in eachelement(dg, cache)
            finish_macrocell_first_subcell!(du, element, mesh, equations, dg, cache)
        end
    end

    @trixi_timeit timer() "source terms" begin
        calc_sources!(backend, du, u, t, source_terms, equations, dg, cache)
    end

    return nothing
end

@inline boundary_condition_side(boundary_conditions::BoundaryConditionPeriodic, side) = boundary_conditions
@inline boundary_condition_side(boundary_conditions, side) = boundary_conditions[side]

# All faces of `element`: the faces inside the macro-cell, the face at +x (stored for both
# adjacent elements in `surface_flux_values`), and the face at -x only at a physical
# boundary (the interface flux at -x is computed by the left neighbor). The flux
# differences are stored in `du`, except for the first subcell, which holds the flux at
# its +x face until `finish_macrocell_first_subcell!` subtracts the flux at -x.
function calc_macrocell_element!(du, pad, u, t, element, boundary_conditions_sides,
                                 mesh::TreeMesh{1}, equations, volume_integral,
                                 dg::MacroCellFV, cache)
    @unpack surface_flux_values = cache.elements
    @unpack neighbor_elements, boundary_ids = cache
    g = nhalo(dg)
    n = nnodes(dg)

    gather_pad!(pad, u, t, element, boundary_conditions_sides, equations, dg, cache)
    invert_and_transform_pad!(pad, volume_integral.moment_inversion, equations,
                              volume_integral)
    line_pad = macrocell_pad_1d(pad, has_point_values(volume_integral.moment_inversion))

    inv_h = inv_subcell_width(mesh, dg, cache, element)

    if boundary_ids[1, element] != 0
        line = pad_line_1d(line_pad, equations, 1, g)
        flux_boundary = macrocell_face_flux(line, 1, equations, volume_integral)
        set_node_vars!(surface_flux_values, flux_boundary, equations, dg, 1, element)
    end

    # Flux at the +x face of the first subcell (completed later)
    line = pad_line_1d(line_pad, equations, 2, g)
    flux_left = macrocell_face_flux(line, 1, equations, volume_integral)
    set_node_vars!(du, flux_left, equations, dg, 1, element)
    for face in 3:(n + 1)
        line = pad_line_1d(line_pad, equations, face, g)
        flux_right = macrocell_face_flux(line, 1, equations, volume_integral)
        set_node_vars!(du, -inv_h * (flux_right - flux_left), equations, dg, face - 1,
                       element)
        flux_left = flux_right
    end
    # The face at +x (for `n = 1`, the face at +x of the first subcell): computed once,
    # stored for both elements
    set_node_vars!(surface_flux_values, flux_left, equations, dg, 2, element)
    right_element = neighbor_elements[2, element]
    if right_element != 0
        set_node_vars!(surface_flux_values, flux_left, equations, dg, 1, right_element)
    end

    return nothing
end

# du[1] = -(f[3/2] - f[1/2]) / h with the flux at the -x face of the element
@inline function finish_macrocell_first_subcell!(du, element, mesh::TreeMesh{1},
                                                 equations, dg::MacroCellFV,
                                                 cache)
    flux_right = get_node_vars(du, equations, dg, 1, element)
    flux_left = get_node_vars(cache.elements.surface_flux_values, equations, dg, 1,
                              element)
    inv_h = inv_subcell_width(mesh, dg, cache, element)
    set_node_vars!(du, -inv_h * (flux_right - flux_left), equations, dg, 1, element)
    return nothing
end

# The subcells of `element` and the halos from the neighbors or the ghost subcells of
# physical boundaries, in conservative variables
@inline function gather_pad!(pad, u, t, element, boundary_conditions_sides, equations,
                             dg, cache)
    @unpack neighbor_elements, boundary_ids = cache
    g = nhalo(dg)
    n = nnodes(dg)

    for i in eachnode(dg), v in eachvariable(equations)
        pad[v, i + g] = u[v, i, element]
    end

    left_element = neighbor_elements[1, element]
    if left_element != 0
        for k in 1:g, v in eachvariable(equations)
            pad[v, g + 1 - k] = u[v, n + 1 - k, left_element]
        end
    else
        gather_boundary_halo!(pad, u, t, boundary_conditions_sides[1], 1,
                              boundary_ids[1, element], element, equations, dg, cache)
    end

    right_element = neighbor_elements[2, element]
    if right_element != 0
        for k in 1:g, v in eachvariable(equations)
            pad[v, n + g + k] = u[v, k, right_element]
        end
    else
        gather_boundary_halo!(pad, u, t, boundary_conditions_sides[2], 2,
                              boundary_ids[2, element], element, equations, dg, cache)
    end

    return nothing
end

# Periodic meshes have no physical boundaries
@inline function gather_boundary_halo!(pad, u, t, ::BoundaryConditionPeriodic, side,
                                       boundary, element, equations, dg, cache)
    return nothing
end

# Physical boundaries are handled by ghost subcells in the halo, so that the
# boundary faces use the same reconstruction and numerical fluxes as all other faces.
@inline function gather_boundary_halo!(pad, u, t, boundary_condition, side, boundary,
                                       element, equations, dg, cache)
    g = nhalo(dg)
    n = nnodes(dg)
    h = 2 / (cache.elements.inverse_jacobian[element] * n)
    x_face = cache.boundaries.node_coordinates[1, boundary]
    # Outward unit normal of the face
    normal = side == 1 ? -one(h) : one(h)
    # Interior subcell `layer` (1 next to the face): mirror image of the ghost
    # subcell `layer`
    interior = GhostColumn1D(u, equations, dg, side == 1, n, element)
    for layer in 1:g
        # Index of the ghost subcell `layer` in the pad
        a = side == 1 ? g + 1 - layer : g + n + layer
        x_ghost = SVector(x_face + (layer - 0.5f0) * h * normal)
        u_ghost = macrocell_ghost_layer(boundary_condition, interior, layer, x_ghost, t,
                                        equations)
        for v in eachvariable(equations)
            pad[v, a] = u_ghost[v]
        end
    end
    return nothing
end

# The interior subcells next to the face of `element` at -x (`minus`) or +x,
# `column(layer)` with `layer = 1` next to the face (a callable struct instead of a
# closure, which would box the captured variables)
struct GhostColumn1D{U, Equations, Solver}
    u::U
    equations::Equations
    dg::Solver
    minus::Bool
    n::Int
    element::Int
end

@inline function (column::GhostColumn1D)(layer)
    @unpack u, equations, dg, minus, n, element = column
    return get_node_vars(u, equations, dg, minus ? layer : n + 1 - layer, element)
end

# Transmissive boundary: the ghost subcells mirror the interior subcells
@inline function macrocell_ghost_state(::BoundaryConditionDoNothing, u_mirror, x, t,
                                       equations)
    return u_mirror
end

# Dirichlet boundary: the boundary values evaluated at the ghost subcell centers
@inline function macrocell_ghost_state(boundary_condition::BoundaryConditionDirichlet,
                                       u_mirror, x, t, equations)
    return boundary_condition.boundary_value_function(x, t, equations)
end

# Ghost state of the halo layer `layer` (1 next to the face) of a boundary face;
# `interior(l)` returns the conservative variables of the interior subcell `l` (1 next
# to the face), which is the mirror image of the ghost subcell `layer = l`
@inline function macrocell_ghost_layer(boundary_condition, interior, layer, x_ghost, t,
                                       equations)
    return macrocell_ghost_state(boundary_condition, interior(layer), x_ghost, t,
                                 equations)
end

# Read the state vector (conservative or reconstruction variables) of pad cell `a`
@inline function get_pad_vars(pad, equations, a)
    return SVector(ntuple(@inline(v->pad[v, a]), Val(nvariables(equations))))
end

# Without point values, the pad cells are transformed to the reconstruction variables
@inline function invert_and_transform_pad!(pad, ::MomentInversionIdentity, equations,
                                           volume_integral)
    transform_pad!(pad, volume_integral.cons2recon, equations)
    return nothing
end

@inline transform_pad!(pad, ::typeof(cons2cons), equations) = nothing

@inline function transform_pad!(pad, cons2recon, equations)
    for a in axes(pad, 2)
        w = cons2recon(get_pad_vars(pad, equations, a), equations)
        for v in eachvariable(equations)
            pad[v, a] = w[v]
        end
    end
    return nothing
end

# Pad as seen by the reconstruction (the pad itself without point values)
@inline macrocell_pad_1d(pad, ::False) = pad

# Coordinate line through the pad, with `line(0)` the subcell left of the face with
# local index `face` (the face between the subcells `face - 1` and `face`) and
# `line(1)` the subcell right of it
@inline function pad_line_1d(pad, equations, face, g)
    return @inline(offset->get_pad_vars(pad, equations, face - 1 + g + offset))
end

# Inverse subcell width of `element` (`inverse_jacobian = 2 / dx` of the element)
@inline function inv_subcell_width(mesh::TreeMesh{1}, dg::MacroCellFV, cache,
                                   element)
    return cache.elements.inverse_jacobian[element] * nnodes(dg) / 2
end

# Initial condition: point values at the subcell centers (as for all DG solvers) or
# subcell averages by Gauss-Legendre quadrature
function compute_coefficients!(backend::Nothing, u, func, t, mesh::TreeMesh{1},
                               equations, dg::MacroCellFV, cache)
    n_quadrature = dg.volume_integral.initial_condition_nodes
    if n_quadrature == 1
        return invoke(compute_coefficients!,
                      Tuple{Nothing, typeof(u), typeof(func), typeof(t),
                            AbstractMesh{1}, typeof(equations), DG, typeof(cache)},
                      backend, u, func, t, mesh, equations, dg, cache)
    end
    nodes, weights = gauss_nodes_weights(n_quadrature, real(dg))
    @threaded for element in eachelement(dg, cache)
        h = 1 / inv_subcell_width(mesh, dg, cache, element)
        for i in eachnode(dg)
            x_c = get_node_coords(cache.elements.node_coordinates, equations, dg, i,
                                  element)
            u_avg = zero(func(x_c, t, equations))
            for q in eachindex(nodes)
                x = SVector(x_c[1] + 0.5f0 * h * nodes[q])
                u_avg = u_avg + 0.5f0 * weights[q] * func(x, t, equations)
            end
            set_node_vars!(u, u_avg, equations, dg, i, element)
        end
    end

    return nothing
end
end # @muladd
