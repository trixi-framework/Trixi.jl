# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Thread-local pads `pad[a, v]` with the pad index `a = i + nhalo` for the subcell `i`
# of the current element; the halo cells `a <= nhalo` and `a > n + nhalo` hold the
# subcells of the neighboring elements or the ghost subcells of physical boundaries.
# The pad index runs fastest, so that each variable of the line is contiguous and the
# reconstruction loops over the subcells vectorize.
# Only the halo cells read by the reconstruction are filled: the face at -x of the
# element is computed by the left neighbor, so with a left neighbor the stencils reach
# only `nhalo - 1` cells to the left and the pad cell `a = 1` stays unused; at a physical
# boundary at -x, the boundary face needs all `nhalo` ghost subcells.
# With the moment inversions to point values, the point values are stored in a second
# thread-local pad `pad_points` of the same size (the averages stay in `pad`); they are
# computed only in the pad cells read by the reconstruction, so that the stencils of
# the moment inversion lie inside the filled cells.
# Thread-local line buffers hold the reconstructed states `w_minus[c + 1, v]` and
# `w_plus[c + 1, v]` of the subcells `c = 0, ..., n + 1` at their faces in negative and
# positive x direction and the numerical fluxes `flux[f, v]` at the faces
# `f = 1, ..., n + 1` (face `f` between the subcells `f - 1` and `f`).
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
    pad_threaded = [Matrix{uEltype}(undef, n + 2 * g,
                                    npadvariables(volume_integral, equations))
                    for _ in 1:Threads.maxthreadid()]
    n_pad_points = has_point_values(volume_integral.moment_inversion) isa True ?
                   n + 2 * g : 0
    pad_points_threaded = [Matrix{uEltype}(undef, n_pad_points,
                                           npadvariables(volume_integral, equations))
                           for _ in 1:Threads.maxthreadid()]
    line_buffers_threaded = [(;
                              w_minus = Matrix{uEltype}(undef, n + 2,
                                                        nvariables(equations)),
                              w_plus = Matrix{uEltype}(undef, n + 2,
                                                       nvariables(equations)),
                              flux = Matrix{uEltype}(undef, n + 1,
                                                     nvariables(equations)))
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

    # Gauss-Legendre quadrature on [-1, 1] for the subcell averages of the initial
    # condition and of the Dirichlet ghost subcells
    nodes, weights = gauss_nodes_weights(volume_integral.initial_condition_nodes,
                                         real(dg))
    subcell_quadrature = (; nodes, weights)

    return (; pad_threaded, pad_points_threaded, line_buffers_threaded,
            neighbor_elements, boundary_ids, subcell_quadrature)
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
            pad_points = cache.pad_points_threaded[Threads.threadid()]
            line_buffers = cache.line_buffers_threaded[Threads.threadid()]
            calc_macrocell_element!(du, pad, pad_points, line_buffers, u, t, element,
                                    boundary_conditions_sides, mesh, equations,
                                    dg.volume_integral, dg, cache)
        end
    end

    @trixi_timeit timer() "interface fluxes" begin
        @threaded for element in eachelement(dg, cache)
            finish_macrocell_first_subcell!(du, element, mesh, equations, dg, cache)
        end
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
# The line is processed in two passes, each a loop that vectorizes: all subcells are
# reconstructed (one variable at a time), then the fluxes at all faces are computed.
function calc_macrocell_element!(du, pad, pad_points, line_buffers, u, t, element,
                                 boundary_conditions_sides, mesh::TreeMesh{1},
                                 equations, volume_integral, dg::MacroCellFV, cache)
    @unpack surface_flux_values = cache.elements
    @unpack neighbor_elements, boundary_ids = cache
    @unpack w_minus, w_plus, flux = line_buffers
    g = nhalo(dg)
    n = nnodes(dg)

    # Copy the subcells of the macro-cell and the halos from the neighbors or the
    # boundaries, then the moment inversion (cell averages to point values) and the
    # transformation to the reconstruction variables
    a_first = gather_pad!(pad, u, t, element, boundary_conditions_sides, equations, dg,
                          cache)
    line_pad, a_first_line, a_last_line = pad_moment_inversion!(pad_points, pad,
                                                                a_first,
                                                                volume_integral.moment_inversion,
                                                                equations)
    pad_center2recon!(line_pad, a_first_line, a_last_line, volume_integral.center2recon,
                      equations)

    # The first face: the boundary face at -x (between the ghost subcell 0 and the
    # subcell 1) at a physical boundary, otherwise the +x face of the subcell 1
    f_first = boundary_ids[1, element] != 0 ? 1 : 2
    reconstruct_line!(w_minus, w_plus, line_pad, volume_integral.reconstruction,
                      f_first - 1, n + 1, g, equations)
    calc_line_fluxes!(flux, w_minus, w_plus, f_first, n + 1, equations, volume_integral)

    inv_h = inv_subcell_width(mesh, dg, cache, element)
    right_element = neighbor_elements[2, element]
    @inbounds for v in eachvariable(equations)
        if f_first == 1
            surface_flux_values[v, 1, element] = flux[1, v]
        end
        # The flux at the +x face of the first subcell (completed later)
        du[v, 1, element] = flux[2, v]
        for i in 2:n
            du[v, i, element] = -inv_h * (flux[i + 1, v] - flux[i, v])
        end
        # The face at +x: computed once, stored for both elements
        surface_flux_values[v, 2, element] = flux[n + 1, v]
        if right_element != 0
            surface_flux_values[v, 1, right_element] = flux[n + 1, v]
        end
    end

    return nothing
end

# Reconstruct the subcells `c_first:c_last` of the line in the pad, one variable at a
# time (vectorized over the subcells). The passes over the line are separate (not
# inlined) functions. Inlined into the element kernel, their loops were not vectorized.
# The pad has `nhalo(dg) >= nhalo(reconstruction)` halo cells on each side, so the
# stencils of the subcells `0:(n + 1)` lie inside the pad (checked by the test suite
# with `--check-bounds=yes`).
@noinline function reconstruct_line!(w_minus, w_plus, pad, reconstruction, c_first,
                                     c_last, g, equations)
    for v in eachvariable(equations)
        @inbounds @simd ivdep for c in c_first:c_last
            a = c + g
            wm, wp = reconstruct_cell(reconstruction,
                                      @inline(offset->@inbounds(pad[a + offset, v])))
            w_minus[c + 1, v] = wm
            w_plus[c + 1, v] = wp
        end
    end
    return nothing
end

# Numerical fluxes at the faces `f_first:f_last` (vectorized over the faces)
@noinline function calc_line_fluxes!(flux, w_minus, w_plus, f_first, f_last, equations,
                                     volume_integral)
    @inbounds @simd ivdep for f in f_first:f_last
        # The subcell `f - 1` on the left, the subcell `f` on the right of the face
        w_ll = SVector(ntuple(@inline(v->w_plus[f, v]), Val(nvariables(equations))))
        w_rr = SVector(ntuple(@inline(v->w_minus[f + 1, v]),
                              Val(nvariables(equations))))
        flux_face = macrocell_face_flux(w_ll, w_rr, 1, equations, volume_integral)
        for v in eachvariable(equations)
            flux[f, v] = flux_face[v]
        end
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
# physical boundaries, in conservative variables. Returns the first filled pad index
# (`1` at a physical boundary at -x, otherwise `2`).
@inline function gather_pad!(pad, u, t, element, boundary_conditions_sides, equations,
                             dg, cache)
    @unpack neighbor_elements, boundary_ids = cache
    g = nhalo(dg)
    n = nnodes(dg)

    # (`create_cache_volume_integral` ensures `n >= g`, so all halo subcells lie in the
    # neighbors)
    @inbounds for v in eachvariable(equations), i in eachnode(dg)
        pad[i + g, v] = u[v, i, element]
    end

    left_element = neighbor_elements[1, element]
    if left_element != 0
        # The face at -x is computed by the left neighbor: `g - 1` halo cells suffice
        @inbounds for v in eachvariable(equations), k in 1:(g - 1)
            pad[g + 1 - k, v] = u[v, n + 1 - k, left_element]
        end
        a_first = 2
    else
        gather_boundary_halo!(pad, u, t, boundary_conditions_sides[1], 1,
                              boundary_ids[1, element], element, equations, dg, cache)
        a_first = 1
    end

    right_element = neighbor_elements[2, element]
    if right_element != 0
        @inbounds for v in eachvariable(equations), k in 1:g
            pad[n + g + k, v] = u[v, k, right_element]
        end
    else
        gather_boundary_halo!(pad, u, t, boundary_conditions_sides[2], 2,
                              boundary_ids[2, element], element, equations, dg, cache)
    end

    return a_first
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
        u_ghost = macrocell_ghost_layer(boundary_condition, interior, layer, x_ghost, h,
                                        t, cache.subcell_quadrature, equations)
        for v in eachvariable(equations)
            pad[a, v] = u_ghost[v]
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
@inline function macrocell_ghost_state(::BoundaryConditionDoNothing, u_mirror, x, h, t,
                                       quadrature, equations)
    return u_mirror
end

# Dirichlet boundary: the averages of the boundary values over the ghost subcells (of
# width `h`, centered at `x`) with the quadrature of the initial condition, see
# `initial_condition_nodes` (with one node: the value at the ghost subcell center)
@inline function macrocell_ghost_state(boundary_condition::BoundaryConditionDirichlet,
                                       u_mirror, x, h, t, quadrature, equations)
    @unpack boundary_value_function = boundary_condition
    @unpack nodes, weights = quadrature
    u_avg = 0.5f0 * weights[1] *
            boundary_value_function(SVector(x[1] + 0.5f0 * h * nodes[1]), t, equations)
    for q in 2:length(nodes)
        x_q = SVector(x[1] + 0.5f0 * h * nodes[q])
        u_avg = u_avg + 0.5f0 * weights[q] * boundary_value_function(x_q, t, equations)
    end
    return u_avg
end

# Ghost state of the halo layer `layer` (1 next to the face) of a boundary face;
# `interior(l)` returns the conservative variables of the interior subcell `l` (1 next
# to the face), which is the mirror image of the ghost subcell `layer = l`
@inline function macrocell_ghost_layer(boundary_condition, interior, layer, x_ghost, h,
                                       t, quadrature, equations)
    return macrocell_ghost_state(boundary_condition, interior(layer), x_ghost, h, t,
                                 quadrature, equations)
end

# Read the state vector (conservative or reconstruction variables) of pad cell `a`
# (as `get_node_vars`: one explicit bounds check, removed by calling it with `@inbounds`)
Base.@propagate_inbounds function get_pad_vars(pad, equations, a)
    @boundscheck checkbounds(pad, a, eachvariable(equations))
    return SVector(ntuple(@inline(v->@inbounds(pad[a, v])), Val(nvariables(equations))))
end

# Moment inversion of the filled pad cells `a >= a_first`; returns the pad read by the
# reconstruction and the range of its cells that hold values
# (`MomentInversionIdentity`: the averages in the pad itself)
@inline function pad_moment_inversion!(pad_points, pad, a_first,
                                       ::MomentInversionIdentity, equations)
    return pad, a_first, size(pad, 1)
end

# Moment inversions to point values: the point values of the pad cells whose stencils
# lie in the filled cells `a_first:size(pad, 1)`, which are all cells read by the
# reconstruction (`nhalo(volume_integral)` includes the halo of the moment inversion)
@inline function pad_moment_inversion!(pad_points, pad, a_first,
                                       inversion::MomentInversionPoint, equations)
    m = nhalo(inversion)
    a_first_points = a_first + m
    a_last_points = size(pad, 1) - m
    invert_line!(pad_points, pad, inversion, a_first_points, a_last_points, equations)
    # Inadmissible point values are replaced by the averages
    @inbounds for a in a_first_points:a_last_points
        u_center = get_pad_vars(pad_points, equations, a)
        if !macrocell_point_value_admissible(u_center, equations)
            for v in eachvariable(equations)
                pad_points[a, v] = pad[a, v]
            end
        end
    end
    return pad_points, a_first_points, a_last_points
end

# Point values of the pad cells `a_first:a_last`, one variable at a time (vectorized
# over the cells, see `reconstruct_line!`)
@noinline function invert_line!(pad_points, pad, inversion, a_first, a_last, equations)
    for v in eachvariable(equations)
        @inbounds @simd ivdep for a in a_first:a_last
            pad_points[a, v] = center_value(inversion,
                                            @inline(offset->@inbounds(pad[a + offset,
                                                                          v])))
        end
    end
    return nothing
end

# Transformation of the pad cells `a_first:a_last` to the reconstruction variables
# (unfilled cells are skipped, since a transformation such as `cons2entropy` may fail
# for arbitrary values)
@inline pad_center2recon!(pad, a_first, a_last, ::typeof(cons2cons), equations) = nothing

@inline function pad_center2recon!(pad, a_first, a_last, center2recon, equations)
    @inbounds for a in a_first:a_last
        w = center2recon(get_pad_vars(pad, equations, a), equations)
        for v in eachvariable(equations)
            pad[a, v] = w[v]
        end
    end
    return nothing
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
    @unpack nodes, weights = cache.subcell_quadrature
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
