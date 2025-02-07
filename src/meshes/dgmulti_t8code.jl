# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function correct_negative_volume(eclass, vertices)
    perm = false
    result = similar(vertices)
    @. result = vertices

    if eclass == T8_ECLASS_TRIANGLE || eclass == T8_ECLASS_QUAD
        # Check if tree's node ordering is right-handed or print a warning.
        let z = zero(eltype(vertices)), o = one(eltype(vertices))
            u = vertices[:, 2] - vertices[:, 1]
            v = vertices[:, 3] - vertices[:, 1]
            w = [z, z, o]

            # Triple product gives signed volume of spanned parallelepiped.
            vol = dot(cross(u, v), w)
            # println("vol = ", vol)

            if vol < z
                # @warn("Discovered negative volumes in `cmesh`: vol = $vol")
                # Flip vertices.
                @. result[:, 2] = vertices[:, 3]
                @. result[:, 3] = vertices[:, 2]

                perm = true
            end
        end

    elseif eclass == T8_ECLASS_TET
        # Check if tree's node ordering is right-handed or print a warning.
        let z = zero(eltype(vertices)), o = one(eltype(vertices))
            u = vertices[:, 2] - vertices[:, 1]
            v = vertices[:, 3] - vertices[:, 1]
            w = vertices[:, 4] - vertices[:, 1]

            # Triple product gives signed volume of spanned parallelepiped.
            vol = dot(cross(u, v), w)

            if vol < z
                # @warn("Discovered negative volumes in `cmesh`: vol = $vol")
                # Flip vertices.
                @. result[:, 2] = vertices[:, 3]
                @. result[:, 3] = vertices[:, 2]

                perm = true
            end
        end

    elseif eclass == T8_ECLASS_HEX
        # Check if tree's node ordering is right-handed or print a warning.
        let z = zero(eltype(vertices)), o = one(eltype(vertices))
            u = vertices[:, 2] - vertices[:, 1]
            v = vertices[:, 3] - vertices[:, 1]
            w = vertices[:, 5] - vertices[:, 1]

            # Triple product gives signed volume of spanned parallelepiped.
            vol = dot(cross(u, v), w)

            if vol < z
                # @warn("Discovered negative volumes in `cmesh`: vol = $vol")
                # Flip vertices.
                @. result[:, 2] = vertices[:, 3]
                @. result[:, 3] = vertices[:, 2]

                @. result[:, 6] = vertices[:, 7]
                @. result[:, 7] = vertices[:, 6]

                perm = true
            end
        end
    elseif eclass == T8_ECLASS_PRISM
        # Check if tree's node ordering is right-handed or print a warning.
        let z = zero(eltype(vertices)), o = one(eltype(vertices))
            u = vertices[:, 2] - vertices[:, 1]
            v = vertices[:, 3] - vertices[:, 1]
            w = vertices[:, 4] - vertices[:, 1]

            # Triple product gives signed volume of spanned parallelepiped.
            vol = dot(cross(u, v), w)

            if vol < z
                # @warn("Discovered negative volumes in `cmesh`: vol = $vol")
                # Flip vertices.
                @. result[:, 2] = vertices[:, 3]
                @. result[:, 3] = vertices[:, 2]

                @. result[:, 5] = vertices[:, 6]
                @. result[:, 6] = vertices[:, 5]

                perm = true
            end
        end
    else
        error("Unknown eclass.")
    end

    return result, perm
end

function t8code2startupdg(eclass, EToV)
    result = similar(EToV)
    @. result = EToV

    if eclass == Trixi.T8code.T8_ECLASS_LINE
        # Do nothing.
    elseif eclass == Trixi.T8code.T8_ECLASS_TRIANGLE
        # Do nothing.

    elseif eclass == Trixi.T8code.T8_ECLASS_TET
        # Do nothing.

    elseif eclass == Trixi.T8code.T8_ECLASS_QUAD
        result[1] = EToV[1]
        result[2] = EToV[2]
        result[3] = EToV[3]
        result[4] = EToV[4]

    elseif eclass == Trixi.T8code.T8_ECLASS_HEX
        result[2] = EToV[3]
        result[3] = EToV[2]
        result[6] = EToV[7]
        result[7] = EToV[6]

    elseif eclass == Trixi.T8code.T8_ECLASS_PRISM
        # Do nothing.
    else
        error("Unsupported element class.")
    end

    return result
end

function compute_EToV(forest::Ptr{t8_forest})
    # Dimension of the mesh.
    ndims = Int(t8_cmesh_get_dimension(t8_forest_get_cmesh(forest)))

    # Number of local elements.
    num_elements = t8_forest_get_local_num_elements(forest)

    # Number of element corners.
    tree_class = t8_forest_get_tree_class(forest, 0)
    elem_class = t8_forest_get_eclass_scheme(forest, tree_class)
    element = t8_forest_get_element_in_tree(forest, 0, 0)
    num_corners = t8_element_num_corners(elem_class, element)
    eclass = t8_forest_get_eclass(forest, 0)

    # Holds all element vertices.
    # vxyz = Array{Float64,3}(undef,3,num_corners,num_elements)
    vxyz = zeros(3, num_corners, num_elements)
    verts = zeros(3, num_corners)

    # Loop over all local trees.
    current_element = 1
    num_local_trees = t8_forest_get_num_local_trees(forest)
    for itree in 1:num_local_trees
        num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree - 1)

        # Loop over local all elements per tree.
        for ielem in 1:num_elements_in_tree
            element = t8_forest_get_element_in_tree(forest, itree - 1, ielem - 1)

            for ivert in 1:num_corners
                t8_forest_element_coordinate(forest, itree - 1, element, ivert - 1,
                                             @view(verts[:, ivert]))
            end

            vxyz[:, :, current_element], _ = correct_negative_volume(eclass, verts)

            current_element += 1
        end
    end

    EToV = zeros(Int, num_elements, num_corners)
    VXYZ = Tuple(Vector{Float64}() for idim in 1:ndims)

    etov = zeros(Int, num_corners)

    min_range = vxyz |> minimum
    max_range = vxyz |> maximum

    num_bins = 1e9
    inverse_bin_size = num_bins / (max_range - min_range)

    hashed_vertices = Dict{Tuple{Tuple(UInt128 for idim in 1:ndims)...}, Int}()

    cumulative_index = 0
    for ielem in 1:num_elements
        for ivert in 1:num_corners
            hashed_xyz = Tuple(trunc(UInt128,
                                     (vxyz[idim, ivert, ielem] - min_range + 0.5) *
                                     inverse_bin_size) for idim in 1:ndims)

            if haskey(hashed_vertices, hashed_xyz)
                index = hashed_vertices[hashed_xyz]
            else
                index = cumulative_index += 1
                hashed_vertices[hashed_xyz] = index
                for idim in 1:ndims
                    push!(VXYZ[idim], vxyz[idim, ivert, ielem])
                end
            end
            etov[ivert] = index
        end

        EToV[ielem, :] = t8code2startupdg(eclass, etov)
    end

    return VXYZ, EToV
end

function compute_coordinates(forest::Ptr{t8_forest}, rd::RefElemData, md::MeshData)
    if rd.element_type isa StartUpDG.Tri
        e_rst = [[1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]]
    elseif rd.element_type isa StartUpDG.Quad
        e_rst = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]]
    elseif rd.element_type isa StartUpDG.Tet
        e_rst = [[1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]]
    elseif rd.element_type isa StartUpDG.Wedge
        e_rst = [[1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]
    elseif rd.element_type isa StartUpDG.Hex
        e_rst = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]
    else
        @error element_type
    end

    ndims = length(md.xyz)
    xyz = tuple([similar(md.x) for _ in 1:ndims]...)

    num_local_trees = t8_forest_get_num_local_trees(forest)

    current_element = 1
    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

        eclass = t8_forest_get_eclass(forest, itree)

        num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)
        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_element_in_tree(forest, itree, ielement)
            num_corners = t8_element_num_corners(eclass_scheme, element)

            verts = zeros(3, num_corners)

            for ivert in 1:num_corners
                t8_forest_element_coordinate(forest, itree, element, ivert - 1,
                                             @view(verts[:, ivert]))
            end

            _, do_perm = correct_negative_volume(eclass, verts)

            if do_perm
                perm = [2, 1, 3]
            else
                perm = [1, 2, 3]
            end

            for i in 1:length(rd.r)
                ref_coords = zeros(3)
                out_coords = Vector{Cdouble}(undef, 3)

                for iref in 1:ndims
                    rst = 0.5 * (1.0 + rd.rst[perm[iref]][i])
                    ref_coords += rst * e_rst[iref]
                end

                t8_forest_element_from_ref_coords(forest, itree, element,
                                                  pointer(ref_coords), 1,
                                                  pointer(out_coords))

                for idim in 1:ndims
                    xyz[idim][i, current_element] = out_coords[idim]
                end
            end

            current_element += 1
        end
    end

    return xyz
end

# This routine may be merged into StartUpDG.jl in the future.
function tag_boundary_nodes(md::MeshData, boundary_name::Symbol = :entire_boundary)
    return Dict(boundary_name => md.mapB)
end

# This routine may be merged into StartUpDG.jl in the future.
function tag_boundary_nodes(md::MeshData, is_on_boundary::Dict{Symbol, <:Function})
    tagged_nodes = (md.mapB[is_on_boundary[k].((xf[md.mapB] for xf in md.xyzf)...)] for k in keys(is_on_boundary))
    return Dict(Pair.(keys(is_on_boundary), tagged_nodes))
end
end # @muladd
