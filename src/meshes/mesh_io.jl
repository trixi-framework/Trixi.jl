# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Save current mesh with some context information as an HDF5 file.
function save_mesh_file(mesh::Union{TreeMesh, P4estMesh, T8codeMesh, DGMultiMesh}, output_directory,
                        timestep = 0)
    save_mesh_file(mesh, output_directory, timestep, mpi_parallel(mesh))
end

function save_mesh_file(mesh::TreeMesh, output_directory, timestep,
                        mpi_parallel::False)
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    # Determine file name based on existence of meaningful time step
    if timestep > 0
        filename = joinpath(output_directory, @sprintf("mesh_%09d.h5", timestep))
    else
        filename = joinpath(output_directory, "mesh.h5")
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        n_cells = length(mesh.tree)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["n_cells"] = n_cells
        attributes(file)["capacity"] = mesh.tree.capacity
        attributes(file)["n_leaf_cells"] = count_leaf_cells(mesh.tree)
        attributes(file)["minimum_level"] = minimum_level(mesh.tree)
        attributes(file)["maximum_level"] = maximum_level(mesh.tree)
        attributes(file)["center_level_0"] = mesh.tree.center_level_0
        attributes(file)["length_level_0"] = mesh.tree.length_level_0
        attributes(file)["periodicity"] = collect(mesh.tree.periodicity)

        # Add tree data
        file["parent_ids"] = @view mesh.tree.parent_ids[1:n_cells]
        file["child_ids"] = @view mesh.tree.child_ids[:, 1:n_cells]
        file["neighbor_ids"] = @view mesh.tree.neighbor_ids[:, 1:n_cells]
        file["levels"] = @view mesh.tree.levels[1:n_cells]
        file["coordinates"] = @view mesh.tree.coordinates[:, 1:n_cells]
    end

    return filename
end

# Save current mesh with some context information as an HDF5 file.
function save_mesh_file(mesh::TreeMesh, output_directory, timestep,
                        mpi_parallel::True)
    # Create output directory (if it does not exist)
    mpi_isroot() && mkpath(output_directory)

    # Determine file name based on existence of meaningful time step
    if timestep >= 0
        filename = joinpath(output_directory, @sprintf("mesh_%09d.h5", timestep))
    else
        filename = joinpath(output_directory, "mesh.h5")
    end

    # Since the mesh is replicated on all ranks, only save from MPI root
    if !mpi_isroot()
        return filename
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        n_cells = length(mesh.tree)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["n_cells"] = n_cells
        attributes(file)["capacity"] = mesh.tree.capacity
        attributes(file)["n_leaf_cells"] = count_leaf_cells(mesh.tree)
        attributes(file)["minimum_level"] = minimum_level(mesh.tree)
        attributes(file)["maximum_level"] = maximum_level(mesh.tree)
        attributes(file)["center_level_0"] = mesh.tree.center_level_0
        attributes(file)["length_level_0"] = mesh.tree.length_level_0
        attributes(file)["periodicity"] = collect(mesh.tree.periodicity)

        # Add tree data
        file["parent_ids"] = @view mesh.tree.parent_ids[1:n_cells]
        file["child_ids"] = @view mesh.tree.child_ids[:, 1:n_cells]
        file["neighbor_ids"] = @view mesh.tree.neighbor_ids[:, 1:n_cells]
        file["levels"] = @view mesh.tree.levels[1:n_cells]
        file["coordinates"] = @view mesh.tree.coordinates[:, 1:n_cells]
    end

    return filename
end

# Does not save the mesh itself to an HDF5 file. Instead saves important attributes
# of the mesh, like its size and the type of boundary mapping function.
# Then, within Trixi2Vtk, the StructuredMesh and its node coordinates are reconstructured from
# these attributes for plotting purposes
# Note: the `timestep` argument is needed for compatibility with the method for
# `StructuredMeshView`
function save_mesh_file(mesh::StructuredMesh, output_directory; system = "",
                        timestep = 0)
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    if isempty(system)
        filename = joinpath(output_directory, "mesh.h5")
    else
        filename = joinpath(output_directory, @sprintf("mesh_%s.h5", system))
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["size"] = collect(size(mesh))
        attributes(file)["mapping"] = mesh.mapping_as_string
    end

    return filename
end

# Does not save the mesh itself to an HDF5 file. Instead saves important attributes
# of the mesh, like its size and the corresponding `.mesh` file used to construct the mesh.
# Then, within Trixi2Vtk, the UnstructuredMesh2D and its node coordinates are reconstructured
# from these attributes for plotting purposes
function save_mesh_file(mesh::UnstructuredMesh2D, output_directory)
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    filename = joinpath(output_directory, "mesh.h5")

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["size"] = length(mesh)
        attributes(file)["mesh_filename"] = mesh.filename
        attributes(file)["periodicity"] = collect(mesh.periodicity)
    end

    return filename
end

# Does not save the mesh itself to an HDF5 file. Instead saves important attributes
# of the mesh, like its size and the type of boundary mapping function.
# Then, within Trixi2Vtk, the P4estMesh and its node coordinates are reconstructured from
# these attributes for plotting purposes
function save_mesh_file(mesh::P4estMesh, output_directory, timestep,
                        mpi_parallel::False)
    # Create output directory (if it does not exist)
    mkpath(output_directory)

    # Determine file name based on existence of meaningful time step
    if timestep > 0
        filename = joinpath(output_directory, @sprintf("mesh_%09d.h5", timestep))
        p4est_filename = @sprintf("p4est_data_%09d", timestep)
    else
        filename = joinpath(output_directory, "mesh.h5")
        p4est_filename = "p4est_data"
    end

    p4est_file = joinpath(output_directory, p4est_filename)

    # Save the complete connectivity and `p4est` data to disk.
    save_p4est!(p4est_file, mesh.p4est)

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["p4est_file"] = p4est_filename

        file["tree_node_coordinates"] = mesh.tree_node_coordinates
        file["nodes"] = Vector(mesh.nodes) # the mesh uses `SVector`s for the nodes
        # to increase the runtime performance
        # but HDF5 can only handle plain arrays
        file["boundary_names"] = mesh.boundary_names .|> String
    end

    return filename
end

function save_mesh_file(mesh::P4estMesh, output_directory, timestep, mpi_parallel::True)
    # Create output directory (if it does not exist)
    mpi_isroot() && mkpath(output_directory)

    # Determine file name based on existence of meaningful time step
    if timestep > 0
        filename = joinpath(output_directory, @sprintf("mesh_%09d.h5", timestep))
        p4est_filename = @sprintf("p4est_data_%09d", timestep)
    else
        filename = joinpath(output_directory, "mesh.h5")
        p4est_filename = "p4est_data"
    end

    p4est_file = joinpath(output_directory, p4est_filename)

    # Save the complete connectivity/p4est data to disk.
    save_p4est!(p4est_file, mesh.p4est)

    # Since the mesh attributes are replicated on all ranks, only save from MPI root
    if !mpi_isroot()
        return filename
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["p4est_file"] = p4est_filename

        file["tree_node_coordinates"] = mesh.tree_node_coordinates
        file["nodes"] = Vector(mesh.nodes) # the mesh uses `SVector`s for the nodes
        # to increase the runtime performance
        # but HDF5 can only handle plain arrays
        file["boundary_names"] = mesh.boundary_names .|> String
    end

    return filename
end

# This routine works for both, serial and MPI parallel mode. The forest
# information is collected on all ranks and then gathered by the root rank.
# Since only the `levels` array of UInt8 and the global number of elements per
# tree (Int32) is necessary to reconstruct the forest it is not worth the
# effort to have a collective write to the HDF5 file. Instead, `levels` and
# `num_elements_per_tree` gets gathered by the root rank and written to disk.
function save_mesh_file(mesh::T8codeMesh, output_directory, timestep,
                        mpi_parallel::Union{False, True})

    # Create output directory (if it does not exist).
    mpi_isroot() && mkpath(output_directory)

    # Determine file name based on existence of meaningful time step.
    if timestep > 0
        filename = joinpath(output_directory, @sprintf("mesh_%09d.h5", timestep))
    else
        filename = joinpath(output_directory, "mesh.h5")
    end

    # Retrieve refinement levels of all elements.
    local_levels = get_levels(mesh)
    if mpi_isparallel()
        count = [length(local_levels)]
        counts = MPI.Gather(view(count, 1), mpi_root(), mpi_comm())

        if mpi_isroot()
            levels = similar(local_levels, ncellsglobal(mesh))
            MPI.Gatherv!(local_levels, MPI.VBuffer(levels, counts),
                         mpi_root(), mpi_comm())
        else
            MPI.Gatherv!(local_levels, nothing, mpi_root(), mpi_comm())
        end
    else
        levels = local_levels
    end

    # Retrieve the number of elements per tree. Since a tree can be distributed
    # among multiple ranks a reduction operation sums them all up. The latter
    # is done on the root rank only.
    num_global_trees = t8_forest_get_num_global_trees(mesh.forest)
    num_elements_per_tree = zeros(t8_gloidx_t, num_global_trees)
    num_local_trees = t8_forest_get_num_local_trees(mesh.forest)
    for local_tree_id in 0:(num_local_trees - 1)
        num_local_elements_in_tree = t8_forest_get_tree_num_elements(mesh.forest,
                                                                     local_tree_id)
        global_tree_id = t8_forest_global_tree_id(mesh.forest, local_tree_id)
        num_elements_per_tree[global_tree_id + 1] = num_local_elements_in_tree
    end

    if mpi_isparallel()
        MPI.Reduce!(num_elements_per_tree, +, mpi_comm())
    end

    # Since the mesh attributes are replicated on all ranks, only save from MPI
    # root.
    if !mpi_isroot()
        return filename
    end

    # Retrieve face connectivity info of the coarse mesh.
    treeIDs, neighIDs, faces, duals, orientations = get_cmesh_info(mesh)

    # Open file (clobber existing content).
    h5open(filename, "w") do file
        # Add context information as attributes.
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["ntrees"] = ntrees(mesh)
        attributes(file)["nelements"] = ncellsglobal(mesh)

        file["tree_node_coordinates"] = mesh.tree_node_coordinates
        file["nodes"] = Vector(mesh.nodes)
        file["boundary_names"] = mesh.boundary_names .|> String
        file["treeIDs"] = treeIDs
        file["neighIDs"] = neighIDs
        file["faces"] = faces
        file["duals"] = duals
        file["orientations"] = orientations
        file["levels"] = levels
        file["num_elements_per_tree"] = num_elements_per_tree
    end

    return filename
end

@inline get_VXYZ(md::StartUpDG.MeshData) = get_VXYZ(md.mesh_type)
@inline get_VXYZ(mesh_type::StartUpDG.VertexMappedMesh) = mesh_type.VXYZ
@inline get_VXYZ(mesh_type::StartUpDG.CurvedMesh) = get_VXYZ(mesh_type.original_mesh_type)
@inline get_VXYZ(mesh_type::StartUpDG.HOHQMeshType) = mesh_type.hmd.VXYZ

@inline get_EToV(md::StartUpDG.MeshData) = get_EToV(md.mesh_type)
@inline get_EToV(mesh_type::StartUpDG.VertexMappedMesh) = mesh_type.EToV
@inline get_EToV(mesh_type::StartUpDG.CurvedMesh) = get_EToV(mesh_type.original_mesh_type)
@inline get_EToV(mesh_type::StartUpDG.HOHQMeshType) = mesh_type.hmd.EToV

function save_mesh_file(mesh::DGMultiMesh, output_directory, timestep,
                        mpi_parallel::False)

    # Create output directory (if it does not exist).
    mkpath(output_directory)

    # Determine file name based on existence of meaningful time step.
    if timestep > 0
        filename = joinpath(output_directory, @sprintf("mesh_%09d.h5", timestep))
    else
        filename = joinpath(output_directory, "mesh.h5")
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
      # Add context information as attributes
      attributes(file)["mesh_type"] = get_name(mesh)
      attributes(file)["ndims"] = ndims(mesh)
      attributes(file)["nelements"] = ncells(mesh)

      if mesh.rd.element_type isa Wedge
          attributes(file)["polydeg_tri"] = mesh.rd.N[2]
          attributes(file)["polydeg_line"] = mesh.rd.N[1]
      else
          attributes(file)["polydeg"] = mesh.rd.N
      end

      attributes(file)["element_type"] = mesh.rd.element_type |> typeof |> nameof |> string

      ## TODO: Is this useful to reconstruct `RefElemData` from this?
      ## # Store quad rule.
      ## for idim = 1:ndims(mesh)
      ##   # ASCII: Char(114) => 'r'
      ##   file[(113 + idim |> Char |> string) * "q"] = mesh.rd.rstq[idim]
      ## end
      ## file["wq"] = mesh.rd.wq

      # Mesh-coordinates per element.
      for idim = 1:ndims(mesh)
        # ASCII: Char(120) => 'x'
        file[119 + idim |> Char |> string] = mesh.md.xyz[idim]
      end

      # Transfer vectors of vectors to a matrix (2D array) and store into h5 file.
      for (idim, vectors) in enumerate(get_VXYZ(mesh.md))
        matrix = zeros(length(vectors[1]), length(vectors))
        for ielem = 1:length(vectors)
          @views matrix[:,ielem] .= vectors[ielem]
        end
        # ASCII: Char(58) => 'X'
        # Vertex-coordinates per element.
        file["V" * (87 + idim |> Char |> string)] = matrix
      end

      # Mapping element corners to vertices `VXYZ`.
      file["EToV"] = get_EToV(mesh.md)

      # TODO: Save boundaries.
      # file["boundary_names"] = mesh.boundary_faces .|> String
    end

    return filename
end

"""
    load_mesh(restart_file::AbstractString; n_cells_max)

Load the mesh from the `restart_file`.
"""
function load_mesh(restart_file::AbstractString; n_cells_max = 0, RealT = Float64)
    if mpi_isparallel()
        mesh_file = get_restart_mesh_filename(restart_file, True())
        return load_mesh_parallel(mesh_file; n_cells_max = n_cells_max, RealT = RealT)
    else
        mesh_file = get_restart_mesh_filename(restart_file, False())
        load_mesh_serial(mesh_file; n_cells_max = n_cells_max, RealT = RealT)
    end
end

function load_mesh_serial(mesh_file::AbstractString; n_cells_max, RealT)
    ndims, mesh_type = h5open(mesh_file, "r") do file
        return read(attributes(file)["ndims"]),
               read(attributes(file)["mesh_type"])
    end

    if mesh_type == "TreeMesh"
        capacity = h5open(mesh_file, "r") do file
            return read(attributes(file)["capacity"])
        end
        mesh = TreeMesh(SerialTree{ndims, RealT}, max(n_cells_max, capacity),
                        RealT = RealT)
        load_mesh!(mesh, mesh_file)
    elseif mesh_type in ("StructuredMesh", "StructuredMeshView")
        size_, mapping_as_string = h5open(mesh_file, "r") do file
            return read(attributes(file)["size"]),
                   read(attributes(file)["mapping"])
        end

        size = Tuple(size_)

        # TODO: `@eval` is evil
        #
        # This should be replaced with something more robust and secure,
        # see https://github.com/trixi-framework/Trixi.jl/issues/541).
        if ndims == 1
            mapping = eval(Meta.parse("""function (xi)
                $mapping_as_string
                mapping(xi)
            end
            """))
        elseif ndims == 2
            mapping = eval(Meta.parse("""function (xi, eta)
                $mapping_as_string
                mapping(xi, eta)
            end
            """))
        else # ndims == 3
            mapping = eval(Meta.parse("""function (xi, eta, zeta)
                $mapping_as_string
                mapping(xi, eta, zeta)
            end
            """))
        end

        mesh = StructuredMesh(size, mapping; RealT = RealT, unsaved_changes = false,
                              mapping_as_string = mapping_as_string)
        mesh.current_filename = mesh_file
    elseif mesh_type == "UnstructuredMesh2D"
        mesh_filename, periodicity_ = h5open(mesh_file, "r") do file
            return read(attributes(file)["mesh_filename"]),
                   read(attributes(file)["periodicity"])
        end
        mesh = UnstructuredMesh2D(mesh_filename; RealT = RealT,
                                  periodicity = periodicity_,
                                  unsaved_changes = false)
        mesh.current_filename = mesh_file
    elseif mesh_type == "P4estMesh"
        p4est_filename, tree_node_coordinates,
        nodes, boundary_names_ = h5open(mesh_file, "r") do file
            return read(attributes(file)["p4est_file"]),
                   read(file["tree_node_coordinates"]),
                   read(file["nodes"]),
                   read(file["boundary_names"])
        end

        boundary_names = boundary_names_ .|> Symbol

        p4est_file = joinpath(dirname(mesh_file), p4est_filename)
        # Prevent Julia crashes when `p4est` can't find the file
        @assert isfile(p4est_file)

        p4est = load_p4est(p4est_file, Val(ndims))

        mesh = P4estMesh{ndims}(p4est, tree_node_coordinates,
                                nodes, boundary_names, mesh_file, false, true)

    elseif mesh_type == "T8codeMesh"
        ndims, ntrees, nelements, tree_node_coordinates,
        nodes, boundary_names_, treeIDs, neighIDs, faces, duals, orientations,
        levels, num_elements_per_tree = h5open(mesh_file, "r") do file
            return read(attributes(file)["ndims"]),
                   read(attributes(file)["ntrees"]),
                   read(attributes(file)["nelements"]),
                   read(file["tree_node_coordinates"]),
                   read(file["nodes"]),
                   read(file["boundary_names"]),
                   read(file["treeIDs"]),
                   read(file["neighIDs"]),
                   read(file["faces"]),
                   read(file["duals"]),
                   read(file["orientations"]),
                   read(file["levels"]),
                   read(file["num_elements_per_tree"])
        end

        boundary_names = boundary_names_ .|> Symbol

        mesh = T8codeMesh(ndims, ntrees, nelements, tree_node_coordinates,
                          nodes, boundary_names, treeIDs, neighIDs, faces,
                          duals, orientations, levels, num_elements_per_tree)

    elseif mesh_type == "DGMultiMesh"
        ndims, nelements, etype_str, EToV = h5open(mesh_file, "r") do file
            return read(attributes(file)["ndims"]),
                   read(attributes(file)["nelements"]),
                   read(attributes(file)["element_type"]),
                   read(file["EToV"])
        end

        # Load RefElemData.
        etype = get_element_type_from_string(etype_str)()

        polydeg = h5open(mesh_file, "r") do file
          if etype isa Wedge
              return tuple(read(attributes(file)["polydeg_tri"]),
                           read(attributes(file)["polydeg_line"]))
          else
              return read(attributes(file)["polydeg"])
          end
        end

        ## TODO: Is this useful to reconstruct `RefElemData`?
        ## # Load quadrature rule.
        ## rstq = h5open(mesh_file, "r") do file
        ##   # ASCII: Char(114) => 'r'
        ##   return tuple([read(file[(113 + i |> Char |> string) * "q"]) for i = 1:ndims]...)
        ## end
        ## rstwq = h5open(mesh_file, "r") do file
        ##   # ASCII: Char(114) => 'r'
        ##   return tuple(rstq..., read(file["wq"]))
        ## end

        # TODO: Make the following more general. But how? @jchan
        if etype isa StartUpDG.Wedge
          factor_a = RefElemData(StartUpDG.Tri(), Polynomial(), polydeg[1])
          factor_b = RefElemData(StartUpDG.Line(), Polynomial(), polydeg[2])

          tensor = StartUpDG.TensorProductWedge(factor_a, factor_b)
          rd = RefElemData(etype, tensor)
        else
          rd = RefElemData(etype, Polynomial(), polydeg)
        end

        # Load physical nodes.
        xyz = h5open(mesh_file, "r") do file
          # ASCII: Char(120) => 'x'
          return tuple([read(file[119 + i |> Char |> string]) for i = 1:ndims]...)
        end

        # Load element vertices.
        vxyz = h5open(mesh_file, "r") do file
          # ASCII: Char(58) => 'X'
          return tuple([read(file["V" * (87 + i |> Char |> string)]) for i = 1:ndims]...)
        end

        if ndims == 1
          md = MeshData(vxyz[1][1,:], EToV, rd)
        else
          # Load MeshData and restore original physical nodes.
          md = MeshData(rd, MeshData(vxyz, EToV, rd), xyz...)
       end

        mesh = DGMultiMesh(md, rd, [])
    else
        error("Unknown mesh type!")
    end

    return mesh
end

function load_mesh!(mesh::SerialTreeMesh, mesh_file::AbstractString)
    mesh.current_filename = mesh_file
    mesh.unsaved_changes = false

    # Read mesh file
    h5open(mesh_file, "r") do file
        # Set domain information
        mesh.tree.center_level_0 = read(attributes(file)["center_level_0"])
        mesh.tree.length_level_0 = read(attributes(file)["length_level_0"])
        mesh.tree.periodicity = Tuple(read(attributes(file)["periodicity"]))

        # Set length
        n_cells = read(attributes(file)["n_cells"])
        resize!(mesh.tree, n_cells)

        # Read in data
        mesh.tree.parent_ids[1:n_cells] = read(file["parent_ids"])
        mesh.tree.child_ids[:, 1:n_cells] = read(file["child_ids"])
        mesh.tree.neighbor_ids[:, 1:n_cells] = read(file["neighbor_ids"])
        mesh.tree.levels[1:n_cells] = read(file["levels"])
        mesh.tree.coordinates[:, 1:n_cells] = read(file["coordinates"])
    end

    return mesh
end

function load_mesh_parallel(mesh_file::AbstractString; n_cells_max, RealT)
    if mpi_isroot()
        ndims_, mesh_type = h5open(mesh_file, "r") do file
            return read(attributes(file)["ndims"]),
                   read(attributes(file)["mesh_type"])
        end
        MPI.Bcast!(Ref(ndims_), mpi_root(), mpi_comm())
        MPI.bcast(mesh_type, mpi_root(), mpi_comm())
    else
        ndims_ = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())[]
        mesh_type = MPI.bcast(nothing, mpi_root(), mpi_comm())
    end

    if mesh_type == "TreeMesh"
        if mpi_isroot()
            n_cells, capacity = h5open(mesh_file, "r") do file
                return read(attributes(file)["n_cells"]),
                       read(attributes(file)["capacity"])
            end
            MPI.Bcast!(Ref(n_cells), mpi_root(), mpi_comm())
            MPI.Bcast!(Ref(capacity), mpi_root(), mpi_comm())
        else
            n_cells = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())[]
            capacity = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())[]
        end

        mesh = TreeMesh(ParallelTree{ndims_, RealT},
                        max(n_cells, n_cells_max, capacity),
                        RealT = RealT)
        load_mesh!(mesh, mesh_file)
    elseif mesh_type == "P4estMesh"
        if mpi_isroot()
            p4est_filename, tree_node_coordinates,
            nodes, boundary_names_ = h5open(mesh_file, "r") do file
                return read(attributes(file)["p4est_file"]),
                       read(file["tree_node_coordinates"]),
                       read(file["nodes"]),
                       read(file["boundary_names"])
            end

            boundary_names = boundary_names_ .|> Symbol

            p4est_file = joinpath(dirname(mesh_file), p4est_filename)

            data = (p4est_file, tree_node_coordinates, nodes, boundary_names)
            MPI.bcast(data, mpi_root(), mpi_comm())
        else
            data = MPI.bcast(nothing, mpi_root(), mpi_comm())
            p4est_file, tree_node_coordinates, nodes, boundary_names = data
        end

        # Prevent Julia crashes when `p4est` can't find the file
        @assert isfile(p4est_file)

        p4est = load_p4est(p4est_file, Val(ndims_))

        mesh = P4estMesh{ndims_}(p4est, tree_node_coordinates,
                                 nodes, boundary_names, mesh_file, false, true)

    elseif mesh_type == "T8codeMesh"
        if mpi_isroot()
            ndims, ntrees, nelements, tree_node_coordinates, nodes,
            boundary_names_, treeIDs, neighIDs, faces, duals, orientations, levels,
            num_elements_per_tree = h5open(mesh_file, "r") do file
                return read(attributes(file)["ndims"]),
                       read(attributes(file)["ntrees"]),
                       read(attributes(file)["nelements"]),
                       read(file["tree_node_coordinates"]),
                       read(file["nodes"]),
                       read(file["boundary_names"]),
                       read(file["treeIDs"]),
                       read(file["neighIDs"]),
                       read(file["faces"]),
                       read(file["duals"]),
                       read(file["orientations"]),
                       read(file["levels"]),
                       read(file["num_elements_per_tree"])
            end

            boundary_names = boundary_names_ .|> Symbol

            data = (ndims, ntrees, nelements, tree_node_coordinates, nodes,
                    boundary_names, treeIDs, neighIDs, faces, duals,
                    orientations, levels, num_elements_per_tree)
            MPI.bcast(data, mpi_root(), mpi_comm())
        else
            data = MPI.bcast(nothing, mpi_root(), mpi_comm())
            ndims, ntrees, nelements, tree_node_coordinates, nodes,
            boundary_names, treeIDs, neighIDs, faces, duals, orientations, levels,
            num_elements_per_tree = data
        end

        mesh = T8codeMesh(ndims, ntrees, nelements, tree_node_coordinates,
                          nodes, boundary_names, treeIDs, neighIDs, faces,
                          duals, orientations, levels, num_elements_per_tree)
    else
        error("Unknown mesh type!")
    end

    return mesh
end

function load_mesh!(mesh::ParallelTreeMesh, mesh_file::AbstractString)
    mesh.current_filename = mesh_file
    mesh.unsaved_changes = false

    if mpi_isroot()
        h5open(mesh_file, "r") do file
            # Set domain information
            mesh.tree.center_level_0 = read(attributes(file)["center_level_0"])
            mesh.tree.length_level_0 = read(attributes(file)["length_level_0"])
            mesh.tree.periodicity = Tuple(read(attributes(file)["periodicity"]))
            MPI.Bcast!(collect(mesh.tree.center_level_0), mpi_root(), mpi_comm())
            MPI.Bcast!(collect(mesh.tree.length_level_0), mpi_root(), mpi_comm())
            MPI.Bcast!(collect(mesh.tree.periodicity), mpi_root(), mpi_comm())

            # Set length
            n_cells = read(attributes(file)["n_cells"])
            MPI.Bcast!(Ref(n_cells), mpi_root(), mpi_comm())
            resize!(mesh.tree, n_cells)

            # Read in data
            mesh.tree.parent_ids[1:n_cells] = read(file["parent_ids"])
            mesh.tree.child_ids[:, 1:n_cells] = read(file["child_ids"])
            mesh.tree.neighbor_ids[:, 1:n_cells] = read(file["neighbor_ids"])
            mesh.tree.levels[1:n_cells] = read(file["levels"])
            mesh.tree.coordinates[:, 1:n_cells] = read(file["coordinates"])
            @views MPI.Bcast!(mesh.tree.parent_ids[1:n_cells], mpi_root(), mpi_comm())
            @views MPI.Bcast!(mesh.tree.child_ids[:, 1:n_cells], mpi_root(), mpi_comm())
            @views MPI.Bcast!(mesh.tree.neighbor_ids[:, 1:n_cells], mpi_root(),
                              mpi_comm())
            @views MPI.Bcast!(mesh.tree.levels[1:n_cells], mpi_root(), mpi_comm())
            @views MPI.Bcast!(mesh.tree.coordinates[:, 1:n_cells], mpi_root(),
                              mpi_comm())
        end
    else # non-root ranks
        # Set domain information
        mesh.tree.center_level_0 = MPI.Bcast!(collect(mesh.tree.center_level_0),
                                              mpi_root(), mpi_comm())
        mesh.tree.length_level_0 = MPI.Bcast!(collect(mesh.tree.length_level_0),
                                              mpi_root(), mpi_comm())[1]
        mesh.tree.periodicity = Tuple(MPI.Bcast!(collect(mesh.tree.periodicity),
                                                 mpi_root(), mpi_comm()))

        # Set length
        n_cells = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())[]
        resize!(mesh.tree, n_cells)

        # Read in data
        @views MPI.Bcast!(mesh.tree.parent_ids[1:n_cells], mpi_root(), mpi_comm())
        @views MPI.Bcast!(mesh.tree.child_ids[:, 1:n_cells], mpi_root(), mpi_comm())
        @views MPI.Bcast!(mesh.tree.neighbor_ids[:, 1:n_cells], mpi_root(), mpi_comm())
        @views MPI.Bcast!(mesh.tree.levels[1:n_cells], mpi_root(), mpi_comm())
        @views MPI.Bcast!(mesh.tree.coordinates[:, 1:n_cells], mpi_root(), mpi_comm())
    end

    # Partition mesh
    partition!(mesh)

    return mesh
end
end # @muladd
