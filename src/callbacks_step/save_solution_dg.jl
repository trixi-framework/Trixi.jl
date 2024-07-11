# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function save_solution_file(u, time, dt, timestep,
                            mesh::Union{SerialTreeMesh, StructuredMesh,
                                        StructuredMeshView,
                                        UnstructuredMesh2D, SerialP4estMesh,
                                        SerialT8codeMesh},
                            equations, dg::DG, cache,
                            solution_callback,
                            element_variables = Dict{Symbol, Any}(),
                            node_variables = Dict{Symbol, Any}();
                            system = "")
    @unpack output_directory, solution_variables = solution_callback

    # Filename based on current time step
    if isempty(system)
        filename = joinpath(output_directory, @sprintf("solution_%09d.h5", timestep))
    else
        filename = joinpath(output_directory,
                            @sprintf("solution_%s_%09d.h5", system, timestep))
    end

    # Convert to different set of variables if requested
    if solution_variables === cons2cons
        data = u
        n_vars = nvariables(equations)
    else
        # Reinterpret the solution array as an array of conservative variables,
        # compute the solution variables via broadcasting, and reinterpret the
        # result as a plain array of floating point numbers
        data = Array(reinterpret(eltype(u),
                                 solution_variables.(reinterpret(SVector{nvariables(equations),
                                                                         eltype(u)}, u),
                                                     Ref(equations))))

        # Find out variable count by looking at output from `solution_variables` function
        n_vars = size(data, 1)
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nelements(dg, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution data
        for v in 1:n_vars
            # Convert to 1D array
            file["variables_$v"] = vec(data[v, .., :])

            # Add variable name as attribute
            var = file["variables_$v"]
            attributes(var)["name"] = varnames(solution_variables, equations)[v]
        end

        # Store element variables
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Add to file
            file["element_variables_$v"] = element_variable

            # Add variable name as attribute
            var = file["element_variables_$v"]
            attributes(var)["name"] = string(key)
        end

        # Store node variables
        for (v, (key, node_variable)) in enumerate(node_variables)
            # Add to file
            file["node_variables_$v"] = node_variable

            # Add variable name as attribute
            var = file["node_variables_$v"]
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end

function save_solution_file(u, time, dt, timestep,
                            mesh::Union{ParallelTreeMesh, ParallelP4estMesh}, equations,
                            dg::DG, cache,
                            solution_callback,
                            element_variables = Dict{Symbol, Any}(),
                            node_variables = Dict{Symbol, Any}();
                            system = "")
    @unpack output_directory, solution_variables = solution_callback

    # Filename based on current time step
    if isempty(system)
        filename = joinpath(output_directory, @sprintf("solution_%09d.h5", timestep))
    else
        filename = joinpath(output_directory,
                            @sprintf("solution_%s_%09d.h5", system, timestep))
    end

    # Convert to different set of variables if requested
    if solution_variables === cons2cons
        data = u
        n_vars = nvariables(equations)
    else
        # Reinterpret the solution array as an array of conservative variables,
        # compute the solution variables via broadcasting, and reinterpret the
        # result as a plain array of floating point numbers
        data = Array(reinterpret(eltype(u),
                                 solution_variables.(reinterpret(SVector{nvariables(equations),
                                                                         eltype(u)}, u),
                                                     Ref(equations))))

        # Find out variable count by looking at output from `solution_variables` function
        n_vars = size(data, 1)
    end

    if HDF5.has_parallel()
        save_solution_file_parallel(data, time, dt, timestep, n_vars, mesh, equations,
                                    dg, cache, solution_variables, filename,
                                    element_variables)
    else
        save_solution_file_on_root(data, time, dt, timestep, n_vars, mesh, equations,
                                   dg, cache, solution_variables, filename,
                                   element_variables)
    end
end

function save_solution_file_parallel(data, time, dt, timestep, n_vars,
                                     mesh::Union{ParallelTreeMesh, ParallelP4estMesh},
                                     equations, dg::DG, cache,
                                     solution_variables, filename,
                                     element_variables = Dict{Symbol, Any}())

    # Calculate element and node counts by MPI rank
    element_size = nnodes(dg)^ndims(mesh)
    element_counts = cache.mpi_cache.n_elements_by_rank
    node_counts = element_counts * element_size
    # Cumulative sum of elements per rank starting with an additional 0
    cum_element_counts = append!(zeros(eltype(element_counts), 1),
                                 cumsum(element_counts))
    # Cumulative sum of nodes per rank starting with an additional 0
    cum_node_counts = append!(zeros(eltype(node_counts), 1), cumsum(node_counts))

    # Open file using parallel HDF5 (clobber existing content)
    h5open(filename, "w", mpi_comm()) do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nelementsglobal(mesh, dg, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution data
        for v in 1:n_vars
            # Need to create dataset explicitly in parallel case
            var = create_dataset(file, "/variables_$v", datatype(eltype(data)),
                                 dataspace((ndofsglobal(mesh, dg, cache),)))
            # Write data of each process in slices (ranks start with 0)
            slice = (cum_node_counts[mpi_rank() + 1] + 1):cum_node_counts[mpi_rank() + 2]
            # Convert to 1D array
            var[slice] = vec(data[v, .., :])
            # Add variable name as attribute
            attributes(var)["name"] = varnames(solution_variables, equations)[v]
        end

        # Store element variables
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Need to create dataset explicitly in parallel case
            var = create_dataset(file, "/element_variables_$v",
                                 datatype(eltype(element_variable)),
                                 dataspace((nelementsglobal(mesh, dg, cache),)))

            # Write data of each process in slices (ranks start with 0)
            slice = (cum_element_counts[mpi_rank() + 1] + 1):cum_element_counts[mpi_rank() + 2]
            # Add to file
            var[slice] = element_variable
            # Add variable name as attribute
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end

function save_solution_file_on_root(data, time, dt, timestep, n_vars,
                                    mesh::Union{ParallelTreeMesh, ParallelP4estMesh},
                                    equations, dg::DG, cache,
                                    solution_variables, filename,
                                    element_variables = Dict{Symbol, Any}())

    # Calculate element and node counts by MPI rank
    element_size = nnodes(dg)^ndims(mesh)
    element_counts = convert(Vector{Cint}, collect(cache.mpi_cache.n_elements_by_rank))
    node_counts = element_counts * Cint(element_size)

    # non-root ranks only send data
    if !mpi_isroot()
        # Send nodal data to root
        for v in 1:n_vars
            MPI.Gatherv!(vec(data[v, .., :]), nothing, mpi_root(), mpi_comm())
        end

        # Send element data to root
        for (key, element_variable) in element_variables
            MPI.Gatherv!(element_variable, nothing, mpi_root(), mpi_comm())
        end

        return filename
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = n_vars
        attributes(file)["n_elements"] = nelementsglobal(mesh, dg, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution data
        for v in 1:n_vars
            # Convert to 1D array
            recv = Vector{eltype(data)}(undef, sum(node_counts))
            MPI.Gatherv!(vec(data[v, .., :]), MPI.VBuffer(recv, node_counts),
                         mpi_root(), mpi_comm())
            file["variables_$v"] = recv

            # Add variable name as attribute
            var = file["variables_$v"]
            attributes(var)["name"] = varnames(solution_variables, equations)[v]
        end

        # Store element variables
        for (v, (key, element_variable)) in enumerate(element_variables)
            # Add to file
            recv = Vector{eltype(data)}(undef, sum(element_counts))
            MPI.Gatherv!(element_variable, MPI.VBuffer(recv, element_counts),
                         mpi_root(), mpi_comm())
            file["element_variables_$v"] = recv

            # Add variable name as attribute
            var = file["element_variables_$v"]
            attributes(var)["name"] = string(key)
        end
    end

    return filename
end
end # @muladd
