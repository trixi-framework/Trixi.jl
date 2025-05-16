# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function save_restart_file(u, time, dt, timestep,
                           mesh::Union{SerialTreeMesh, StructuredMesh,
                                       UnstructuredMesh2D, SerialP4estMesh,
                                       SerialT8codeMesh},
                           equations, dg::DG, cache,
                           restart_callback)
    @unpack output_directory = restart_callback

    # Filename based on current time step
    filename = joinpath(output_directory, @sprintf("restart_%09d.h5", timestep))

    # Restart files always store conservative variables
    data = u

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = nvariables(equations)
        attributes(file)["n_elements"] = nelements(dg, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution
        for v in eachvariable(equations)
            # Convert to 1D array
            file["variables_$v"] = vec(data[v, .., :])

            # Add variable name as attribute
            var = file["variables_$v"]
            attributes(var)["name"] = varnames(cons2cons, equations)[v]
        end
    end

    return filename
end

# Load solution variables from restart file and 
# convert them to a different polynomial degree.
# Version for non-MPI-parallel run
function convert_restart_file_polydeg!(u, file,
                                       mesh, equations, dg::DGSEM, cache,
                                       nnodes_file, conversion_matrix)
    all_variables = zeros(eltype(u),
                          (nvariables(equations),
                           ntuple(_ -> nnodes_file, ndims(mesh))...,
                           nelements(dg, cache)))
    for v in eachvariable(equations)
        all_variables[v, .., :] = read(file["variables_$v"])
    end

    # Perform interpolation/projection to new polynomial degree
    for element in eachelement(dg, cache)
        u[.., element] = multiply_dimensionwise(conversion_matrix,
                                                all_variables[.., element])
    end
end

# Version for MPI-parallel run with serial I/O, i.e., `HDF5.has_parallel() == false`
function convert_restart_file_polydeg!(u_global, file,
                                       mesh::Union{ParallelTreeMesh, ParallelP4estMesh,
                                                   ParallelT8codeMesh}, equations,
                                       dg::DGSEM, cache,
                                       nnodes_file, conversion_matrix)
    n_elements_global = nelementsglobal(mesh, dg, cache)
    all_variables_global = zeros(eltype(u_global),
                                 (nvariables(equations),
                                  ntuple(_ -> nnodes_file, ndims(mesh))...,
                                  n_elements_global))
    for v in eachvariable(equations)
        all_variables_global[v, .., :] = read(file["variables_$v"])
    end

    # Perform interpolation/projection to new polynomial degree
    for element in 1:n_elements_global
        u_global[.., element] = multiply_dimensionwise(conversion_matrix,
                                                       all_variables_global[..,
                                                                            element])
    end
end

# Version for MPI-parallel run with MPI-parallel I/O, i.e., `HDF5.has_parallel() == true`
function convert_restart_file_polydeg!(u, file, slice,
                                       mesh, equations, dg::DGSEM, cache,
                                       nnodes_file, conversion_matrix)
    all_variables = zeros(eltype(u),
                          (nvariables(equations),
                           ntuple(_ -> nnodes_file, ndims(mesh))...,
                           nelements(dg, cache)))
    for v in eachvariable(equations)
        var = file["variables_$v"]
        # Read in only data slice of this process
        all_variables[v, .., :] = reshape(read(var)[slice],
                                          size(@view all_variables[v, .., :]))
    end

    # Perform interpolation/projection to new polynomial degree
    for element in eachelement(dg, cache)
        u[.., element] = multiply_dimensionwise(conversion_matrix,
                                                all_variables[.., element])
    end
end

function load_restart_file(mesh::Union{SerialTreeMesh, StructuredMesh,
                                       UnstructuredMesh2D, SerialP4estMesh,
                                       SerialT8codeMesh},
                           equations, dg::DG, cache,
                           restart_file, interpolate_high2low)

    # allocate memory
    u_ode = allocate_coefficients(mesh, equations, dg, cache)
    u = wrap_array_native(u_ode, mesh, equations, dg, cache)

    h5open(restart_file, "r") do file
        # Read attributes to perform some sanity checks
        if read(attributes(file)["ndims"]) != ndims(mesh)
            error("restart mismatch: ndims differs from value in restart file")
        end
        if read(attributes(file)["equations"]) != get_name(equations)
            error("restart mismatch: equations differ from value in restart file")
        end
        if read(attributes(file)["n_elements"]) != nelements(dg, cache)
            error("restart mismatch: number of elements in solver differs from value in restart file")
        end
        for v in eachvariable(equations)
            # Check if variable name matches
            var = file["variables_$v"]
            if (name = read(attributes(var)["name"])) !=
               varnames(cons2cons, equations)[v]
                error("mismatch: variables_$v should be '$(varnames(cons2cons, equations)[v])', but found '$name'")
            end
        end

        ### Read variable data ###
        polydeg_file = read(attributes(file)["polydeg"])
        if polydeg_file != polydeg(dg) # Conversion is necessary
            nnodes_file = polydeg_file + 1
            nodes_file = gauss_lobatto_nodes_weights(nnodes_file)[1]

            nodes_solver = gauss_lobatto_nodes_weights(nnodes(dg))[1]

            if polydeg_file < polydeg(dg) || interpolate_high2low # Interpolation from lower to higher
                conversion_matrix = polynomial_interpolation_matrix(nodes_file,
                                                                    nodes_solver)
            else # Projection from higher to lower
                conversion_matrix = polynomial_l2projection_matrix(nodes_file,
                                                                   nodes_solver,
                                                                   Val(:gauss_lobatto))
            end
            convert_restart_file_polydeg!(u, file, mesh, equations, dg, cache,
                                          nnodes_file, conversion_matrix)
        else # Read in variables separately
            for v in eachvariable(equations)
                u[v, .., :] = read(file["variables_$v"])
            end
        end
    end

    return u_ode
end

function save_restart_file(u, time, dt, timestep,
                           mesh::Union{ParallelTreeMesh, ParallelP4estMesh,
                                       ParallelT8codeMesh}, equations,
                           dg::DG, cache,
                           restart_callback)
    @unpack output_directory = restart_callback
    # Filename based on current time step
    filename = joinpath(output_directory, @sprintf("restart_%09d.h5", timestep))

    if HDF5.has_parallel()
        save_restart_file_parallel(u, time, dt, timestep, mesh, equations, dg, cache,
                                   filename)
    else
        save_restart_file_on_root(u, time, dt, timestep, mesh, equations, dg, cache,
                                  filename)
    end
end

function save_restart_file_parallel(u, time, dt, timestep,
                                    mesh::Union{ParallelTreeMesh, ParallelP4estMesh,
                                                ParallelT8codeMesh},
                                    equations, dg::DG, cache,
                                    filename)

    # Restart files always store conservative variables
    data = u

    # Calculate element and node counts by MPI rank
    element_size = nnodes(dg)^ndims(mesh)
    element_counts = convert(Vector{Cint}, collect(cache.mpi_cache.n_elements_by_rank))
    node_counts = element_counts * Cint(element_size)
    # Cumulative sum of nodes per rank starting with an additional 0
    cum_node_counts = append!(zeros(eltype(node_counts), 1), cumsum(node_counts))

    # Open file (clobber existing content)
    h5open(filename, "w", mpi_comm()) do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = nvariables(equations)
        attributes(file)["n_elements"] = nelementsglobal(mesh, dg, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution
        for v in eachvariable(equations)
            # Need to create dataset explicitly in parallel case
            var = create_dataset(file, "/variables_$v", datatype(eltype(data)),
                                 dataspace((ndofsglobal(mesh, dg, cache),)))
            # Write data of each process in slices (ranks start with 0)
            slice = (cum_node_counts[mpi_rank() + 1] + 1):cum_node_counts[mpi_rank() + 2]
            # Convert to 1D array
            var[slice] = vec(data[v, .., :])
            # Add variable name as attribute
            attributes(var)["name"] = varnames(cons2cons, equations)[v]
        end
    end

    return filename
end

function save_restart_file_on_root(u, time, dt, timestep,
                                   mesh::Union{ParallelTreeMesh, ParallelP4estMesh,
                                               ParallelT8codeMesh},
                                   equations, dg::DG, cache,
                                   filename)

    # Restart files always store conservative variables
    data = u

    # Calculate element and node counts by MPI rank
    element_size = nnodes(dg)^ndims(mesh)
    element_counts = convert(Vector{Cint}, collect(cache.mpi_cache.n_elements_by_rank))
    node_counts = element_counts * Cint(element_size)

    # non-root ranks only send data
    if !mpi_isroot()
        # Send nodal data to root
        for v in eachvariable(equations)
            MPI.Gatherv!(vec(data[v, .., :]), nothing, mpi_root(), mpi_comm())
        end

        return filename
    end

    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = nvariables(equations)
        attributes(file)["n_elements"] = nelements(dg, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution
        for v in eachvariable(equations)
            # Convert to 1D array
            recv = Vector{eltype(data)}(undef, sum(node_counts))
            MPI.Gatherv!(vec(data[v, .., :]), MPI.VBuffer(recv, node_counts),
                         mpi_root(), mpi_comm())
            file["variables_$v"] = recv

            # Add variable name as attribute
            var = file["variables_$v"]
            attributes(var)["name"] = varnames(cons2cons, equations)[v]
        end
    end

    return filename
end

function load_restart_file(mesh::Union{ParallelTreeMesh, ParallelP4estMesh,
                                       ParallelT8codeMesh}, equations,
                           dg::DG, cache, restart_file, interpolate_high2low)
    if HDF5.has_parallel()
        load_restart_file_parallel(mesh, equations, dg, cache,
                                   restart_file, interpolate_high2low)
    else
        load_restart_file_on_root(mesh, equations, dg, cache,
                                  restart_file, interpolate_high2low)
    end
end

function load_restart_file_parallel(mesh::Union{ParallelTreeMesh, ParallelP4estMesh,
                                                ParallelT8codeMesh},
                                    equations, dg::DG, cache,
                                    restart_file, interpolate_high2low)
    # allocate memory
    u_ode = allocate_coefficients(mesh, equations, dg, cache)
    u = wrap_array_native(u_ode, mesh, equations, dg, cache)

    # read in parallel
    h5open(restart_file, "r", mpi_comm()) do file
        # Read attributes to perform some sanity checks
        if read(attributes(file)["ndims"]) != ndims(mesh)
            error("restart mismatch: ndims differs from value in restart file")
        end
        if read(attributes(file)["equations"]) != get_name(equations)
            error("restart mismatch: equations differ from value in restart file")
        end
        if read(attributes(file)["n_elements"]) != nelementsglobal(mesh, dg, cache)
            error("restart mismatch: number of elements in solver differs from value in restart file")
        end
        for v in eachvariable(equations)
            # Check if variable name matches
            var = file["variables_$v"]
            if (name = read(attributes(var)["name"])) !=
               varnames(cons2cons, equations)[v]
                error("mismatch: variables_$v should be '$(varnames(cons2cons, equations)[v])', but found '$name'")
            end
        end

        ### Read variable data ###
        element_counts = convert(Vector{Cint},
                                 collect(cache.mpi_cache.n_elements_by_rank))
        polydeg_file = read(attributes(file)["polydeg"])
        if polydeg_file != polydeg(dg) # Conversion is necessary
            nnodes_file = polydeg_file + 1
            nodes_file = gauss_lobatto_nodes_weights(nnodes_file)[1]

            nodes_solver = gauss_lobatto_nodes_weights(nnodes(dg))[1]
            if polydeg_file < polydeg(dg) || interpolate_high2low # Interpolation from lower to higher
                conversion_matrix = polynomial_interpolation_matrix(nodes_file,
                                                                    nodes_solver)
            else # Projection from higher to lower
                conversion_matrix = polynomial_l2projection_matrix(nodes_file,
                                                                   nodes_solver,
                                                                   Val(:gauss_lobatto))
            end

            # Calculate element and node counts by MPI rank                                                                   
            element_size_file = nnodes_file^ndims(mesh)
            node_counts_file = element_counts * Cint(element_size_file)
            # Cumulative sum of nodes per rank starting with an additional 0
            cum_node_counts_file = append!(zeros(eltype(node_counts_file), 1),
                                           cumsum(node_counts_file))

            # Read data of each process in slices (ranks start with 0)
            slice = (cum_node_counts_file[mpi_rank() + 1] + 1):cum_node_counts_file[mpi_rank() + 2]

            convert_restart_file_polydeg!(u, file, slice, mesh, equations, dg, cache,
                                          nnodes_file, conversion_matrix)
        else # Read in variables separately
            # Calculate element and node counts by MPI rank
            element_size = nnodes(dg)^ndims(mesh)
            node_counts = element_counts * Cint(element_size)
            # Cumulative sum of nodes per rank starting with an additional 0
            cum_node_counts = append!(zeros(eltype(node_counts), 1),
                                      cumsum(node_counts))
            # Read data of each process in slices (ranks start with 0)
            slice = (cum_node_counts[mpi_rank() + 1] + 1):cum_node_counts[mpi_rank() + 2]

            for v in eachvariable(equations)
                var = file["variables_$v"]
                # Convert 1D array back to actual size of `u`
                u[v, .., :] = reshape(read(var)[slice], size(@view u[v, .., :]))
            end
        end
    end

    return u_ode
end

function load_restart_file_on_root(mesh::Union{ParallelTreeMesh, ParallelP4estMesh,
                                               ParallelT8codeMesh},
                                   equations, dg::DG, cache,
                                   restart_file, interpolate_high2low)

    # Calculate element and node counts by MPI rank
    element_size = nnodes(dg)^ndims(mesh)
    element_counts = convert(Vector{Cint}, collect(cache.mpi_cache.n_elements_by_rank))
    node_counts = element_counts * Cint(element_size)

    # allocate memory
    u_ode = allocate_coefficients(mesh, equations, dg, cache)
    u = wrap_array_native(u_ode, mesh, equations, dg, cache)

    # non-root ranks only receive data
    if !mpi_isroot()
        # Receive nodal data from root
        for v in eachvariable(equations)
            # put Scatterv in both blocks of the if condition to avoid type instability
            if isempty(u)
                data = eltype(u)[]
                MPI.Scatterv!(nothing, data, mpi_root(), mpi_comm())
            else
                data = @view u[v, .., :]
                MPI.Scatterv!(nothing, data, mpi_root(), mpi_comm())
            end
        end

        return u_ode
    end

    # read only on MPI root
    h5open(restart_file, "r") do file
        # Read attributes to perform some sanity checks
        if read(attributes(file)["ndims"]) != ndims(mesh)
            error("restart mismatch: ndims differs from value in restart file")
        end
        if read(attributes(file)["equations"]) != get_name(equations)
            error("restart mismatch: equations differ from value in restart file")
        end
        if read(attributes(file)["n_elements"]) != nelements(dg, cache)
            error("restart mismatch: number of elements in solver differs from value in restart file")
        end

        ### Read variable data ###
        polydeg_file = read(attributes(file)["polydeg"])
        if polydeg_file != polydeg(dg) # Conversion is necessary
            nnodes_file = polydeg_file + 1
            nodes_file = gauss_lobatto_nodes_weights(nnodes_file)[1]

            nodes_solver = gauss_lobatto_nodes_weights(nnodes(dg))[1]
            if polydeg_file < polydeg(dg) || interpolate_high2low # Interpolation from lower to higher
                conversion_matrix = polynomial_interpolation_matrix(nodes_file,
                                                                    nodes_solver)
            else # Projection from higher to lower
                conversion_matrix = polynomial_l2projection_matrix(nodes_file,
                                                                   nodes_solver,
                                                                   Val(:gauss_lobatto))
            end

            # We perform the interpolation of all elements on the root rank.
            # Thus, we need the allocate the global array
            u_global = zeros(eltype(u),
                             (nvariables(equations),
                              ntuple(_ -> nnodes(dg), ndims(mesh))...,
                              nelementsglobal(mesh, dg, cache)))

            convert_restart_file_polydeg!(u_global, file, mesh, equations, dg, cache,
                                          nnodes_file, conversion_matrix)
            for v in eachvariable(equations)
                sendbuf = MPI.VBuffer(@view(u_global[v, .., :]), node_counts)
                MPI.Scatterv!(sendbuf, @view(u[v, .., :]), mpi_root(), mpi_comm())
            end
        else # Read in variables separately
            for v in eachvariable(equations)
                # Check if variable name matches
                var = file["variables_$v"]
                if (name = read(attributes(var)["name"])) !=
                   varnames(cons2cons, equations)[v]
                    error("mismatch: variables_$v should be '$(varnames(cons2cons, equations)[v])', but found '$name'")
                end

                sendbuf = MPI.VBuffer(read(file["variables_$v"]), node_counts)
                MPI.Scatterv!(sendbuf, @view(u[v, .., :]), mpi_root(), mpi_comm())
            end
        end
    end

    return u_ode
end

# Store controller values for an adaptive time stepping scheme
function save_adaptive_time_integrator(integrator,
                                       controller, restart_callback)
    # Save only on root
    if mpi_isroot()
        @unpack output_directory = restart_callback
        timestep = integrator.stats.naccept

        # Filename based on current time step
        filename = joinpath(output_directory, @sprintf("restart_%09d.h5", timestep))

        # Open file (preserve existing content)
        h5open(filename, "r+") do file
            # Add context information as attributes both for PIController and PIDController
            attributes(file)["time_integrator_qold"] = integrator.qold
            attributes(file)["time_integrator_dtpropose"] = integrator.dtpropose
            # For PIDController is necessary to save additional parameters
            if hasproperty(controller, :err) # Distinguish PIDController from PIController
                attributes(file)["time_integrator_controller_err"] = controller.err
            end
        end
    end
end
end # @muladd
