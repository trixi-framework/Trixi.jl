# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function save_restart_file(u, time, dt, timestep,
                           mesh::Union{TreeMeshSerial, StructuredMesh,
                                       UnstructuredMesh2D, P4estMeshSerial,
                                       T8codeMeshSerial},
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

# Compute the matrix converting the nodal solution values of a restart file with
# polynomial degree `polydeg_file` to the polynomial degree of the solver `dg`.
# Low -> high: interpolation (exact). High -> low: interpolation if
# `interpolate_high2low == true`, otherwise L2 projection.
# The L2 projection preserves the integral of `u` over each element in reference
# coordinates. Thus, it conserves the discrete mass `sum_{i,j} w_i w_j J_{ij} u_{ij}`,
# i.e., the quantity conserved by the DGSEM, only on meshes with an element-wise
# constant Jacobian (such as the `TreeMesh`).
# On curvilinear meshes, the Jacobian varies within an element and
# the LGL quadrature is not exact for `J * u`, so conservation is
# not guaranteed without special treatment, see `correct_mass_restart!`.
# Restart files do not store the type of the basis, i.e., the nodes of the stored
# solution are not known. Thus, the conversion is only supported for the
# Legendre-Gauss-Lobatto nodes of the `LobattoLegendreBasis`, which are assumed for
# both the restart file and the solver.
function restart_conversion_matrix(polydeg_file, dg::DG{<:LobattoLegendreBasis},
                                   interpolate_high2low)
    RealT = real(dg)
    nodes_file = gauss_lobatto_nodes_weights(polydeg_file + 1, RealT)[1]
    nodes_solver = gauss_lobatto_nodes_weights(nnodes(dg), RealT)[1]

    if (polydeg_file < polydeg(dg)) || interpolate_high2low # Interpolation
        return polynomial_interpolation_matrix(nodes_file, nodes_solver)
    else # Projection from higher to lower polynomial degree
        return polynomial_l2projection_matrix(nodes_file, nodes_solver, Val(:gauss),
                                              RealT)
    end
end

function restart_conversion_matrix(polydeg_file, dg::DG, interpolate_high2low)
    throw(ArgumentError("restart mismatch: the polynomial degree in the restart file " *
                        "($polydeg_file) differs from the one of the solver " *
                        "($(polydeg(dg))). Converting the solution to a different " *
                        "polynomial degree is only supported for a `DGSEM` solver " *
                        "with a `LobattoLegendreBasis`, but the solver uses a " *
                        "$(nameof(typeof(dg.basis)))."))
end

# Inverse Jacobian at the LGL nodes of the polynomial degree `polydeg_file` of the
# restart file, i.e., the inverse Jacobian used by the simulation that wrote it.
# On the `TreeMesh`, the Jacobian is constant per element and thus independent of the
# polynomial degree.
function inverse_jacobian_restart_file(mesh::TreeMesh, equations, dg, cache,
                                       polydeg_file)
    return cache.elements.inverse_jacobian
end

function inverse_jacobian_restart_file(mesh::Union{StructuredMesh, P4estMesh,
                                                   T8codeMesh},
                                       equations, dg, cache, polydeg_file)
    basis_file = LobattoLegendreBasis(real(dg), polydeg_file)
    elements_file = init_elements(mesh, equations, basis_file, real(dg))
    return elements_file.inverse_jacobian
end

function inverse_jacobian_restart_file(mesh::UnstructuredMesh2D, equations, dg, cache,
                                       polydeg_file)
    basis_file = LobattoLegendreBasis(real(dg), polydeg_file)
    elements_file = init_elements(mesh, equations, basis_file, real(dg), real(dg))
    return elements_file.inverse_jacobian
end

# Compute the discrete mass `sum_{i,j,(k)} w_i w_j (w_k) J_{ij(k)} u_{ij(k)}` of all
# variables and the discrete volume `sum_{i,j,(k)} w_i w_j (w_k) J_{ij(k)}` of an
# element. The discrete mass is the quantity conserved by the DGSEM.
# Dimension agnostic, i.e., valid for 1D, 2D, and 3D meshes. Since this is called only
# once when loading a restart file, it is not optimized for performance.
function element_mass_and_volume_restart(u, element, weights, inverse_jacobian, mesh)
    n_variables = size(u, 1)
    mass = zeros(eltype(u), n_variables)
    volume = zero(eltype(inverse_jacobian))
    for node in CartesianIndices(ntuple(_ -> length(weights), ndims(mesh)))
        indices = Tuple(node)
        volume_node = prod(i -> weights[i], indices) /
                      get_inverse_jacobian(inverse_jacobian, mesh, indices..., element)
        for v in 1:n_variables
            mass[v] += volume_node * u[v, indices..., element]
        end
        volume += volume_node
    end
    return mass, volume
end

# Correct the discrete mass of the converted solution `u` such that it agrees with the
# discrete mass of the solution `u_file` of the restart file in each element.
# The discrete mass `sum w J u` is in general not preserved by the conversion on
# curvilinear meshes, since the LGL quadrature is not exact for `J * u` and the Jacobian
# is evaluated at different nodes before and after the conversion. On all meshes, it is
# not preserved by the interpolation from a higher to a lower polynomial degree.
# To restore exact conservation, the same constant
# `delta = (mass_file - mass_new) / volume_new` is added to all nodal values of each
# element. If the discrete volumes `sum w J` of an element before and after the conversion
# differ (e.g., for curved elements whose Jacobian is not integrated exactly by the
# LGL quadrature), `delta` does not vanish for constant states. Thus, the correction
# trades exact free-stream preservation for exact conservation.
function correct_mass_restart!(u, u_file, polydeg_file,
                               mesh, equations, dg::DGSEM, cache)
    weights_file = gauss_lobatto_nodes_weights(polydeg_file + 1, real(dg))[2]
    inverse_jacobian_file = inverse_jacobian_restart_file(mesh, equations, dg, cache,
                                                          polydeg_file)
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    for element in eachelement(dg, cache)
        mass_file, _ = element_mass_and_volume_restart(u_file, element, weights_file,
                                                       inverse_jacobian_file, mesh)
        mass, volume = element_mass_and_volume_restart(u, element, weights,
                                                       inverse_jacobian, mesh)
        delta = (mass_file - mass) / volume
        for node in CartesianIndices(ntuple(_ -> nnodes(dg), ndims(mesh)))
            for v in eachvariable(equations)
                u[v, Tuple(node)..., element] += delta[v]
            end
        end
    end

    return nothing
end

# Convert the nodal values `u_file` [variable, i, (j, (k),) element] of the solution
# stored in a restart file with polynomial degree `polydeg_file` to the polynomial
# degree of the solver and store them in `u`. Both arrays contain only the elements
# of this process. Optionally, the discrete mass of each element is corrected
# afterwards, see `correct_mass_restart!`.
function convert_restart_file_polydeg!(u, u_file, polydeg_file,
                                       mesh, equations, dg::DGSEM, cache,
                                       interpolate_high2low, correct_mass)
    conversion_matrix = restart_conversion_matrix(polydeg_file, dg,
                                                  interpolate_high2low)

    # Perform interpolation/projection to new polynomial degree
    for element in eachelement(dg, cache)
        u[.., element] = multiply_dimensionwise(conversion_matrix,
                                                u_file[.., element])
    end

    if correct_mass
        correct_mass_restart!(u, u_file, polydeg_file, mesh, equations, dg, cache)
    end

    return nothing
end

# Allocate an array for the nodal values of the solution with `nnodes_file` nodes per
# dimension stored in a restart file for the elements of this process
function allocate_restart_file_data(u, nnodes_file, mesh, equations, dg, cache)
    return zeros(eltype(u),
                 (nvariables(equations),
                  ntuple(_ -> nnodes_file, ndims(mesh))...,
                  nelements(dg, cache)))
end

function load_restart_file(mesh::Union{TreeMeshSerial, StructuredMesh,
                                       UnstructuredMesh2D, P4estMeshSerial,
                                       T8codeMeshSerial},
                           equations, dg::DG, cache,
                           restart_file, interpolate_high2low, correct_mass)

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
            u_file = allocate_restart_file_data(u, polydeg_file + 1,
                                                mesh, equations, dg, cache)
            for v in eachvariable(equations)
                u_file[v, .., :] = read(file["variables_$v"])
            end
            convert_restart_file_polydeg!(u, u_file, polydeg_file,
                                          mesh, equations, dg, cache,
                                          interpolate_high2low, correct_mass)
        else # Read in variables separately
            for v in eachvariable(equations)
                u[v, .., :] = read(file["variables_$v"])
            end
        end
    end

    return u_ode
end

function save_restart_file(u, time, dt, timestep,
                           mesh::Union{TreeMeshParallel, P4estMeshParallel,
                                       T8codeMeshParallel}, equations,
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
                                    mesh::Union{TreeMeshParallel, P4estMeshParallel,
                                                T8codeMeshParallel},
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
                                   mesh::Union{TreeMeshParallel, P4estMeshParallel,
                                               T8codeMeshParallel},
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

function load_restart_file(mesh::Union{TreeMeshParallel, P4estMeshParallel,
                                       T8codeMeshParallel}, equations,
                           dg::DG, cache, restart_file, interpolate_high2low,
                           correct_mass)
    if HDF5.has_parallel()
        load_restart_file_parallel(mesh, equations, dg, cache,
                                   restart_file, interpolate_high2low, correct_mass)
    else
        load_restart_file_on_root(mesh, equations, dg, cache,
                                  restart_file, interpolate_high2low, correct_mass)
    end
end

function load_restart_file_parallel(mesh::Union{TreeMeshParallel, P4estMeshParallel,
                                                T8codeMeshParallel},
                                    equations, dg::DG, cache,
                                    restart_file, interpolate_high2low, correct_mass)
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

            # Calculate element and node counts by MPI rank
            element_size_file = nnodes_file^ndims(mesh)
            node_counts_file = element_counts * Cint(element_size_file)
            # Cumulative sum of nodes per rank starting with an additional 0
            cum_node_counts_file = append!(zeros(eltype(node_counts_file), 1),
                                           cumsum(node_counts_file))

            # Read data of each process in slices (ranks start with 0)
            slice = (cum_node_counts_file[mpi_rank() + 1] + 1):cum_node_counts_file[mpi_rank() + 2]

            u_file = allocate_restart_file_data(u, nnodes_file,
                                                mesh, equations, dg, cache)
            for v in eachvariable(equations)
                var = file["variables_$v"]
                # Convert 1D array back to actual size of `u_file`
                u_file[v, .., :] = reshape(read(var)[slice],
                                           size(@view u_file[v, .., :]))
            end
            convert_restart_file_polydeg!(u, u_file, polydeg_file,
                                          mesh, equations, dg, cache,
                                          interpolate_high2low, correct_mass)
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

function load_restart_file_on_root(mesh::Union{TreeMeshParallel, P4estMeshParallel,
                                               T8codeMeshParallel},
                                   equations, dg::DG, cache,
                                   restart_file, interpolate_high2low, correct_mass)
    # allocate memory
    u_ode = allocate_coefficients(mesh, equations, dg, cache)
    u = wrap_array_native(u_ode, mesh, equations, dg, cache)

    # Only the root rank reads the restart file. All ranks need to know its polynomial
    # degree to decide whether a conversion is necessary.
    if mpi_isroot()
        polydeg_file = h5open(restart_file, "r") do file
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

            return Int(read(attributes(file)["polydeg"]))
        end
        MPI.Bcast!(Ref(polydeg_file), mpi_root(), mpi_comm())
    else
        polydeg_file = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())[]
    end

    ### Read variable data ###
    if polydeg_file != polydeg(dg) # Conversion is necessary
        # Distribute the data of the restart file to all ranks, which then convert the
        # data of their own elements. This is required for the mass correction, since
        # each rank knows only the geometry of its own elements.
        u_file = allocate_restart_file_data(u, polydeg_file + 1,
                                            mesh, equations, dg, cache)
        scatter_restart_file_variables!(u_file, restart_file, equations, cache)
        convert_restart_file_polydeg!(u, u_file, polydeg_file,
                                      mesh, equations, dg, cache,
                                      interpolate_high2low, correct_mass)
    else # Read in variables separately
        scatter_restart_file_variables!(u, restart_file, equations, cache)
    end

    return u_ode
end

# Read the solution variables of a restart file on the root rank and distribute them to
# the arrays `data` [variable, i, (j, (k),) element] of all ranks
function scatter_restart_file_variables!(data, restart_file, equations, cache)
    # Calculate node counts by MPI rank
    element_size = prod(size(data)[2:(end - 1)])
    element_counts = convert(Vector{Cint}, collect(cache.mpi_cache.n_elements_by_rank))
    node_counts = element_counts * Cint(element_size)

    if mpi_isroot()
        h5open(restart_file, "r") do file
            for v in eachvariable(equations)
                sendbuf = MPI.VBuffer(read(file["variables_$v"]), node_counts)
                MPI.Scatterv!(sendbuf, @view(data[v, .., :]), mpi_root(), mpi_comm())
            end
        end
    else
        # non-root ranks only receive data
        for v in eachvariable(equations)
            # put Scatterv in both blocks of the if condition to avoid type instability
            if isempty(data)
                MPI.Scatterv!(nothing, eltype(data)[], mpi_root(), mpi_comm())
            else
                MPI.Scatterv!(nothing, @view(data[v, .., :]), mpi_root(), mpi_comm())
            end
        end
    end

    return nothing
end

# Store controller values for an adaptive time stepping scheme
function save_adaptive_time_integrator(integrator, restart_callback)
    # Save only on root
    if mpi_isroot()
        @unpack output_directory = restart_callback
        timestep = integrator.stats.naccept

        # Filename based on current time step
        filename = joinpath(output_directory, @sprintf("restart_%09d.h5", timestep))

        # Open file (preserve existing content)
        h5open(filename, "r+") do file
            attributes(file)["time_integrator_dtpropose"] = integrator.dtpropose
            store_controller!(file, integrator)
        end
    end
end
end # @muladd
