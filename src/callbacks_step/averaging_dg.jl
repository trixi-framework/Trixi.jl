function save_averaging_file(averaging_callback, mesh::TreeMesh, equations, dg::DGSEM, cache)
  @unpack output_directory, filename, mean_values = averaging_callback
  h5open(joinpath(output_directory, filename), "w") do file
    # Add context information
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["polydeg"] = polydeg(dg)
    attributes(file)["nelements"] = nelements(dg, cache)

    # Store data as 1D arrays
    file["v_mean"] = mean_values.v_mean[:]
    file["c_mean"] = mean_values.c_mean[:]
    file["rho_mean"] = mean_values.rho_mean[:]
    file["vorticity_mean"] = mean_values.vorticity_mean[:]
  end

  return filename
end


function load_averaging_file(averaging_file, mesh::TreeMesh, equations, dg::DGSEM, cache)
  # Read and check mesh and solver info
  n_dims, n_nodes, n_elements = h5open(averaging_file, "r") do file
    n_dims = read(attributes(file)["ndims"])
    n_nodes = read(attributes(file)["polydeg"]) + 1
    n_elements = read(attributes(file)["nelements"])

    @assert n_dims == ndims(mesh) "ndims differs from value in averaging file"
    @assert n_nodes - 1 == polydeg(dg) "polynomial degree in solver differs from value in averaging file"
    @assert n_elements == nelements(dg, cache) "nelements in solver differs from value in averaging file"

    return n_dims, n_nodes, n_elements
  end

  # Read data as 1D vectors
  v_mean_vec, c_mean_vec, rho_mean_vec, vorticity_mean_vec = h5open(averaging_file, "r") do file
    return read(file["v_mean"]),
           read(file["c_mean"]),
           read(file["rho_mean"]),
           read(file["vorticity_mean"])
  end

  # Reshape arrays into solver-specific format
  v_mean         = reshape(v_mean_vec,         (n_dims, repeat([n_nodes], n_dims)..., n_elements))
  c_mean         = reshape(c_mean_vec,         (repeat([n_nodes], n_dims)..., n_elements))
  rho_mean       = reshape(rho_mean_vec,       (repeat([n_nodes], n_dims)..., n_elements))
  vorticity_mean = reshape(vorticity_mean_vec, (repeat([n_nodes], n_dims)..., n_elements))

  return (; v_mean, c_mean, rho_mean, vorticity_mean)
end