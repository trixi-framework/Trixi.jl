# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function save_averaging_file(averaging_callback, mesh::TreeMesh, equations, dg::DGSEM,
                             cache)
    @unpack output_directory, filename, mean_values = averaging_callback
    h5open(joinpath(output_directory, filename), "w") do file
        # Add context information
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_elements"] = nelements(dg, cache)

        # Store all mean variables as multi-dimensional arrays
        for field in fieldnames(typeof(mean_values))
            name = string(field)
            data = getfield(mean_values, field)
            file[name] = data
        end
    end

    return filename
end

function load_averaging_file(averaging_file, mesh::TreeMesh, equations, dg::DGSEM,
                             cache)
    # Read and check mesh and solver info
    h5open(averaging_file, "r") do file
        n_dims = read(attributes(file)["ndims"])
        n_nodes = read(attributes(file)["polydeg"]) + 1
        n_elements = read(attributes(file)["n_elements"])

        @assert n_dims==ndims(mesh) "ndims differs from value in averaging file"
        @assert n_nodes - 1==polydeg(dg) "polynomial degree in solver differs from value in averaging file"
        @assert n_elements==nelements(dg, cache) "nelements in solver differs from value in averaging file"
    end

    # Read and return mean values
    v_mean, c_mean, rho_mean, vorticity_mean = h5open(averaging_file, "r") do file
        return read(file["v_mean"]),
               read(file["c_mean"]),
               read(file["rho_mean"]),
               read(file["vorticity_mean"])
    end

    return (; v_mean, c_mean, rho_mean, vorticity_mean)
end
end # @muladd
