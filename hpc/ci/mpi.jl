using MPI

mode = only(ARGS)
ranks, threads = if mode == "mpi"
    (3, 1)
elseif mode == "cuda_mpi"
    (2, 4)
else
    error("Unknown MPI validation: $mode.")
end

project = dirname(Base.active_project())
script = joinpath(@__DIR__, "validate.jl")
run(`$(MPI.mpiexec()) --map-by slot:PE=$threads --bind-to core --report-bindings -n $ranks $(Base.julia_cmd()) --startup-file=no -C $(ENV["JULIA_CPU_TARGET"]) --check-bounds=yes --threads=$threads --project=$project $script $mode`)
