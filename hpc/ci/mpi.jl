using MPI

mode = only(ARGS)
threads, mapping = if mode == "mpi"
    (1, "ppr:1:node:PE=1")
elseif mode == "cuda_mpi"
    (4, "slot:PE=4")
else
    error("Unknown MPI validation: $mode.")
end

project = dirname(Base.active_project())
script = joinpath(@__DIR__, "validate.jl")
target = ENV["JULIA_CPU_TARGET"]
julia = `$(Base.julia_cmd()) --startup-file=no -C $target --check-bounds=yes
         --threads=$threads --project=$project $script $mode`
run(`$(MPI.mpiexec()) --map-by $mapping --bind-to core --report-bindings -n 2 $julia`)
