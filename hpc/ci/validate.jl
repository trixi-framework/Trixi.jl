using TOML

const root = ENV["CI_PROJECT_DIR"]
const mode = only(ARGS)
@assert "$(VERSION.major).$(VERSION.minor)" == ENV["JULIA_VERSION"]
@assert Base.JLOptions().check_bounds == 1
@assert unsafe_string(Base.JLOptions().cpu_target) == ENV["JULIA_CPU_TARGET"]
@assert readchomp(`git -C $root rev-parse HEAD`) == ENV["CI_COMMIT_SHA"]

using Trixi, CUDA, MPI, Libdl, Test, TestItemRunner
include(joinpath(root, "test", "test_trixi.jl"))
function run_items(names)
    selected = Set{String}()
    TestItemRunner.run_tests(joinpath(root, "test");
                             filter = item -> begin
                                 item.name in names || return false
                                 push!(selected, item.name)
                                 return true
                             end, verbose = true)
    @assert selected == Set(names)
end
function affinity()
    line = match(r"(?m)^Cpus_allowed_list:\s*(.*)$",
    read("/proc/self/status", String)).captures[1]
    cpus = Int[]
    for part in split(line, ',')
        limits = parse.(Int, split(part, '-'))
        append!(cpus, first(limits):last(limits))
    end
    @assert length(cpus) >= Threads.nthreads()
    println("Rank $(MPI.Comm_rank(MPI.COMM_WORLD)), threads $(Threads.nthreads()), CPUs $line.")
    return cpus
end

try
    preferences = TOML.parsefile(joinpath(root, "hpc", "Project.toml"))["preferences"]
    @assert MPI.MPIPreferences.binary == "system"
    @assert MPI.MPIPreferences.abi == preferences["MPIPreferences"]["abi"]
    @assert realpath(Libdl.dlpath(MPI.libmpi)) ==
            realpath(preferences["MPIPreferences"]["libmpi"])
    @assert Base.thisminor(CUDA.runtime_version()) ==
            VersionNumber(preferences["CUDA_Runtime_jll"]["version"])
    cpus = affinity()
    if mode in ("polyester", "kernelabstractions")
        @assert MPI.Comm_size(MPI.COMM_WORLD) == 1
        @assert Threads.nthreads() == (mode == "polyester" ? 4 : 3)
        @assert Trixi._PREFERENCE_THREADING == Symbol(mode)
        names = if mode == "polyester"
            ("P4estMesh2D: elixir_advection_basic.jl",
             "T8codeMesh2D: elixir_advection_basic.jl")
        else
            ("KernelAbstractions CPU 2D: elixir_advection_basic.jl",)
        end
        run_items(names)
    elseif mode == "cuda"
        @assert MPI.Comm_size(MPI.COMM_WORLD) == 1
        @assert Threads.nthreads() == 4
        CUDA.functional() || error("CUDA is required for this job.")
        CUDA.versioninfo()
        run_items(("CUDA 2D: elixir_advection_basic.jl Float32 / CUDA",))
    elseif mode in ("mpi", "cuda_mpi")
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nranks = MPI.Comm_size(comm)
        @assert nranks == (mode == "mpi" ? 3 : 2)
        @assert Threads.nthreads() == (mode == "mpi" ? 1 : 4)
        highest_cpu = MPI.Allreduce(maximum(cpus), max, comm)
        mask = zeros(Int, highest_cpu + 1)
        mask[cpus .+ 1] .= 1
        masks = reshape(MPI.Allgather(mask, comm), :, nranks)
        @assert all(sum(masks; dims = 2) .<= 1)
        if mode == "mpi"
            @assert Trixi.HDF5.has_parallel()
            filename = joinpath(ENV["RESULTS"], "parallel.h5")
            Trixi.HDF5.h5open(filename, "w", comm) do file
                dataset = Trixi.HDF5.create_dataset(file, "ranks",
                                                    Trixi.HDF5.datatype(Int),
                                                    Trixi.HDF5.dataspace(4, nranks);
                                                    dxpl_mpio = :collective)
                dataset[:, rank + 1] = fill(rank, 4)
                close(dataset)
            end
            MPI.Barrier(comm)
            Trixi.HDF5.h5open(filename, "r", comm) do file
                @assert read(file, "ranks"; dxpl_mpio = :collective)[:, rank + 1] ==
                        fill(rank, 4)
            end
            run_items(("P4estMesh MPI 2D: elixir_advection_basic.jl",
                       "T8codeMesh MPI 2D: elixir_advection_basic.jl"))
        else
            @assert MPI.Allreduce(Int(CUDA.functional()), min, comm) == 1
            local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
            @assert MPI.Comm_size(local_comm) == 2
            Trixi.KernelAbstractions.device!(CUDABackend(),
                                             MPI.Comm_rank(local_comm) + 1)
            uuid = collect(codeunits(string(CUDA.uuid(CUDA.device()))))
            uuids = reshape(MPI.Allgather(uuid, comm), :, nranks)
            @assert length(unique(String(uuids[:, i]) for i in 1:nranks)) == 2
            println("Rank $rank uses GPU $(CUDA.uuid(CUDA.device())).")
            send = CUDA.fill(Float64(rank + 1), 1024)
            recv = similar(send)
            CUDA.synchronize()
            MPI.Sendrecv!(send, recv, comm; dest = 1 - rank, source = 1 - rank)
            @assert all(Array(recv) .== Float64(2 - rank))
            @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                         "elixir_advection_basic.jl"),
                                l2=[8.311947673061856e-6], linf=[6.627000273229378e-5],
                                real_type=Float64, storage_type=CuArray)
            @assert ode.u0 isa CuArray{Float64}
            @assert mesh isa Trixi.P4estMeshParallel{2}
        end
        MPI.Barrier(comm)
    else
        error("Unknown validation check: $mode.")
    end
catch error
    showerror(stderr, error, catch_backtrace())
    println(stderr)
    MPI.Abort(MPI.COMM_WORLD, 1)
    rethrow()
end
