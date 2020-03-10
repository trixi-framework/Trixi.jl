module Parallel

export finalize
export is_mpi_enabled
export comm
export domain_id
export ndomains
export is_mpi_root
export mpi_root
export @mpi_root
export @mpi_parallel
export mpi_print
export mpi_println
export Request
export Irecv!
export Isend
export Waitall!
export Reduce!
export Gather!
export Gatherv
export Scatterv
export Scatterv_in_place!
export Allreduce!
export Allgather!
export Bcast!
export mpi_finalize


# Allows quick manual disabling of MPI usage, even if MPI is available
_use_mpi = true


# Check if MPI package is available and initialize it accordingly
if Base.find_package("MPI") !== nothing && _use_mpi
  using MPI

  if !MPI.Initialized() && !MPI.Finalized()
    MPI.Init()
  end

  const _is_mpi_enabled = true
  const _comm = MPI.COMM_WORLD
  const _domain_id = MPI.Comm_rank(_comm)
  const _n_domains = MPI.Comm_size(_comm)
  const Request = MPI.Request
  Irecv!(args...) = MPI.Irecv!(args...)
  Isend(args...) = MPI.Isend(args...)
  Waitall!(args...) = MPI.Waitall!(args...)
  Reduce!(args...) = MPI.Reduce!(args...)
  Gather!(args...) = MPI.Gather!(args...)
  Gatherv(args...) = MPI.Gatherv(args...)
  Scatterv(args...) = MPI.Scatterv(args...)
  Scatterv_in_place!(args...) = MPI.Scatterv_in_place!(args...)
  Allreduce!(args...) = MPI.Allreduce!(args...)
  Allgather!(args...) = MPI.Allgather!(args...)
  Bcast!(args...) = MPI.Bcast!(args...)
else
  const _is_mpi_enabled = false
  const _comm = nothing
  const _domain_id = 0
  const _n_domains = 1
  const Request = Nothing
  Irecv!(args...) = nothing
  Isend(args...) = nothing
  Waitall!(args...) = nothing
  Reduce!(args...) = nothing
  Gather!(args...) = nothing
  Gatherv(args...) = nothing
  Scatterv(args...) = nothing
  Scatterv_in_place!(args...) = nothing
  Allreduce!(args...) = nothing
  Allgather!(args...) = nothing
  Bcast!(args...) = nothing
end


# Method to check if MPI is available in general
is_mpi_enabled() = _is_mpi_enabled

# Default MPI communicator
comm() = _comm

# Convenience methods
domain_id() = _domain_id
n_domains() = _n_domains
is_mpi_root() = domain_id() == 0
mpi_root() = 0

# Check if this is a parallel run
is_parallel() = is_mpi_enabled() && n_domains() > 1


# Macro to enable expressions only if MPI is used
macro mpi_enabled(expr)
  if is_mpi_enabled()
    return :($(esc(expr)))
  end
end

# Macro to enable expressions only if MPI is used and more than one rank is used
macro mpi_parallel(expr)
  if is_mpi_enabled() && is_parallel()
    return :($(esc(expr)))
  end
end

# Macro to enable expressions only on MPI root (or if MPI is disabled)
macro mpi_root(expr)
  if !is_mpi_enabled() || is_mpi_root()
    return :($(esc(expr)))
  end
end

# Print rank information in addition to message itself
mpi_print(msg) = print("rank $(domain_id()): $(msg...)")
mpi_println(msg) = println("rank $(domain_id()): $(msg...)")


# Finalize MPI
function mpi_finalize()
  if is_mpi_enabled() && _use_mpi && !isinteractive() && MPI.Initialized() && !MPI.Finalized()
    MPI.Finalize()
  end
end


end # module Parallel
