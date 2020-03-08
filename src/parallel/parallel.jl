module Parallel

export finalize
export is_mpi_enabled
export comm
export domain_id
export ndomains
export is_mpi_root
export @mpi
export @mpi_root


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
else
  const _is_mpi_enabled = false
  const _comm = nothing
  const _domain_id = 0
  const _n_domains = 1
end


# Method to check if MPI is available in general
is_mpi_enabled() = _is_mpi_enabled

# Default MPI communicator
comm() = _comm

# Convenience methods
domain_id() = _domain_id
n_domains() = _n_domains
is_mpi_root() = domain_id() == 0

# Check if this is a parallel run
if_parallel() = is_mpi_enabled() && n_domains() > 1


# Macro to enable expressions only if MPI is used
macro mpi(expr)
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

# Macro to enable expressions only if MPI is used and this is the MPI root
macro mpi_root(expr)
  if is_mpi_enabled() && is_mpi_root()
    return :($(esc(expr)))
  end
end


# Finalize MPI
function finalize()
  if is_mpi_enabled() && _use_mpi && !isinteractive() && MPI.Initialized() && !MPI.Finalized()
    MPI.Finalize()
  end
end


end # module Parallel
