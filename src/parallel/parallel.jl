module Parallel

export finalize
export have_mpi
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

  const _have_mpi = true
  const _comm = MPI.COMM_WORLD
  const _domain_id = MPI.Comm_rank(_comm)
  const _n_domains = MPI.Comm_size(_comm)
else
  const _have_mpi = false
  const _comm = nothing
  const _domain_id = 0
  const _n_domains = 1
end


# Method to check if MPI is available in general
have_mpi() = _have_mpi

# Default MPI communicator
comm() = _comm

# Convenience methods
domain_id() = _domain_id
n_domains() = _n_domains
is_mpi_root() = domain_id() == 0

# Check if this is a parallel run
if_parallel() = have_mpi() && n_domains() > 1


# Macro to enable expressions only if MPI is used
macro mpi(expr)
  if have_mpi()
    return :($(esc(expr)))
  end
end

# Macro to enable expressions only if MPI is used and more than one rank is used
macro mpi_parallel(expr)
  if have_mpi() && is_parallel()
    return :($(esc(expr)))
  end
end

# Macro to enable expressions only if MPI is used and this is the MPI root
macro mpi_root(expr)
  if have_mpi() && is_mpi_root()
    return :($(esc(expr)))
  end
end


# Finalize MPI
function finalize()
  if have_mpi() && _use_mpi && !isinteractive() && MPI.Initialized() && !MPI.Finalized()
    MPI.Finalize()
  end
end


end # module Parallel
