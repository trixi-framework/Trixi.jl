# Running selected MPI test items

This file complements [`AGENTS.md`](../../AGENTS.md). Test items with the tag
`:mpi` (e.g., in `test/test_mpi_p4est_2d.jl`) exercise the MPI code (e.g.,
`P4estMeshParallel`) only when run with more than one MPI rank; in a single
Julia process, the mesh is not parallel. Thus, run selected MPI test items in
the shell (not in a persistent session via MCP) with `mpiexec`, as the `mpi`
suite does, but without running the complete suite:

1. Write a script calling `@run_package_tests` with a filter as in
   "Running selected test items" in `AGENTS.md`, e.g.,
   `run_agents/run_mpi_items.jl`. The script must be located in `run_agents/`:
   when called from a file, `@run_package_tests` searches the parent directory
   of the directory of this file, so a script in another directory (e.g., a
   temporary directory) finds no test items.
2. Launch it from the root of the repository via `MPI.mpiexec()`:
   ```bash
   julia --project=run_agents -e 'using MPI; julia = Base.julia_cmd()[1]
       run(`$(MPI.mpiexec()) -n 2 $julia --threads=1 --check-bounds=yes --project=run_agents run_agents/run_mpi_items.jl`)'
   ```
   Do not call the `mpiexec` binary of the MPI.jl artifact directly from the
   shell; it fails since `libhwloc` is not found.
3. Each rank prints its own test summary, and the output of the ranks may be
   interleaved. Check that all ranks report only passed tests and that the
   output of the elixirs shows `#MPI ranks: 2`.
4. Remove the script afterward.
