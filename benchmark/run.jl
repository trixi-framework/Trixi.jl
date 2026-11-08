# Measures the performance index (PID) of an elixir and writes it in seconds to the output file.
#
#   julia benchmark/run.jl <elixir> <options> <output>
#
# The options are Julia code evaluated in `Main`, their last value is a named tuple of `trixi_include` overrides, for example
#
#   julia benchmark/run.jl examples/p4est_3d_dgsem/elixir_euler_source_terms.jl \
#       "(; initial_refinement_level = 2, maxiters = 20, analysis_callback = TrivialCallback(), save_solution = TrivialCallback())" pid.txt
#
# The PID is the wall time per right-hand side evaluation and global degree of freedom, multiplied by the number of ranks and threads.
# It follows the definition of the `AnalysisCallback` and stays constant under ideal scaling.
# The difference to a run with a single step removes the setup time from the measurement.
# The overrides must disable the analysis callback, since it resets the counter of right-hand side evaluations.

using Trixi

elixir, options, output = ARGS
# `trixi_include` resolves relative paths against this file instead of the working directory.
elixir = abspath(elixir)
overrides = Core.eval(Main, Meta.parseall(options))

function run_elixir(elixir, overrides)
    Trixi.MPI.Barrier(Trixi.mpi_comm())
    seconds = @elapsed trixi_include(elixir; overrides...)
    semi = @invokelatest Main.semi
    # Elixirs either solve into `sol` or step an `integrator`.
    solution = (@invokelatest isdefined(Main, :sol)) ? (@invokelatest Main.sol) :
               (@invokelatest Main.integrator).sol
    # An aborted run, for example after a vanishing time step, measures fewer steps of a wrong solution.
    @assert Symbol(solution.retcode) in (:Success, :MaxIters) "The elixir stopped with $(solution.retcode)."
    return seconds, Trixi.ncalls(semi.performance_counter), Trixi.ndofsglobal(semi)
end

# The first run compiles the code.
run_elixir(elixir, (; overrides..., maxiters = 1))
setup_seconds, setup_calls, _ = run_elixir(elixir, (; overrides..., maxiters = 1))
seconds, calls, ndofs = run_elixir(elixir, overrides)
@assert calls > setup_calls

pid = (seconds - setup_seconds) * Trixi.mpi_nranks() * Threads.nthreads() /
      (ndofs * (calls - setup_calls))
if Trixi.mpi_isroot()
    write(output, string(pid))
    println("PID $pid s with $(Trixi.mpi_nranks()) ranks, $(Threads.nthreads()) threads, ",
            "$ndofs degrees of freedom and $(calls - setup_calls) right-hand side evaluations.")
end
