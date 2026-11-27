# Measures the performance index (PID) of a scaling case and writes it in seconds to a file.
#
#   julia --threads=4 benchmark/scaling/run.jl <case> <output>
#   mpiexec -n 4 julia benchmark/scaling/run.jl <case> <output>
#
# A case is a Julia file that defines the `elixir` and its `trixi_include` `overrides`,
# for example `euler_source_terms.jl`.
# The overrides must disable the analysis callback, since it resets the call counter.
#
# The PID is the wall time per right-hand side evaluation and global degree of freedom,
# multiplied by the number of ranks and threads.
# It follows the definition of the `AnalysisCallback` and is constant under ideal scaling.
# The difference to a run with a single step removes the setup time from the measurement.
# Both runs are repeated, the fastest repetition filters out disturbances of single runs.
# Thread and rank placement is left to the caller.

using Trixi

case, output = ARGS
include(abspath(case))

function run_elixir(elixir, overrides)
    Trixi.MPI.Barrier(Trixi.mpi_comm())
    seconds = @elapsed trixi_include(elixir; overrides...)
    semi = @invokelatest Main.semi
    # Elixirs either solve into `sol` or step an `integrator`.
    solution = (@invokelatest isdefined(Main, :sol)) ? (@invokelatest Main.sol) :
               (@invokelatest Main.integrator).sol
    # An aborted run, for example after a vanishing time step, measures a wrong solution.
    retcode = solution.retcode
    @assert Symbol(retcode) in (:Success, :MaxIters) "The elixir stopped with $retcode."
    return seconds, Trixi.ncalls(semi.performance_counter), Trixi.ndofsglobal(semi)
end

const repetitions = 3
single_step = (; overrides..., maxiters = 1)
# The first run compiles the code.
run_elixir(elixir, single_step)
setup_seconds, setup_calls, _ = minimum(_ -> run_elixir(elixir, single_step), 1:repetitions)
seconds, calls, ndofs = minimum(_ -> run_elixir(elixir, overrides), 1:repetitions)
@assert calls > setup_calls

pid = (seconds - setup_seconds) * Trixi.mpi_nranks() * Threads.nthreads() /
      (ndofs * (calls - setup_calls))
if Trixi.mpi_isroot()
    write(output, string(pid))
    println("PID $pid s with $(Trixi.mpi_nranks()) ranks, $(Threads.nthreads()) threads, ",
            "$ndofs degrees of freedom and ",
            "$(calls - setup_calls) right-hand side evaluations.")
end
