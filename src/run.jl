
# Apply the function `f` to `expr` and all sub-expressions recursively.
walkexpr(f, expr::Expr) = f(Expr(expr.head, (walkexpr(f, arg) for arg in expr.args)...))
walkexpr(f, x) = f(x)

# Replace assignments to `key` in `expr` by `key = val` for all `(key,val)` in `kwargs`.
function replace_assignments(expr; kwargs...)
  # replace explicit and keyword assignemnts
  expr = walkexpr(expr) do x
    if x isa Expr
      for (key,val) in kwargs
        if (x.head === Symbol("=") || x.head === :kw) && x.args[1] === Symbol(key)
          x.args[2] = :( $val )
          # dump(x)
        end
      end
    end
    return x
  end

  return expr
end

# find a (keyword or common) assignment to `destination` in `expr`
# and return the assigned value
function find_assignment(expr, destination)
  # declare result to be able to assign to it in the closure
  local result

  # find explicit and keyword assignemnts
  walkexpr(expr) do x
    if x isa Expr
      if (x.head === Symbol("=") || x.head === :kw) && x.args[1] === Symbol(destination)
        result = x.args[2]
        # dump(x)
      end
    end
    return x
  end

  result
end


# Note: Wa can't call the method below `Trixi.include` since that is created automatically
# inside `module Trixi` to `include` source files and evaluate them within the global scope
# of `Trixi`. However, users will want to evaluate in the global scope of `Main` or something
# similar to manage dependencies on their own.
"""
    trixi_include([mod::Module=Main,] elixir::AbstractString; kwargs...)

`include` the file `elixir` and evaluate its content in the global scope of module `mod`.
You can override specific assignments in `elixir` by supplying keyword arguments.
It's basic purpose is to make it easier to modify some parameters while running Trixi from the
REPL. Additionally, this is used in tests to reduce the computational burden for CI while still
providing examples with sensible default values for users.

# Examples

```jldoctest
julia> trixi_include(default_example(), tspan=(0.0, 0.1))
...
julia> sol.t[end]
0.1
```
"""
function trixi_include(mod::Module, elixir::AbstractString; kwargs...)
  Base.include(ex -> replace_assignments(ex; kwargs...), mod, elixir)
end

trixi_include(elixir::AbstractString; kwargs...) = trixi_include(Main, elixir; kwargs...)


"""
    convergence_test([mod::Module=Main,] elixir::AbstractString, iterations; kwargs...)

Run `iterations` Trixi simulations using the setup given in `elixir` and compute
the experimental order of convergence (EOC) in the ``L^2`` and ``L^\\infty`` norm.
In each iteration, the `initial_refinement_level` will be increased by 1.
Additional keyword arguments `kwargs...` and the optional module `mod` are passed directly
to [`trixi_include`](@ref).
"""
function convergence_test(mod::Module, elixir::AbstractString, iterations; kwargs...)
  @assert(iterations > 1, "Number of iterations must be bigger than 1 for a convergence analysis")

  # Types of errors to be calcuated
  errors = Dict(:l2 => Float64[], :linf => Float64[])

  # get the initial_refinement_level from the elixir
  code = read(elixir, String)
  expr = Meta.parse("begin $code end")
  initial_refinement_level = find_assignment(expr, :initial_refinement_level)

  # run simulations and extract errors
  for iter in 1:iterations
    println("Running convtest iteration ", iter, "/", iterations)
    trixi_include(mod, elixir; kwargs..., initial_refinement_level=initial_refinement_level+iter-1)
    l2_error, linf_error = mod.analysis_callback(mod.sol)

    # collect errors as one vector to reshape later
    append!(errors[:l2],   l2_error)
    append!(errors[:linf], linf_error)

    println("\n\n")
    println("#"^80)
  end

  # number of variables
  variablenames = varnames_cons(mod.equations)
  nvariables = length(variablenames)

  # Reshape errors to get a matrix where the i-th row represents the i-th iteration
  # and the j-th column represents the j-th variable
  errorsmatrix = Dict(kind => transpose(reshape(error, (nvariables, iterations))) for (kind, error) in errors)

  # Calculate EOCs where the columns represent the variables
  # As dx halves in every iteration the denominator needs to be log(1/2)
  eocs = Dict(kind => log.(error[2:end, :] ./ error[1:end-1, :]) ./ log(1 / 2) for (kind, error) in errorsmatrix)

  eoc_mean_values = Dict{Symbol,Any}()
  eoc_mean_values[:variables] = variablenames

  for (kind, error) in errorsmatrix
    println(kind)

    for v in variablenames
      @printf("%-20s", v)
    end
    println("")

    for k = 1:nvariables
      @printf("%-10s", "error")
      @printf("%-10s", "EOC")
    end
    println("")

    # Print errors for the first iteration
    for k = 1:nvariables
      @printf("%-10.2e", error[1, k])
      @printf("%-10s", "-")
    end
    println("")

    # For the following iterations print errors and EOCs
    for j = 2:iterations
      for k = 1:nvariables
        @printf("%-10.2e", error[j, k])
        @printf("%-10.2f", eocs[kind][j-1, k])
      end
      println("")
    end
    println("")

    # Print mean EOCs
    mean_values = zeros(nvariables)
    for v in 1:nvariables
      mean_values[v] = sum(eocs[kind][:, v]) ./ length(eocs[kind][:, v])
      @printf("%-10s", "mean")
      @printf("%-10.2f", mean_values[v])
    end
    eoc_mean_values[kind] = mean_values
    println("")
    println("-"^80)
  end

  return eoc_mean_values
end

convergence_test(elixir::AbstractString, iterations; kwargs...) = convergence_test(Main, elixir::AbstractString, iterations; kwargs...)



# TODO: Taal remove
"""
    run(parameters_file; verbose=false, refinement_level_increment=0, parameters...)

Run a Trixi simulation with the parameters in `parameters_file`.
Parameters can be overriden by specifying them as keyword arguments (see examples).

If `verbose` is `true`, additional output will be generated on the terminal
that may help with debugging. If a value for `refinement_level_increment` is given,
`initial_refinement_level` will be increased by this value before running the simulation (mostly
used by EOC analysis).

# Examples
```julia
julia> Trixi.run("examples/parameters_advection_basic.toml", verbose=true)
[...]
```

Without changing the parameters file we can start a simulation with `polydeg = 1` and
`t_end = 0.5` as follows:
```julia
julia> Trixi.run("examples/parameters_advection_basic.toml", polydeg=1, t_end=0.5)
[...]
```
"""
function run(parameters_file; verbose=false, refinement_level_increment=0, parameters...)
  # Reset timer
  reset_timer!(timer())

  # Read command line or keyword arguments and parse parameters file
  init_parameters(parameters_file; verbose=verbose,
      refinement_level_increment=refinement_level_increment, parameters...)

  # Separate initialization and execution into two functions such that Julia can specialize
  # the code in `run_simulation` for the actual type of `solver` and `mesh`
  if parameter("equations") == "euler_gravity"
    globals[:euler_gravity] = true
    mesh, solver, time_parameters, time_integration_function = init_simulation_euler_gravity()
    run_simulation_euler_gravity(mesh, solver, time_parameters, time_integration_function)
  else
    globals[:euler_gravity] = false
    mesh, solver, time_parameters, time_integration_function = init_simulation()
    run_simulation(mesh, solver, time_parameters, time_integration_function)
  end
end


# TODO: Taal remove
function init_parameters(parameters_file=nothing; verbose=false, refinement_level_increment=0, parameters...)
  # Set global verbosity
  globals[:verbose] = verbose

  # Parse parameters file
  @timeit timer() "read parameter file" parse_parameters_file(parameters_file)

  # Override specified parameters
  for (parameter, value) in parameters
    setparameter(string(parameter), value)
  end

  # Start simulation with an increased initial refinement level if specified
  # for convergence analysis
  if refinement_level_increment != 0
    setparameter("initial_refinement_level",
      parameter("initial_refinement_level") + refinement_level_increment)
  end
end


# TODO: Taal remove
function init_simulation()
  # Print starup message
  print_startup_message()

  # Get number of dimensions
  ndims_ = parameter("ndims")::Int

  # Check if this is a restart from a previous result or a new simulation
  restart = parameter("restart", false)
  if restart
    restart_filename = parameter("restart_filename")
  end

  # Initialize mesh
  if restart
    mpi_print("Loading mesh... ")
    @timeit timer() "mesh loading" mesh = load_mesh(restart_filename)
    mpi_isparallel() && MPI.Barrier(mpi_comm())
    mpi_println("done")
  else
    mpi_print("Creating mesh... ")
    @timeit timer() "mesh creation" mesh = generate_mesh()
    mesh.current_filename = save_mesh_file(mesh, parameter("output_directory", "out"))
    mesh.unsaved_changes = false
    mpi_isparallel() && MPI.Barrier(mpi_comm())
    mpi_println("done")
  end

  # Initialize system of equations
  mpi_print("Initializing system of equations... ")
  equations_name = parameter("equations")
  equations = make_equations(equations_name, ndims_)
  mpi_isparallel() && MPI.Barrier(mpi_comm())
  mpi_println("done")

  # Initialize solver
  mpi_print("Initializing solver... ")
  solver_name = parameter("solver", valid=["dg"])
  solver = make_solver(solver_name, equations, mesh)
  mpi_isparallel() && MPI.Barrier(mpi_comm())
  mpi_println("done")

  # Sanity checks
  # If DG volume integral type is weak form, volume flux type must be flux_central,
  # as everything else does not make sense
  if solver.volume_integral_type === Val(:weak_form) && solver.volume_flux_function !== flux_central
    error("using the weak formulation with a volume flux other than 'flux_central' does not make sense")
  end

  if equations isa AbstractIdealGlmMhdEquations && solver.volume_integral_type === Val(:weak_form)
    error("The weak form is not implemented for $equations.")
  end

  # Initialize solution
  amr_interval = parameter("amr_interval", 0)
  adapt_initial_condition = parameter("adapt_initial_condition", true)
  adapt_initial_condition_only_refine = parameter("adapt_initial_condition_only_refine", true)
  if restart
    mpi_print("Loading restart file...")
    time, step = load_restart_file!(solver, restart_filename)
    mpi_isparallel() && MPI.Barrier(mpi_comm())
    mpi_println("done")
  else
    mpi_print("Applying initial conditions... ")
    t_start = parameter("t_start")
    time = t_start
    step = 0
    set_initial_condition!(solver, time)
    mpi_isparallel() && MPI.Barrier(mpi_comm())
    mpi_println("done")

    # If AMR is enabled, adapt mesh and re-apply ICs
    if amr_interval > 0 && adapt_initial_condition
      @timeit timer() "initial condition AMR" has_changed = adapt!(mesh, solver, time,
          only_refine=adapt_initial_condition_only_refine)

      # Iterate until mesh does not change anymore
      while has_changed
        set_initial_condition!(solver, time)
        @timeit timer() "initial condition AMR" has_changed = adapt!(mesh, solver, time,
            only_refine=adapt_initial_condition_only_refine)
      end

      # Save mesh file
      mesh.current_filename = save_mesh_file(mesh, parameter("output_directory", "out"))
      mesh.unsaved_changes = false
    end
  end
  t_end = parameter("t_end")

  # Init time integration
  time_integration_scheme = Symbol(parameter("time_integration_scheme", "timestep_carpenter_kennedy_erk54_2N!"))
  time_integration_function = eval(time_integration_scheme)

  # Print setup information
  solution_interval = parameter("solution_interval", 0)
  restart_interval = parameter("restart_interval", 0)
  polydeg = parameter("polydeg") # FIXME: This is currently the only DG-specific code in here
  n_steps_max = parameter("n_steps_max")
  cfl = parameter("cfl")
  source_terms = parameter("source_terms", "none")
  n_leaf_cells = count_leaf_cells(mesh.tree)
  min_level = minimum_level(mesh.tree)
  max_level = maximum_level(mesh.tree)
  domain_center = mesh.tree.center_level_0
  domain_length = mesh.tree.length_level_0
  min_dx = domain_length / 2^max_level
  max_dx = domain_length / 2^min_level
  s = ""
  s *= """| Simulation setup
          | ----------------
          | working directory:  $(pwd())
          | parameters file:    $(parameter("parameters_file"))
          | equations:          $(get_name(equations))
          | | #variables:       $(nvariables(equations))
          | | variable names:   $(join(varnames_cons(equations), ", "))
          | source_terms:       $source_terms
          | restart:            $(restart ? "yes" : "no")
          """
  if restart
    s *= "| | restart timestep: $step\n"
    s *= "| | restart time:     $time\n"
  else
    s *= "| initial conditions: $(get_name(solver.initial_condition))\n"
    s *= "| t_start:            $t_start\n"
  end
  s *= """| t_end:              $t_end
          | AMR:                $(amr_interval > 0 ? "yes" : "no")
          """
  if amr_interval > 0
    s *= "| | AMR interval:     $amr_interval\n"
    s *= "| | adapt ICs:        $(adapt_initial_condition ? "yes" : "no")\n"
  end
  s *= """| n_steps_max:        $n_steps_max
          | time integration:   $(get_name(time_integration_function))
          | restart interval:   $restart_interval
          | solution interval:  $solution_interval
          | #MPI ranks:         $(mpi_nranks())
          | #threads/rank:      $(Threads.nthreads())
          |
          | Solver (local)
          | | solver:           $solver_name
          | | polydeg:          $polydeg
          | | CFL:              $cfl
          | | volume integral:  $(get_name(solver.volume_integral_type))
          | | volume flux:      $(get_name(solver.volume_flux_function))
          | | surface flux:     $(get_name(solver.surface_flux_function))
          | | #elements:        $(solver.n_elements)
          | | #interfaces:      $(solver.n_interfaces)
          | | #boundaries:      $(solver.n_boundaries)
          | | #l2mortars:       $(solver.n_l2mortars)
          | | #DOFs:            $(ndofs(solver))
          |
          | Mesh (global)
          | | #cells:           $(length(mesh.tree))
          | | #leaf cells:      $n_leaf_cells
          | | minimum level:    $min_level
          | | maximum level:    $max_level
          | | domain center:    $(join(domain_center, ", "))
          | | domain length:    $domain_length
          | | minimum dx:       $min_dx
          | | maximum dx:       $max_dx
          """
  mpi_println()
  mpi_println(s)

  # Set up main loop
  save_final_solution = parameter("save_final_solution", true)
  save_final_restart = parameter("save_final_restart", true)
  analysis_interval = parameter("analysis_interval", 0)
  if analysis_interval > 0
    alive_interval = parameter("alive_interval", div(analysis_interval, 10))
  else
    alive_interval = 0
  end

  # Save initial conditions if desired
  if !restart && parameter("save_initial_solution", true)
    # we need to make sure, that derived quantities, such as e.g. blending
    # factor is already computed for the initial condition
    @notimeit timer() rhs!(solver, time)
    save_solution_file(solver, mesh, time, 0, step)
  end

  # Print initial solution analysis and initialize solution analysis
  if analysis_interval > 0
    analyze_solution(solver, mesh, time, 0, step, 0, 0)
  end

  time_parameters = (time=time, step=step, t_end=t_end, cfl=cfl,
                    n_steps_max=n_steps_max,
                    save_final_solution=save_final_solution,
                    save_final_restart=save_final_restart,
                    analysis_interval=analysis_interval,
                    alive_interval=alive_interval,
                    solution_interval=solution_interval,
                    amr_interval=amr_interval,
                    restart_interval=restart_interval)
  return mesh, solver, time_parameters, time_integration_function
end


# TODO: Taal remove
function run_simulation(mesh, solver, time_parameters, time_integration_function)
  @unpack time, step, t_end, cfl, n_steps_max,
          save_final_solution, save_final_restart,
          analysis_interval, alive_interval,
          solution_interval, amr_interval,
          restart_interval = time_parameters

  loop_start_time = time_ns()
  analysis_start_time = time_ns()
  output_time = 0.0
  n_analysis_timesteps = 0

  # Declare variables to return error norms calculated in main loop
  # (needed such that the values are accessible outside of the main loop)
  local l2_error, linf_error

  # Start main loop (loop until final time step is reached)
  finalstep = false
  first_loop_iteration = true
  @timeit timer() "main loop" while !finalstep
    # Calculate time step size
    @timeit timer() "calculate dt" dt = calc_dt(solver, cfl)

    # Abort if time step size is NaN
    if isnan(dt)
      error("time step size `dt` is NaN")
    end

    # If the next iteration would push the simulation beyond the end time, set dt accordingly
    if time + dt > t_end || isapprox(time + dt, t_end)
      dt = t_end - time
      finalstep = true
    end

    # Evolve solution by one time step
    time_integration_function(solver, time, dt)
    step += 1
    time += dt
    n_analysis_timesteps += 1

    # Check if we reached the maximum number of time steps
    if step == n_steps_max
      finalstep = true
    end

    # Check steady-state integration residual
    if solver.equations isa AbstractHyperbolicDiffusionEquations
      resid = maximum(abs, view(solver.elements.u_t, 1, .., :))

      if mpi_isparallel()
        resid = MPI.Allreduce!(Ref(resid), max, mpi_comm())[]
      end

      if resid <= solver.equations.resid_tol
        mpi_println()
        mpi_println("-"^80)
        mpi_println("  Steady state tolerance of ", solver.equations.resid_tol,
                    " reached at time ", time)
        mpi_println("-"^80)
        mpi_println()
        finalstep = true
      end
    end

    # Analyze solution errors
    if analysis_interval > 0 && (step % analysis_interval == 0 || finalstep)
      # Calculate absolute and relative runtime
      if mpi_isparallel()
        total_dofs = MPI.Reduce!(Ref(ndofs(solver)), +, mpi_root(), mpi_comm())
        total_dofs = mpi_isroot() ? total_dofs[] : -1
      else
        total_dofs = ndofs(solver)
      end
      runtime_absolute = (time_ns() - loop_start_time) / 10^9
      runtime_relative = ((time_ns() - analysis_start_time - output_time) / 10^9 /
                          (n_analysis_timesteps * total_dofs))

      # Analyze solution
      l2_error, linf_error = @timeit timer() "analyze solution" analyze_solution(
          solver, mesh, time, dt, step, runtime_absolute, runtime_relative)

      # Reset time and counters
      analysis_start_time = time_ns()
      output_time = 0.0
      n_analysis_timesteps = 0
      if finalstep
        mpi_println("-"^80)
        mpi_println("Trixi simulation run finished.    Final time: $time    Time steps: $step")
        mpi_println("-"^80)
        mpi_println()
      end
    elseif alive_interval > 0 && step % alive_interval == 0 && mpi_isroot()
      runtime_absolute = (time_ns() - loop_start_time) / 10^9
      @printf("#t/s: %6d | dt: %.4e | Sim. time: %.4e | Run time: %.4e s\n",
              step, dt, time, runtime_absolute)
    end

    # Write solution file
    if solution_interval > 0 && (
        step % solution_interval == 0 || (finalstep && save_final_solution))
      output_start_time = time_ns()
      @timeit timer() "I/O" begin
        # Compute current AMR indicator values such that it can be written to
        # the solution file for the current number of elements
        if amr_interval > 0
          calc_amr_indicator(solver, mesh, time)
        end

        # If mesh has changed, write a new mesh file name
        if mesh.unsaved_changes
          mesh.current_filename = save_mesh_file(mesh, parameter("output_directory", "out"), step)
          mesh.unsaved_changes = false
        end

        # Then write solution file
        save_solution_file(solver, mesh, time, dt, step)
      end
      output_time += time_ns() - output_start_time
    end

    # Write restart file
    if restart_interval > 0 && (
        step % restart_interval == 0 || (finalstep && save_final_restart))
      output_start_time = time_ns()
      @timeit timer() "I/O" begin
        # If mesh has changed, write a new mesh file
        if mesh.unsaved_changes
          mesh.current_filename = save_mesh_file(mesh, parameter("output_directory", "out"), step)
          mesh.unsaved_changes = false
        end

        # Then write restart file
        save_restart_file(solver, mesh, time, dt, step)
      end
      output_time += time_ns() - output_start_time
    end

    # Perform adaptive mesh refinement
    if amr_interval > 0 && (step % amr_interval == 0) && !finalstep
      @timeit timer() "AMR" has_changed = adapt!(mesh, solver, time)

      # Store if mesh has changed to write changed mesh file before next restart/solution output
      if has_changed
        mesh.unsaved_changes = true
      end
    end

    # The following call ensures that when doing memory allocation
    # measurements, the memory allocations for JIT compilation are discarded
    # (since virtually all relevant methods have already been called by now)
    if first_loop_iteration
      clear_malloc_data()
      first_loop_iteration = false
    end
  end

  # Print timer information
  if mpi_isroot()
    print_timer(timer(), title="Trixi.jl", allocations=true, linechars=:ascii, compact=false)
    println()
  end

  # Distribute l2_errors from root such that all ranks have correct return value
  if mpi_isparallel()
    l2_error   = convert(typeof(l2_error),   MPI.Bcast!(collect(l2_error),   mpi_root(), mpi_comm()))
    linf_error = convert(typeof(linf_error), MPI.Bcast!(collect(linf_error), mpi_root(), mpi_comm()))
  end

  # Return error norms for EOC calculation
  return l2_error, linf_error, varnames_cons(solver.equations)
end


# TODO: Taal migrate
"""
    convtest(parameters_file, iterations; parameters...)

Run multiple Trixi simulations with the parameters in `parameters_file` and compute
the experimental order of convergence (EOC) in the ``L^2`` and ``L^\\infty`` norm.
The number of runs is specified by `iterations` and in each run the initial
refinement level will be increased by 1. Parameters can be overriden by specifying them as
additional keyword arguments, which are passed to the respective call to `run`.
"""
function convtest(parameters_file, iterations; parameters...)
  if mpi_isroot()
    @assert(iterations > 1, "Number of iterations must be bigger than 1 for a convergence analysis")
  end

  # Types of errors to be calcuated
  errors = Dict(:L2 => Float64[], :Linf => Float64[])

  # Declare variable to access variable names after for loop
  local variablenames

  # Run trixi and extract errors
  for i = 1:iterations
    mpi_println(string("Running convtest iteration ", i, "/", iterations))
    l2_error, linf_error, variablenames = run(parameters_file; refinement_level_increment = i - 1,
                                              parameters...)

    # Collect errors as one vector to reshape later
    append!(errors[:L2], l2_error)
    append!(errors[:Linf], linf_error)
  end

  # Number of variables
  nvariables = length(variablenames)

  # Reshape errors to get a matrix where the i-th row represents the i-th iteration
  # and the j-th column represents the j-th variable
  errorsmatrix = Dict(kind => transpose(reshape(error, (nvariables, iterations))) for (kind, error) in errors)

  # Calculate EOCs where the columns represent the variables
  # As dx halves in every iteration the denominator needs to be log(1/2)
  eocs = Dict(kind => log.(error[2:end, :] ./ error[1:end-1, :]) ./ log(1 / 2) for (kind, error) in errorsmatrix)


  if mpi_isroot()
    for (kind, error) in errorsmatrix
      println(kind)

      for v in variablenames
        @printf("%-20s", v)
      end
      println("")

      for k = 1:nvariables
        @printf("%-10s", "error")
        @printf("%-10s", "EOC")
      end
      println("")

      # Print errors for the first iteration
      for k = 1:nvariables
        @printf("%-10.2e", error[1, k])
        @printf("%-10s", "-")
      end
      println("")

      # For the following iterations print errors and EOCs
      for j = 2:iterations
        for k = 1:nvariables
          @printf("%-10.2e", error[j, k])
          @printf("%-10.2f", eocs[kind][j-1, k])
        end
        println("")
      end
      println("")

      # Print mean EOCs
      for k = 1:nvariables
        @printf("%-10s", "mean")
        @printf("%-10.2f", sum(eocs[kind][:, k]) ./ length(eocs[kind][:, k]))
      end
      println("")
      println("-"^80)
    end
  end
end


"""
    compute_linear_structure(parameters_file, source_terms=nothing; verbose=false, parameters...)

Computes the exact Jacobian `A` of a linear DG operator wrapped as a `LinearMap` and the right hand side `b`.
Returns `A, b`.
"""
function compute_linear_structure(parameters_file, source_terms=nothing; verbose=false, parameters...)
  # Read command line or keyword arguments and parse parameters file
  init_parameters(parameters_file; verbose=verbose, parameters...)
  globals[:euler_gravity] = false
  mesh, solver, time_parameters = init_simulation()

  equations(solver) isa Union{AbstractLinearScalarAdvectionEquation, AbstractHyperbolicDiffusionEquations} ||
    throw(ArgumentError("Only linear problems are supported."))

  # get the right hand side from the source terms
  solver.elements.u .= 0
  rhs!(solver, 0)
  b = vec(-solver.elements.u_t) |> copy

  # set the source terms to zero to extract the linear operator
  if solver isa Dg1D
    solver = Dg1D(solver.equations, solver.surface_flux_function, solver.volume_flux_function, solver.initial_condition,
                  source_terms, mesh, polydeg(solver))
  elseif solver isa Dg2D
    solver = Dg2D(solver.equations, solver.surface_flux_function, solver.volume_flux_function, solver.initial_condition,
                  source_terms, mesh, polydeg(solver))
  elseif solver isa Dg3D
    solver = Dg3D(solver.equations, solver.surface_flux_function, solver.volume_flux_function, solver.initial_condition,
                  source_terms, mesh, polydeg(solver))
  else
    error("not implemented")
  end
  A = LinearMap(length(solver.elements.u), ismutating=true) do dest,src
    vec(solver.elements.u) .= src
    rhs!(solver, 0)
    dest .= vec(solver.elements.u_t)
  end

  A, b
end


"""
    compute_jacobian_dg(parameters_file=nothing; verbose=false, parameters...)

Uses DG right hand side operator and simple second order finite difference to compute the Jacobian `J` of the operator.
The linearisation state is the initial condition from the parameter file.
Returns `J`.
"""
function compute_jacobian_dg(parameters_file; verbose=false, parameters...)
  # Read command line or keyword arguments and parse parameters file
  init_parameters(parameters_file; verbose=verbose, parameters...)
  # function does not support multi-physics
  if parameter("equations") == "euler_gravity"
    throw(ArgumentError("Multi-physics such as Euler-gravity is not supported"))
  end
  globals[:euler_gravity] = false

  # linearisation state is initial condition
  mesh, dg, time_parameters = init_simulation()
  # store initial state
  u0 = dg.elements.u |> copy

  #compute residual of linearisation state
  rhs!(dg, 0)
  res0 = vec(dg.elements.u_t) |> copy

  # initialize Jacobian matrix
  J = zeros(length(dg.elements.u),length(dg.elements.u))

  # use second order finite difference to estimate Jacobian matrix
  for idx in eachindex(dg.elements.u)
    # determine size of fluctuation
    epsilon = sqrt(eps(u0[idx]))
    # plus fluctuation
    dg.elements.u[idx] = u0[idx] + epsilon
    rhs!(dg, 0)
    # stores the right hand side with plus epsilon fluctuation
    res_p = vec(dg.elements.u_t) |> copy
    # minus fluctuation
    dg.elements.u[idx] = u0[idx] - epsilon
    rhs!(dg, 0)
    # stores the right hand side with minus epsilon fluctuation
    res_m = vec(dg.elements.u_t) |> copy
    # restore linearisation state
    dg.elements.u[idx] = u0[idx]
    # central second order finite difference
    J[:,idx] = (res_p - res_m) / (2 * epsilon)
  end

  return J
end

# Include source file with init and run methods for coupled Euler-gravity simulations
include("run_euler_gravity.jl")
