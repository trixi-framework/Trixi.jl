# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    convergence_test([mod::Module=Main,] elixir::AbstractString, iterations; kwargs...)

Run `iterations` Trixi.jl simulations using the setup given in `elixir` and compute
the experimental order of convergence (EOC) in the ``L^2`` and ``L^\\infty`` norm.
In each iteration, the resolution of the respective mesh will be doubled.
Additional keyword arguments `kwargs...` and the optional module `mod` are passed directly
to [`trixi_include`](@ref).

This function assumes that the spatial resolution is set via the keywords
`initial_refinement_level` (an integer) or `cells_per_dimension` (a tuple of
integers, one per spatial dimension).
"""
function convergence_test(mod::Module, elixir::AbstractString, iterations; kwargs...)
    @assert(iterations>1,
            "Number of iterations must be bigger than 1 for a convergence analysis")

    # Types of errors to be calculated
    errors = Dict(:l2 => Float64[], :linf => Float64[])

    initial_resolution = extract_initial_resolution(elixir, kwargs)

    # run simulations and extract errors
    for iter in 1:iterations
        println("Running convtest iteration ", iter, "/", iterations)

        include_refined(mod, elixir, initial_resolution, iter; kwargs)

        l2_error, linf_error = mod.analysis_callback(mod.sol)

        # collect errors as one vector to reshape later
        append!(errors[:l2], l2_error)
        append!(errors[:linf], linf_error)

        println("\n\n")
        println("#"^100)
    end

    # Use raw error values to compute EOC
    analyze_convergence(errors, iterations, mod.semi)
end

# Analyze convergence for any semidiscretization
# Note: this intermediate method is to allow dispatching on the semidiscretization
function analyze_convergence(errors, iterations, semi::AbstractSemidiscretization)
    _, equations, _, _ = mesh_equations_solver_cache(semi)
    variablenames = varnames(cons2cons, equations)
    analyze_convergence(errors, iterations, variablenames)
end

# This method is called with the collected error values to actually compute and print the EOC
function analyze_convergence(errors, iterations,
                             variablenames::Union{Tuple, AbstractArray})
    nvariables = length(variablenames)

    # Reshape errors to get a matrix where the i-th row represents the i-th iteration
    # and the j-th column represents the j-th variable
    errorsmatrix = Dict(kind => transpose(reshape(error, (nvariables, iterations)))
                        for (kind, error) in errors)

    # Calculate EOCs where the columns represent the variables
    # As dx halves in every iteration the denominator needs to be log(1/2)
    eocs = Dict(kind => log.(error[2:end, :] ./ error[1:(end - 1), :]) ./ log(1 / 2)
                for (kind, error) in errorsmatrix)

    eoc_mean_values = Dict{Symbol, Any}()
    eoc_mean_values[:variables] = variablenames

    for (kind, error) in errorsmatrix
        println(kind)

        for v in variablenames
            @printf("%-20s", v)
        end
        println("")

        for k in 1:nvariables
            @printf("%-10s", "error")
            @printf("%-10s", "EOC")
        end
        println("")

        # Print errors for the first iteration
        for k in 1:nvariables
            @printf("%-10.2e", error[1, k])
            @printf("%-10s", "-")
        end
        println("")

        # For the following iterations print errors and EOCs
        for j in 2:iterations
            for k in 1:nvariables
                @printf("%-10.2e", error[j, k])
                @printf("%-10.2f", eocs[kind][j - 1, k])
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
        println("-"^100)
    end

    return eoc_mean_values
end

function convergence_test(elixir::AbstractString, iterations; kwargs...)
    convergence_test(Main, elixir::AbstractString, iterations; kwargs...)
end

# Helper methods used in the functions defined above

# Searches for the assignment that specifies the mesh resolution in the elixir
function extract_initial_resolution(elixir, kwargs)
    code = read(elixir, String)
    expr = Meta.parse("begin \n$code \nend")

    try
        # get the initial_refinement_level from the elixir
        initial_refinement_level = TrixiBase.find_assignment(expr,
                                                             :initial_refinement_level)

        if haskey(kwargs, :initial_refinement_level)
            return kwargs[:initial_refinement_level]
        else
            return initial_refinement_level
        end
    catch e
        # If `initial_refinement_level` is not found, we will get an `ArgumentError`
        if isa(e, ArgumentError)
            try
                # get cells_per_dimension from the elixir
                cells_per_dimension = eval(TrixiBase.find_assignment(expr,
                                                                     :cells_per_dimension))

                if haskey(kwargs, :cells_per_dimension)
                    return kwargs[:cells_per_dimension]
                else
                    return cells_per_dimension
                end
            catch e2
                # If `cells_per_dimension` is not found either
                if isa(e2, ArgumentError)
                    throw(ArgumentError("`convergence_test` requires the elixir to define " *
                                        "`initial_refinement_level` or `cells_per_dimension`"))
                else
                    rethrow()
                end
            end
        else
            rethrow()
        end
    end
end

# runs the specified elixir with a doubled resolution each time iter is increased by 1
# works for TreeMesh
function include_refined(mod, elixir, initial_refinement_level::Int, iter; kwargs)
    trixi_include(mod, elixir; kwargs...,
                  initial_refinement_level = initial_refinement_level + iter - 1)
end

# runs the specified elixir with a doubled resolution each time iter is increased by 1
# works for StructuredMesh
function include_refined(mod, elixir, cells_per_dimension::NTuple{NDIMS, Int}, iter;
                         kwargs) where {NDIMS}
    new_cells_per_dimension = cells_per_dimension .* 2^(iter - 1)

    trixi_include(mod, elixir; kwargs..., cells_per_dimension = new_cells_per_dimension)
end
end # @muladd
