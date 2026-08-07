using Trixi
using Trixi: trixi_include, wrap_array, mesh_equations_solver_cache, trixi_backend,
             have_nonconservative_terms, nelements, nvariables
using KernelAbstractions
using Printf

polydegs = (2, 3, 4, 5, 6)
versions = (0, 1, 2, 3, 4, 5, 6, 7, 8)
trees_per_dimension = (4, 4, 4)
levels = (3, 4)
cases = ((name = "Euler / flux_ranocha",
          elixir = joinpath(@__DIR__, "elixir_euler_source_terms.jl"),
          flux = flux_ranocha,
          extra_kwargs = (;)),
         (name = "MHD / combined",
          elixir = joinpath(@__DIR__,
                            "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
          flux = Trixi.flux_hindenlang_gassner_nonconservative_powell,
          # Avoid NaN due to the callback GLM
          extra_kwargs = (equations = IdealGlmMhdEquations3D(5 / 3, 1.0),)))

storage_type = nothing
real_type = Float64
nrep = 30                            # timed calls per batch
nouter = 5
outdir = joinpath(pkgdir(Trixi), "run", "results")

function setup_case(case, polydeg, initial_refinement_level, turbo)
    volume_flux = case.flux
    if turbo
        volume_flux = FluxTurbo(volume_flux)
    end

    elixir_module = Module()

    trixi_include(elixir_module, case.elixir;
                  polydeg = polydeg,
                  trees_per_dimension = trees_per_dimension,
                  initial_refinement_level = initial_refinement_level,
                  volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
                  storage_type = storage_type, real_type = real_type,
                  callbacks = nothing, sol = nothing,
                  case.extra_kwargs...)
    ode = elixir_module.ode
    semi = ode.p
    u_ode = ode.u0
    du_ode = similar(ode.u0)
    u = wrap_array(u_ode, semi)
    du = wrap_array(du_ode, semi)
    fill!(du, zero(eltype(du)))
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    backend = trixi_backend(u)
    return (; u, du, u_ode, du_ode, mesh, equations, solver, cache, backend,
            n_elements = nelements(solver, cache),
            n_variables = nvariables(equations))
end

function calc_volume_integral_version!(setup, version, turbo)
    if turbo
        turbo_trait = Trixi.True()
    else
        turbo_trait = Trixi.False()
    end
    Trixi.calc_volume_integral!(setup.backend, setup.du, setup.u, setup.mesh,
                                have_nonconservative_terms(setup.equations),
                                setup.equations, setup.solver.volume_integral,
                                setup.solver, setup.cache,
                                Val(version), turbo_trait)
end

@show storage_type
@show real_type
@show versions
@show levels

rows = NamedTuple[]
for case in cases,
    polydeg in polydegs,
    initial_refinement_level in levels

    timings = Dict{Tuple{Bool, Int}, NTuple{2, Float64}}()
    errors = Dict{Int, Float64}()
    n_elements = 0
    n_dofs = 0
    failed = false

    for turbo in (false, true)
        GC.gc()
        case_setup = try
            setup_case(case, polydeg, initial_refinement_level, turbo)
        catch err
            message = sprint(showerror, err)
            if occursin("out of memory", lowercase(message))
                @printf("\n%s  polydeg %d  level %d : does not fit\n",
                        case.name, polydeg, initial_refinement_level)
            else
                @printf("\n%s  polydeg %d  level %d : FAILED, not a memory limit\n",
                        case.name, polydeg, initial_refinement_level)
                println(message)
            end
            failed = true
            break
        end
        n_elements = case_setup.n_elements
        n_dofs = n_elements * (polydeg + 1)^3

        du_reference = nothing
        for version in versions
            # warm up
            launched = try
                calc_volume_integral_version!(case_setup, version, turbo)
                KernelAbstractions.synchronize(case_setup.backend)
                true
            catch
                false
            end
            if !launched
                continue
            end

            time_total = 0.0
            time_min = Inf
            for _ in 1:nouter
                time_batch = @elapsed begin
                    for _ in 1:nrep
                        calc_volume_integral_version!(case_setup, version, turbo)
                    end
                    KernelAbstractions.synchronize(case_setup.backend)
                end
                time_total += time_batch
                time_min = min(time_min, time_batch / nrep)
            end
            time_mean = time_total / (nouter * nrep)
            timings[(turbo, version)] = (time_mean, time_min)

            fill!(case_setup.du, zero(eltype(case_setup.du)))
            calc_volume_integral_version!(case_setup, version, turbo)
            KernelAbstractions.synchronize(case_setup.backend)
            du_version = Array(case_setup.du)
            if du_reference === nothing
                du_reference = du_version
            end
            relative_error = maximum(abs.(du_version .- du_reference)) /
                             maximum(abs, du_reference)
            errors[version] = max(get(errors, version, 0.0), relative_error)
        end
        case_setup = nothing
    end
    if failed
        continue
    end

    @printf("\n=== %s   polydeg %d   %d elements   %.2fM DOF ===\n",
            case.name, polydeg, n_elements, n_dofs/1e6)
    println(rpad("version", 9), rpad("plain mean", 12), rpad("plain min", 12),
            rpad("turbo mean", 12), rpad("turbo min", 12), rpad("turbo/plain", 13),
            "max |du - du_v0|")
    for version in versions
        plain = get(timings, (false, version), nothing)
        turbo_timing = get(timings, (true, version), nothing)
        if plain === nothing && turbo_timing === nothing
            @printf("%-9d%s\n", version, "  --  launch failed")
            continue
        end
        speedup = "  --  "
        if plain !== nothing && turbo_timing !== nothing
            speedup = @sprintf("%.2fx", plain[1]/turbo_timing[1])
        end
        show_time(t) = t === nothing ? "  --  " : @sprintf("%.2f", t * 1e9/n_dofs)
        @printf("%-9d%-12s%-12s%-12s%-12s%-13s%.2e\n", version,
                show_time(plain === nothing ? nothing : plain[1]),
                show_time(plain === nothing ? nothing : plain[2]),
                show_time(turbo_timing === nothing ? nothing : turbo_timing[1]),
                show_time(turbo_timing === nothing ? nothing : turbo_timing[2]),
                speedup, get(errors, version, NaN))
        for (turbo, timing) in ((false, plain), (true, turbo_timing))
            timing === nothing && continue
            label = turbo ? case.name * " + FluxTurbo" : case.name
            push!(rows,
                  (case = label, polydeg = polydeg,
                   level = initial_refinement_level, elements = n_elements,
                   version = version, time = timing[1], time_min = timing[2],
                   ns_per_dof = timing[1] * 1e9 / n_dofs,
                   ns_per_dof_min = timing[2] * 1e9 / n_dofs,
                   relative_error = get(errors, version, NaN)))
        end
    end
end

mkpath(outdir)
filename = string("benchmark_all",
                  "_v", join(versions, "-"),
                  "_p", join(polydegs, "-"),
                  "_t", prod(trees_per_dimension), "_L", join(levels, "-"),
                  ".csv")
open(joinpath(outdir, filename), "w") do io
    isempty(rows) && return
    println(io, join(string.(keys(rows[1])), ","))
    for r in rows
        fields = map(values(r)) do value
            if value isa AbstractFloat
                @sprintf("%.10g", value)
            else
                string(value)
            end
        end
        println(io, join(fields, ","))
    end
end
println("\nwrote ", joinpath(outdir, filename))
