using Trixi
using Trixi: trixi_include, wrap_array, mesh_equations_solver_cache, trixi_backend,
             have_nonconservative_terms, nelements, nvariables
using KernelAbstractions
using Printf

polydegs = (2, 3, 4, 5, 6)
versions = (0, 1, 2, 3, 4, 5, 6, 7, 8)
polydegs = (3, 4, 5, 6, 7)
versions = (0, 1, 2, 5)
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
    turbo in (false, true)
    initial_refinement_level in levels,
    polydeg in polydegs

    GC.gc()
    if turbo
        case_label = case.name * " + FluxTurbo"
    else
        case_label = case.name
    end

    case_setup = try
        setup_case(case, polydeg, initial_refinement_level, turbo)
    catch err
        message = sprint(showerror, err)
        if occursin("out of memory", lowercase(message))
            @printf("\n%s  polydeg %d  level %d : does not fit\n",
                    case_label, polydeg, initial_refinement_level)
        else
            @printf("\n%s  polydeg %d  level %d : FAILED, not a memory limit\n",
                    case_label, polydeg, initial_refinement_level)
            println(message)
        end
        continue
    end

    n_dofs = case_setup.n_elements * (polydeg + 1)^3
    @printf("\n=== %s   polydeg %d   %d elements   %.2fM DOF ===\n",
            case_label, polydeg, case_setup.n_elements, n_dofs/1e6)
    println(rpad("version", 10), rpad("mean [ns/dof]", 15), rpad("min [ns/dof]", 14),
            rpad("vs v0", 10), "max |du - du_v0|")

    du_reference = nothing
    time_reference = NaN
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
            @printf("%-10d%s\n", version, "  --  launch failed")
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

        fill!(case_setup.du, zero(eltype(case_setup.du)))
        calc_volume_integral_version!(case_setup, version, turbo)
        KernelAbstractions.synchronize(case_setup.backend)
        du_version = Array(case_setup.du)
        if du_reference === nothing
            du_reference = du_version
            time_reference = time_mean
        end
        relative_error = maximum(abs.(du_version .- du_reference)) /
                         maximum(abs, du_reference)

        push!(rows,
              (case = case_label, polydeg = polydeg,
               level = initial_refinement_level,
               elements = case_setup.n_elements, version = version,
               time = time_mean, time_min = time_min,
               ns_per_dof = time_mean * 1e9 / n_dofs,
               ns_per_dof_min = time_min * 1e9 / n_dofs,
               vs_v0 = time_reference / time_mean,
               relative_error = relative_error))
        @printf("%-10d%-15.2f%-14.2f%-10s%.2e\n", version,
                time_mean * 1e9/n_dofs,
                time_min * 1e9/n_dofs,
                @sprintf("%.2fx", time_reference/time_mean),
                relative_error)
    end
    case_setup = nothing
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
