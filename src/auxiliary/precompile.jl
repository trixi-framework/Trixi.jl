# This file is used to generate precompile statements. This increases the
# precompilation time but decreases the time to first simulation - at least
# when precompiled methods can be reused.
# See https://timholy.github.io/SnoopCompile.jl/stable/snoop_pc/

using SnoopPrecompile: @precompile_all_calls

@precompile_all_calls begin
  # 2D
  let
    equations = CompressibleEulerEquations2D(1.4)
    solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

    coordinates_min = (0.0, 0.0)
    coordinates_max = (2.0, 2.0)
    refinement_patches = (
         (type="box", coordinates_min=(0.0, 0.0), coordinates_max=(1.0, 1.0)),
       )
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level=1,
                    refinement_patches=refinement_patches,
                    n_cells_max=100)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                        source_terms=source_terms_convergence_test)

    tspan = (0.0, 2.0)
    ode = semidiscretize(semi, tspan)

    du_ode = similar(ode.u0)
    Trixi.rhs!(du_ode, ode.u0, semi, tspan[1])

    summary_callback = SummaryCallback()

    analysis_interval = 100
    analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
    let u_ode = ode.u0
      GC.@preserve u_ode du_ode begin
        u = Trixi.wrap_array(u_ode, semi)
        du = Trixi.wrap_array(du_ode, semi)
        analysis_callback.affect!(devnull, du, u, u_ode, first(tspan), semi)
      end
    end

    alive_callback = AliveCallback(analysis_interval=analysis_interval)
  end

  # 3D
  let
    equations = CompressibleEulerEquations3D(1.4)
    solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

    coordinates_min = (0.0, 0.0, 0.0)
    coordinates_max = (2.0, 2.0, 2.0)
    refinement_patches = (
         (type="box", coordinates_min=(0.0, 0.0, 0.0), coordinates_max=(1.0, 1.0, 1.0)),
       )
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level=1,
                    refinement_patches=refinement_patches,
                    n_cells_max=100)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                        source_terms=source_terms_convergence_test)

    tspan = (0.0, 2.0)
    ode = semidiscretize(semi, tspan)

    du_ode = similar(ode.u0)
    Trixi.rhs!(du_ode, ode.u0, semi, tspan[1])

    summary_callback = SummaryCallback()

    analysis_interval = 100
    analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
    let u_ode = ode.u0
      GC.@preserve u_ode du_ode begin
        u = Trixi.wrap_array(u_ode, semi)
        du = Trixi.wrap_array(du_ode, semi)
        analysis_callback.affect!(devnull, du, u, u_ode, first(tspan), semi)
      end
    end

    alive_callback = AliveCallback(analysis_interval=analysis_interval)
  end
end
