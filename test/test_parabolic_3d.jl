module TestExamplesParabolic3D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "SemidiscretizationHyperbolicParabolic (3D)" begin
#! format: noindent

@trixi_testset "DGMulti: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_3d",
                                 "elixir_navierstokes_convergence.jl"),
                        cells_per_dimension=(4, 4, 4), tspan=(0.0, 0.1),
                        l2=[
                            0.0005532846479614563,
                            0.000659263463988067,
                            0.0007776436003494915,
                            0.000659263463988129,
                            0.0038073624941206956
                        ],
                        linf=[
                            0.001703986341275776,
                            0.0026285618026252733,
                            0.00353105737957371,
                            0.002628561802588858,
                            0.015587831432887
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "DGMulti: elixir_navierstokes_convergence_curved.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_3d",
                                 "elixir_navierstokes_convergence_curved.jl"),
                        cells_per_dimension=(4, 4, 4), tspan=(0.0, 0.1),
                        l2=[
                            0.0014027227340680359,
                            0.0021322235583299425,
                            0.002787374145873934,
                            0.002458747307842109,
                            0.009978368214450204
                        ],
                        linf=[
                            0.006341750448945582,
                            0.010306014425485621,
                            0.015207402553448324,
                            0.010968264060799426,
                            0.04745438898236998
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "DGMulti: elixir_navierstokes_taylor_green_vortex.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "dgmulti_3d",
                                 "elixir_navierstokes_taylor_green_vortex.jl"),
                        cells_per_dimension=(4, 4, 4), tspan=(0.0, 0.25),
                        l2=[
                            0.00018257125088549987,
                            0.015589736346235174,
                            0.015589736346235415,
                            0.021943924698669025,
                            0.019273688367502154
                        ],
                        linf=[
                            0.0006268461326666142,
                            0.03218881686243058,
                            0.03218881686357877,
                            0.053872494644958,
                            0.05183811394229565
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh3D: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        l2=[
                            0.0019582188528520267,
                            0.002653449504302849,
                            0.002898264205184317,
                            0.0026534495043028534,
                            0.009511572365092744
                        ],
                        linf=[
                            0.013680656759089693,
                            0.03569104501543785,
                            0.023526343547761893,
                            0.03569104501543733,
                            0.11482570604049513
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh3D: elixir_navierstokes_convergence.jl (isothermal walls)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)),
                        l2=[
                            0.001954686519653731,
                            0.0026554367897028506,
                            0.0028927304027240026,
                            0.0026554367897028437,
                            0.00959635179660988
                        ],
                        linf=[
                            0.013680508110646583,
                            0.03567344635942522,
                            0.024024936779738822,
                            0.035673446359425674,
                            0.11839497110814179
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh3D: elixir_navierstokes_convergence.jl (Entropy gradient variables)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        gradient_variables=GradientVariablesEntropy(),
                        l2=[
                            0.0019770444875097737,
                            0.002652475094640119,
                            0.0029086003083239236,
                            0.002652475094640097,
                            0.009509568981441823
                        ],
                        linf=[
                            0.013879361129145007,
                            0.035262606093049195,
                            0.02355419709739138,
                            0.03526260609304984,
                            0.11719963716518933
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh3D: elixir_navierstokes_convergence.jl (Entropy gradient variables, isothermal walls)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        gradient_variables=GradientVariablesEntropy(),
                        heat_bc_top_bottom=Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                                              t,
                                                                                                                                              equations),
                                                                                             equations)),
                        l2=[
                            0.0019746314233993435,
                            0.0026547682591448896,
                            0.0029070310636460494,
                            0.0026547682591448922,
                            0.00958779288300152
                        ],
                        linf=[
                            0.013879193801400458,
                            0.03524408452641245,
                            0.023986146220843566,
                            0.035244084526412915,
                            0.1200505651257302
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh3D: elixir_navierstokes_convergence.jl (flux differencing)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        volume_integral=VolumeIntegralFluxDifferencing(flux_central),
                        l2=[
                            0.0019582188528208754,
                            0.0026534495043017935,
                            0.002898264205195059,
                            0.0026534495043017917,
                            0.009511572364832972
                        ],
                        linf=[
                            0.013680656758958687,
                            0.03569104501523916,
                            0.02352634354684648,
                            0.03569104501523987,
                            0.11482570603774533
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh3D: elixir_navierstokes_convergence.jl (Refined mesh)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        tspan=(0.0, 0.0))
    LLID = Trixi.local_leaf_cells(mesh.tree)
    num_leaves = length(LLID)
    @assert num_leaves % 16 == 0
    Trixi.refine!(mesh.tree, LLID[1:Int(num_leaves / 16)])
    tspan = (0.0, 0.25)
    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 boundary_conditions = (boundary_conditions,
                                                                        boundary_conditions_parabolic),
                                                 source_terms = source_terms_navier_stokes_convergence_test)
    ode = semidiscretize(semi, tspan)
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
    callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)
    sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
                dt = 1e-5,
                ode_default_options()..., callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈
          [0.00031093362536287433;
           0.0006473493036800964;
           0.0007705277238221976;
           0.0006280517917194624;
           0.0009039277899421355]
    @test linf_error ≈
          [0.0023694155363713776;
           0.01063493262248095;
           0.006772070862041679;
           0.010640551561807883;
           0.019256819037817507]
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh3D: elixir_navierstokes_taylor_green_vortex.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_navierstokes_taylor_green_vortex.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.25),
                        l2=[
                            0.00024173250389635442,
                            0.015684268393762454,
                            0.01568426839376248,
                            0.021991909545192333,
                            0.02825413672911425
                        ],
                        linf=[
                            0.0008410587892853094,
                            0.04740176181772552,
                            0.04740176181772507,
                            0.07483494924031157,
                            0.150181591534448
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh3D: elixir_navierstokes_taylor_green_vortex.jl (Refined mesh)" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_navierstokes_taylor_green_vortex.jl"),
                        tspan=(0.0, 0.0))
    LLID = Trixi.local_leaf_cells(mesh.tree)
    num_leaves = length(LLID)
    @assert num_leaves % 32 == 0
    Trixi.refine!(mesh.tree, LLID[1:Int(num_leaves / 32)])
    tspan = (0.0, 0.1)
    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver)
    ode = semidiscretize(semi, tspan)
    analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                         save_analysis = true,
                                         extra_analysis_integrals = (energy_kinetic,
                                                                     energy_internal,
                                                                     enstrophy))
    callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)
    # Use CarpenterKennedy2N54 since `RDPK3SpFSAL49` gives slightly different results on different machines
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
                dt = 5e-3,
                save_everystep = false, callback = callbacks)
    l2_error, linf_error = analysis_callback(sol)
    @test l2_error ≈ [
        7.314319856736271e-5,
        0.006266480163542894,
        0.006266489911815533,
        0.008829222305770226,
        0.0032859166842329228
    ]
    @test linf_error ≈ [
        0.0002943968186086554,
        0.013876261980614757,
        0.013883619864959451,
        0.025201279960491936,
        0.018679364985388247
    ]
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 100
        @test (@allocated Trixi.rhs_parabolic!(du_ode, u_ode, semi, t)) < 100
    end
end

@trixi_testset "P4estMesh3D: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_3d_dgsem",
                                 "elixir_navierstokes_convergence.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.1),
                        l2=[
                            0.00026599105557723507,
                            0.00046187779448444603,
                            0.0005424899076194272,
                            0.00046187779448445546,
                            0.0015846392584275121
                        ],
                        linf=[
                            0.0025241668964857134,
                            0.006308461684409397,
                            0.004334939668473314,
                            0.006308461684396753,
                            0.03176343483364796
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh3D: elixir_navierstokes_taylor_green_vortex.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_3d_dgsem",
                                 "elixir_navierstokes_taylor_green_vortex.jl"),
                        initial_refinement_level=2, tspan=(0.0, 0.25),
                        surface_flux=FluxHLL(min_max_speed_naive),
                        l2=[
                            0.0001547509861140407,
                            0.015637861347119624,
                            0.015637861347119687,
                            0.022024699158522523,
                            0.009711013505930812
                        ],
                        linf=[
                            0.0006696415247340326,
                            0.03442565722527785,
                            0.03442565722577423,
                            0.06295407168705314,
                            0.032857472756916195
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh3D: elixir_advection_diffusion_amr.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_advection_diffusion_amr.jl"),
                        initial_refinement_level=2,
                        base_level=2,
                        med_level=3,
                        max_level=4,
                        l2=[0.0003687746805397333],
                        linf=[0.0015844326511498252])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "TreeMesh3D: elixir_advection_diffusion_nonperiodic.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_advection_diffusion_nonperiodic.jl"),
                        l2=[0.0009808996243280868],
                        linf=[0.01732621559135459])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh3D: elixir_navierstokes_taylor_green_vortex_amr.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_3d_dgsem",
                                 "elixir_navierstokes_taylor_green_vortex_amr.jl"),
                        initial_refinement_level=0,
                        max_level=2,
                        tspan=(0.0, 0.1),
                        l2=[
                            0.001106911564430018,
                            0.013872454764036899,
                            0.013872454764036934,
                            0.012060120516483785,
                            0.14491994688373158
                        ],
                        linf=[
                            0.004408900465271981,
                            0.05154019951528149,
                            0.05154019951517075,
                            0.035283556918085636,
                            0.6804797525555557
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh3D: elixir_navierstokes_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_3d_dgsem",
                                 "elixir_navierstokes_blast_wave_amr.jl"),
                        tspan=(0.0, 0.01),
                        l2=[
                            0.009472104410520866, 0.0017883742549557149,
                            0.0017883742549557147, 0.0017883742549557196,
                            0.024388540048562748
                        ],
                        linf=[
                            0.6782397526873181, 0.17663702154066238,
                            0.17663702154066266, 0.17663702154066238, 1.7327849844825238
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh3D: elixir_navierstokes_viscous_shock.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_3d_dgsem",
                                 "elixir_navierstokes_viscous_shock.jl"),
                        l2=[
                            0.0002576235461250765,
                            0.0001433693418567713,
                            1.5583069105517042e-16,
                            1.257551423107977e-16,
                            0.00017369872990116004
                        ],
                        linf=[
                            0.0016731930282756213,
                            0.0010638586882356638,
                            2.738015991633e-15,
                            3.281831854493919e-15,
                            0.0011495231318404686
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "P4estMesh3D: elixir_navierstokes_viscous_shock_dirichlet_bc.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "p4est_3d_dgsem",
                                 "elixir_navierstokes_viscous_shock_dirichlet_bc.jl"),
                        l2=[
                            0.0002576236289909761,
                            0.00014336952925040063,
                            1.446929414778897e-16,
                            1.1843685819074592e-16,
                            0.00017369912976070335
                        ],
                        linf=[
                            0.0016731934566309725,
                            0.0010638647433908188,
                            2.9923988317619584e-15,
                            3.501658589636682e-15,
                            0.0011495334919645606
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
