module TestExamplesDGMulti2D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "dgmulti_2d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "DGMulti 2D" begin
  @trixi_testset "elixir_euler_weakform.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4, 4),
      l2 = [0.0013536930300254945, 0.0014315603442106193, 0.001431560344211359, 0.0047393341007602625],
      linf = [0.001514260921466004, 0.0020623991944839215, 0.002062399194485476, 0.004897700392503701],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4, 4),
      approximation_type = SBP(),
      l2 = [0.0074706882014934735, 0.005306220583603261, 0.005306220583613591, 0.014724842607716771],
      linf = [0.021563604940952885, 0.01359397832530762, 0.013593978324845324, 0.03270995869587523]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (Quadrilateral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4, 4),
      element_type = Quad(),
      l2 = [0.00031892254415307093, 0.00033637562986771894, 0.0003363756298680649, 0.0011100259064243145],
      linf = [0.001073298211445639, 0.0013568139808282087, 0.0013568139808290969, 0.0032249020004324613]
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (EC) " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4, 4),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      l2 = [0.007801417730672109, 0.00708583561714128, 0.0070858356171393, 0.015217574294198809],
      linf = [0.011572828457858897, 0.013965298735070686, 0.01396529873508534, 0.04227683691807904],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4, 4),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      approximation_type = SBP(),
      l2 = [0.01280067571168776, 0.010607599608273302, 0.010607599608239775, 0.026408338014056548],
      linf = [0.037983023185674814, 0.05321027922533417, 0.05321027922608157, 0.13392025411844033],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (Quadrilateral elements, SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
      cells_per_dimension = (4, 4),
      element_type = Quad(),
      volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
      surface_integral = SurfaceIntegralWeakForm(flux_ranocha),
      approximation_type = SBP(),
      l2 = [0.0029373718090697975, 0.0030629360605489465, 0.003062936060545615, 0.0068486089344859755],
      linf = [0.01360165305316885, 0.01267402847925303, 0.012674028479251254, 0.02210545278615017],
    )
  end

  @trixi_testset "elixir_euler_weakform.jl (convergence)" begin
    mean_convergence = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"), 2)
    @test isapprox(mean_convergence[:l2], [4.243843382379403, 4.128314378833922, 4.128314378397532, 4.081366752807379], rtol=0.05)
  end

  @trixi_testset "elixir_euler_weakform_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
      l2 = [0.0014986508075708323, 0.001528523420746786, 0.0015285234207473158, 0.004846505183839211],
      linf = [0.0015062108658376872, 0.0019373508504645365, 0.0019373508504538783, 0.004742686826709086]
    )
  end

  @trixi_testset "elixir_euler_triangulate_pkg_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangulate_pkg_mesh.jl"),
      l2 = [4.664661209491976e-6, 3.7033509525940745e-6, 4.794877426562555e-6, 1.2682723101532175e-5],
      linf = [2.5099852761334418e-5, 2.2683684021362893e-5, 2.6180448559287584e-5, 5.5752932611508044e-5]
    )
  end

  @trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability.jl"),
      cells_per_dimension = (32, 32), tspan = (0.0, 0.2),
      l2 = [0.11140378947116614, 0.06598161188703612, 0.10448953167839563, 0.16023209181809595],
      linf = [0.24033843177853664, 0.1659992245272325, 0.1235468309508845, 0.26911424973147735],
    )
  end

  @trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl (Quadrilateral elements, GaussSBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability.jl"),
      cells_per_dimension = (32, 32), element_type = Quad(), approximation_type=GaussSBP(), tspan = (0.0, 0.2),
      l2 = [0.11141270656347146, 0.06598888014584121, 0.1044902203749932, 0.16023037364774995],
      linf = [0.2414760062126462, 0.1662111846065654, 0.12344140473946856, 0.26978428189564774]
    )
  end

  @trixi_testset "elixir_euler_rayleigh_taylor_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_rayleigh_taylor_instability.jl"),
      cells_per_dimension = (8, 8), tspan = (0.0, 0.2),
      l2 = [0.03548329484912729, 0.002591414376082683, 0.006916327792623457, 0.016235069002818153],
      linf = [0.4783963902824797, 0.022527207050681054, 0.040307056293369226, 0.0852365428206836],
    )
  end

  @trixi_testset "elixir_euler_brown_minion_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_brown_minion_vortex.jl"),
      cells_per_dimension = 4, tspan = (0.0, 0.1),
      l2 = [0.0066800016110776066, 0.021516763475855016, 0.01069652423536524, 0.15052841129693573],
      linf = [0.01544756362800248, 0.09517304772476806, 0.021957154972646383, 0.33773439650806303],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Quad)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      cells_per_dimension = 4,
      l2 = [0.1562707966203802, 0.19692319035938727, 0.19692319035938863, 0.10641395363893119,
            0.7221962824695998, 0.07678102773723876, 0.07678102773723913, 0.11142445367787217,
            0.006627119909705745],
      linf = [0.16447597822733662, 0.244157345789029, 0.24415734578903472, 0.11982440036793476,
              0.7450328339751362, 0.06357382685763713, 0.0635738268576378, 0.1058830287485999,
              0.005740591170062146],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
      cells_per_dimension = 4, element_type = Tri(),
      l2 = [0.1348987236501758, 0.1588650593363661, 0.15886505933636905, 0.08571028568614296,
            0.6031732736338957, 0.06286520146660214, 0.06286520146660948, 0.0894610083051161,
            0.003693188791960107],
      linf = [0.16290247390873458, 0.2256891306641319, 0.2256891306641336, 0.09476017042552534,
              0.6906308908961734, 0.05349939593012487, 0.05349939593013042, 0.08830587480616725,
              0.0029551359803035027]
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave_SBP.jl (Quad)" begin
    # These setups do not pass CI reliably, see
    # https://github.com/trixi-framework/Trixi.jl/pull/880 and
    # https://github.com/trixi-framework/Trixi.jl/issues/881
    @test_skip @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave_SBP.jl"),
      cells_per_dimension = 4,
      l2 = [0.15825983698241494, 0.19897219694837923, 0.19784182473275247, 0.10482833997417325,
            0.7310752391255246, 0.07374056714564853, 0.07371172293240634, 0.10782032253431281,
            0.004921676235111545],
      linf = [0.1765644464978685, 0.2627803272865769, 0.26358136695848144, 0.12347681727447984,
              0.7733289736898254, 0.06695360844467957, 0.06650382120802623, 0.10885097000919097,
              0.007212567638078835],
    )
  end

  @trixi_testset "elixir_mhd_weak_blast_wave_SBP.jl (Tri)" begin
    # These setups do not pass CI reliably, see
    # https://github.com/trixi-framework/Trixi.jl/pull/880 and
    # https://github.com/trixi-framework/Trixi.jl/issues/881
    @test_skip @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave_SBP.jl"),
      cells_per_dimension = 4, element_type=Tri(), tspan = (0.0, 0.2),
      l2 = [0.13825044764021147, 0.15472815448314997, 0.1549093274293255, 0.053103596213755405,
            0.7246162776815603, 0.07730777596615901, 0.07733438386480523, 0.109893463921706,
            0.00617678167062838],
      linf = [0.22701306227317952, 0.2905255794821543, 0.2912409425436937, 0.08051361477962096,
              1.0842974228656006, 0.07866000368926784, 0.0786646354518149, 0.1614896380292925,
              0.010358210347485542],
    )
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl (Quad, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      cells_per_dimension = 8, element_type = Quad(), approximation_type = SBP(),
      l2 = [0.0028731817391463733, 0.033473047741338476, 0.04873655357923285, 2.7341648767587304e-15],
      linf = [0.010385010095182778, 0.08750628939565086, 0.12088392994348407, 9.325873406851315e-15]
    )
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl (Tri, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      cells_per_dimension = 8, element_type = Tri(), approximation_type = SBP(),
      l2 = [0.005912374812011487, 0.0993653660027428, 0.1670914777143732, 3.2948191109718012e-15],
      linf = [0.02076003852980346, 0.29169601664914424, 0.5674183379872275, 1.1546319456101628e-14]
    )
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl (Tri, Polynomial)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      cells_per_dimension = 8, element_type = Tri(), approximation_type = Polynomial(),
      # The last l2, linf error are the L2 projection error in approximating `b`, so they are not
      # zero for general non-collocated quadrature rules (e.g., for `element_type=Tri()`, `polydeg > 2`).
      l2 = [0.0011751205240013974, 0.021530712606619058, 0.022675456966150455, 1.813056761616414e-5],
      linf = [0.001888045014140971, 0.05466838692127718, 0.06345885709961152, 3.3989933098554914e-5]
    )
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl (Quad, Polynomial)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      cells_per_dimension = 8, element_type = Quad(), approximation_type = Polynomial(),
      # The last l2, linf error are the L2 projection error in approximating `b`. However, this is zero
      # for `Quad()` elements with `Polynomial()` approximations because the quadrature rule defaults to
      # a `(polydeg + 1)`-point Gauss quadrature rule in each coordinate (in general, StartUpDG.jl defaults
      # to the quadrature rule with the fewest number of points which exactly integrates the mass matrix).
      l2 = [0.00010550686466016327, 0.005212211092466598, 0.005529801974869844, 3.062184075681645e-15],
      linf = [0.000259995400729629, 0.0072236204211630906, 0.010364675200833062, 1.021405182655144e-14]
    )
  end


end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
