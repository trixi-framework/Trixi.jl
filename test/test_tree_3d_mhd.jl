module TestExamples3DMHD

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_3d_dgsem")

@testset "MHD" begin
  @trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.017590099293094203, 0.017695875823827714, 0.017695875823827686, 0.017698038279620777, 0.07495006099352074, 0.010391801950005755, 0.010391801950005759, 0.010393502246627087, 2.524766553484067e-16],
      linf = [0.28173002819718196, 0.3297583616136297, 0.32975836161363004, 0.356862935505337, 1.2893514981209626, 0.10950981489747313, 0.10950981489747136, 0.11517234329681891, 2.0816911067714202e-15])
  end

  @trixi_testset "elixir_mhd_ec.jl with initial_condition=initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [4.270231310667203e-16, 2.4381208042014784e-15, 5.345107673575357e-15, 3.00313882171883e-15, 1.7772703118758417e-14, 1.0340110783830874e-15, 1.1779095371939702e-15, 9.961878521814573e-16, 8.1201730630719145e-16],
      linf = [2.4424906541753444e-15, 2.881028748902281e-14, 2.4646951146678475e-14, 2.3092638912203256e-14, 2.3447910280083306e-13, 1.7763568394002505e-14, 1.0436096431476471e-14, 2.042810365310288e-14, 7.057203733035201e-15],
      atol = 1000*eps(),
      initial_condition=initial_condition_constant)
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [0.0032217291057246157, 0.009511644936958913, 0.004217358459420256, 0.011591709179125335, 0.009456218722393708, 0.00916500047763897, 0.005069863732625444, 0.011503011541926135, 0.003988175543749985],
      linf = [0.01188593784273051, 0.03638015998373141, 0.01568200398945724, 0.04666974730787579, 0.031235294705421968, 0.03316343064943483, 0.011539436992528018, 0.04896687646520839, 0.018714054039927555])
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl with flux_derigs_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [0.003755235939722358, 0.009062519246840721, 0.004096299856228109, 0.011429935838448906, 0.006897420817511043, 0.00900886245212482, 0.004926537542780259, 0.01153285554590683, 0.0037842060148666886],
      linf = [0.012982853115883541, 0.0320228076558316, 0.011575276754611022, 0.04425778643430531, 0.02478109022285846, 0.03198699034954189, 0.009761077061886558, 0.04433669321441455, 0.01618905441148782],
      volume_flux = (flux_derigs_etal, flux_nonconservative_powell))
  end

  @trixi_testset "elixir_mhd_alfven_wave_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_mortar.jl"),
      l2   = [0.001879021634926363, 0.007032724521848316, 0.0032793932234187325, 0.009056594733320348, 0.007514150120617965, 0.007328739509868727, 0.00309794018112387, 0.009026356949274878, 0.0035732583778049776],
      linf = [0.013734346970999622, 0.06173467158736011, 0.02183946452704291, 0.06258216169457917, 0.03672304497348122, 0.055120532123884625, 0.018202716205672487, 0.06133688282205586, 0.019888161885935608],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl with Orszag-Tang setup + flux_hll" begin
    # OBS! This setup does not make much sense and is only used to exercise all components of the
    # flux_hll implementation
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [0.004391143689111404, 0.04144737547475548, 0.041501307637678286, 0.04150353006408862, 0.03693135855995625, 0.021125605214031118, 0.03295607553556973, 0.03296235755245784, 7.16035229384135e-6],
      linf = [0.017894703320895378, 0.08486850681397005, 0.0891044523165206, 0.08492024792056754, 0.10448301878352373, 0.05381260695579509, 0.0884774018719996, 0.07784546966765199, 7.71609149516089e-5],
      initial_condition = function initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations3D)
        # The classical Orszag-Tang vortex test case adapted to 3D. Setup is taken from
        # Table 4 of the paper
        # - M. Bohm, A. R. Winters, G. J. Gassner, D. Derigs, F. Hindenlang, & J. Saur (2020)
        #   An entropy stable nodal discontinuous Galerkin method for the resistive MHD
        #   equations. Part I: Theory and numerical verification
        #   [doi: 10.1016/j.jcp.2018.06.027](https://doi.org/10.1016/j.jcp.2018.06.027)
        # Domain must be [0, 1]^3 , γ = 5/3
        rho = 25.0 / (36.0 * pi)
        v1 = -sin(2.0*pi*x[3])
        v2 =  sin(2.0*pi*x[1])
        v3 =  sin(2.0*pi*x[2])
        p = 5.0 / (12.0 * pi)
        B1 = -sin(2.0*pi*x[3]) / (4.0*pi)
        B2 =  sin(4.0*pi*x[1]) / (4.0*pi)
        B3 =  sin(4.0*pi*x[2]) / (4.0*pi)
        psi = 0.0
        return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
      end,
      surface_flux = (flux_hll, flux_nonconservative_powell),
      volume_flux  = (flux_central, flux_nonconservative_powell),
      coordinates_min = (0.0, 0.0, 0.0),
      coordinates_max = (1.0, 1.0, 1.0),
      initial_refinement_level=3,
      cfl = 1.1,
      tspan = (0.0, 0.06))
  end

  @trixi_testset "elixir_mhd_ec_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec_shockcapturing.jl"),
      l2   = [0.0186712969755079, 0.01620736832264799, 0.01620736832264803, 0.016207474382769683, 0.07306422729650594, 0.007355137041002365, 0.0073551370410023425, 0.00735520932001833, 0.000506140942330923],
      linf = [0.28040713666979633, 0.27212885844703694, 0.2721288584470349, 0.2837380205051839, 0.7915852408267114, 0.08770240288089526, 0.08770240288089792, 0.08773409387876674, 0.050221095224119834])
  end
end

end # module
