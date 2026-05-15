using Trixi

# use a simple 1D advection setup
trixi_include(joinpath(examples_dir(), "tree_1d_dgsem/elixir_advection_basic.jl"),
              initial_refinement_level = 3)

# Test direct constructor
local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1e-6,),
                                                     variables = ((u, equations) -> u[1],))
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi)

# run once to initialize the cell averages
global_limiter!(sol.u[end], nothing, semi, 0.0)
global_limiter!.cell_averages[5] = SVector(-0.1) # violate positivity

u = Trixi.wrap_array(sol.u[end], semi)

Trixi.global_cell_average_limiter!(u, global_limiter!.cell_averages,
                                   global_limiter!.davis_yin_Z,
                                   global_limiter!.projected_cell_averages,
                                   global_limiter!.pseudo_inverse_cell_volumes_vector,
                                   local_limiter!.thresholds, # TODO: generalize to multiple variables
                                   global_limiter!.global_limiter_tol,
                                   global_limiter!.max_davis_yin_iterations,
                                   Trixi.mesh_equations_solver_cache(semi)...)

global_limiter!.cell_averages


using Test
reference_cell_averages = SVector.([1.1722191699202247
                                    1.4358800215443228
                                    1.4358398499238705
                                    1.1721221870493048
                                    1.0e-6
                                    0.5355484115711128
                                    0.5355885831915655
                                    0.7993062460661308])

@test isapprox(global_limiter!.cell_averages, reference_cell_averages; rtol = 1e-7)


using BenchmarkTools
@btime Trixi.global_cell_average_limiter!($u,$(global_limiter!.cell_averages),
                                          $(global_limiter!.davis_yin_z),
                                          $(global_limiter!.projected_cell_averages),
                                          $(global_limiter!.pseudo_inverse_cell_volumes_vector),
                                          $(local_limiter!.thresholds), # TODO: generalize to multiple variables
                                          $(global_limiter!.global_limiter_tol),
                                          $(global_limiter!.max_davis_yin_iterations),
                                          Trixi.mesh_equations_solver_cache(semi)...)