# Overview of how to reproduce results in Euler-gravity paper

Run the following commands from the Julia REPL. The code used in the
[paper](https://arxiv.org/abs/2008.10593)
was written for Trixi.jl `v0.2`. Since breaking changes were introduced
in Trixi.jl `v0.3`, the code provided below does not reproduce the examples
exactly. Complete instructions how to reproduce the numerical experiments
using an older version of Trixi.jl are available at
https://doi.org/10.5281/zenodo.3996575.

## Sec. 4.1.1, Table 2, EOC tests compressible Euler
**polydeg = 3**:
```julia
julia> using Trixi

julia> convergence_test("examples/paper_self_gravitating_gas_dynamics/elixir_euler_convergence.jl", 4)
```

**polydeg = 4**:
```julia
julia> using Trixi

julia> convergence_test("examples/paper_self_gravitating_gas_dynamics/elixir_euler_convergence.jl", 4, polydeg=4)
```

## Sec. 4.1.2, Table 3, EOC tests hyperbolic diffusion
**polydeg = 3**:
```julia
julia> using Trixi

julia> convergence_test("examples/paper_self_gravitating_gas_dynamics/elixir_hypdiff_convergence.jl", 4)
```

**polydeg = 4**:
```julia
julia> using Trixi

julia> convergence_test("examples/paper_self_gravitating_gas_dynamics/elixir_hypdiff_convergence.jl", 4, polydeg=4)
```

## Sec. 4.1.3, Table 4, EOC tests coupled Euler-gravity
**polydeg = 3**:
```julia
julia> using Trixi

julia> convergence_test("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_convergence.jl", 4)
```

**polydeg = 4**:
```julia
julia> using Trixi

julia> convergence_test("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_convergence.jl", 4, polydeg=4)
```

## Sec. 4.1.3, Table 5, EOC tests coupled Euler-gravity (update gravity once per step)
This is only available in Trixi.jl `v0.2`.

## Sec. 4.2.1, Figures 3 + 5a, Jeans energies with Euler/CK45 and gravity/CK45
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_jeans_instability.jl")
```

## Sec. 4.2.1, Figure 4, Jeans energies with Euler/CK45 and gravity/CK45 (update gravity once per step)
This is only available in Trixi.jl `v0.2`.

## Sec. 4.2.1, Figure 5b, Jeans energies with Euler/CK45 and gravity/RK3S*
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_jeans_instability.jl",
                      parameters=ParametersEulerGravity(background_density=1.5e7,
                                                        gravitational_constant=6.674e-8,
                                                        cfl=2.4,
                                                        resid_tol=1.0e-4,
                                                        n_iterations_max=1000,
                                                        timestep_gravity=timestep_gravity_erk52_3Sstar!))
```

## Sec. 4.2.1, Creating Jeans energies figures 3 and 4
One must also shrink the analysis interval in the above command, e.g.,
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_jeans_instability.jl",
                     analysis_interval=1)
```
to generate necessary data for the plots to look nice. Then run the python
script with the analysis file from the run as input
```
./jeans_all_in_one.py analysis.dat
```
to generate the figure.

## Sec. 4.2.2, Figure 6, T=0.5, AMR meshes for Sedov + gravity
**T = 0.0 and T = 0.5:**
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_sedov_blast_wave.jl", tspan=(0.0, 0.5))
```

**T = 1.0:**
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_sedov_blast_wave.jl")
```

## Sec. 4.2.2, Figure 7a, T=0.5, Sedov + gravity with Euler/CK45 and gravity/RK3S*
**AMR mesh:**
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_sedov_blast_wave.jl", tspan=(0.0, 0.5))
```

**Uniform mesh:**
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_sedov_blast_wave.jl",
                     amr_callback=TrivialCallback(), initial_refinement_level=8, t_end=0.5)
```

## Sec. 4.2.2, Figure 7b, T=1.0, Sedov + gravity with Euler/CK45 and gravity/RK3S*
**AMR mesh:**
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_sedov_blast_wave.jl")
```

**Uniform mesh:**
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_sedov_blast_wave.jl",
                     amr_callback=TrivialCallback(), initial_refinement_level=8)
```

## Sec. 4.2.2, Table 6, Sedov + gravity, performance uniform vs. AMR
**AMR mesh:**
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_sedov_blast_wave.jl")
```

**Uniform mesh:**
```julia
julia> using Trixi

julia> trixi_include("examples/paper_self_gravitating_gas_dynamics/elixir_eulergravity_sedov_blast_wave.jl",
                     amr_callback=TrivialCallback(), initial_refinement_level=8)
```

### Postprocessing
To postprocess the solution files use [`Trixi2Vtk`](https://github.com/trixi-framework/Trixi2Vtk.jl)
with `nvisnodes=16` and `format=:vti`. Then use Paraview.
