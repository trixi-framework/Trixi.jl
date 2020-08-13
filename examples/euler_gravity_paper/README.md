# Overview of how to reproduce results in Euler-gravity paper

## Sec. 4.1.1, Table 2, EOC tests compressible Euler
**N = 3**:
```julia
Trixi.convtest("examples/euler_gravity_paper/parameters_no_gravity_manufac.toml", 4)
```

**N = 4**:
```julia
Trixi.convtest("examples/euler_gravity_paper/parameters_no_gravity_manufac.toml", 4, N=4)
```

## Sec. 4.1.2, Table 3, EOC tests hyperbolic diffusion
**N = 3**:
```julia
Trixi.convtest("examples/euler_gravity_paper/parameters_hyp_diff_nonperiodic.toml", 4)
```

**N = 4**:
```julia
Trixi.convtest("examples/euler_gravity_paper/parameters_hyp_diff_nonperiodic.toml", 4, N=4)
```

## Sec. 4.1.3, Table 4, EOC tests coupled Euler-gravity
**N = 3**:
```julia
Trixi.convtest("examples/euler_gravity_paper/parameters_coupling_convergence_test.toml", 4)
```

**N = 4**:
```julia
Trixi.convtest("examples/euler_gravity_paper/parameters_coupling_convergence_test.toml", 4, N=4)
```

## Sec. 4.1.3, Table 5, EOC tests coupled Euler-gravity (update gravity once per step)
```julia
Trixi.convtest("examples/euler_gravity_paper/parameters_coupling_convergence_test.toml", 4,
               update_gravity_once_per_stage=false)
```

## Sec. 4.2.1, Figures 3 + 5a, Jeans energies with Euler/CK45 and gravity/CK45
```julia
Trixi.run("examples/euler_gravity_paper/parameters_jeans_instability.toml")
```

## Sec. 4.2.1, Figure 4, Jeans energies with Euler/CK45 and gravity/CK45 (update gravity once per step)
```julia
Trixi.run("examples/euler_gravity_paper/parameters_jeans_instability.toml",
          update_gravity_once_per_stage=false)
```

## Sec. 4.2.1, Figure 5b, Jeans energies with Euler/CK45 and gravity/RK3S*
```julia
Trixi.run("examples/euler_gravity_paper/parameters_jeans_instability.toml",
          time_integration_scheme_gravity="timestep_gravity_erk52_3Sstar!", cfl_gravity=1.2)
```

### Creating Jeans energies figures
One must also include shrink the analysis interval in the above command, e.g.,
```julia
Trixi.run("examples/euler_gravity_paper/parameters_jeans_instability.toml",
          analysis_interval=1)
```
to generate necessary data for the plots to look nice. Then run the python script with the analysis file from the run as input
```
./jeans_all_in_one.py analysis.dat
```
to generate the figure
