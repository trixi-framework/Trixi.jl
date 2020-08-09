# Overview of how to reproduce results in Euler-gravity paper

## Sec. 4.1.2, Table 3, EOC tests hyperbolic diffusion
**N = 3**:
```julia
Trixi.convtest("examples/euler_gravity_paper/parameters_hyp_diff_nonperiodic.toml", 4)
```

**N = 4**:
```julia
Trixi.convtest("examples/euler_gravity_paper/parameters_hyp_diff_nonperiodic.toml", 4, N=4)
```
