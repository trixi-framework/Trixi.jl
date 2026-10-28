# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    Smagorinsky(; C_s = 0.17, Prandtl_turbulent = 0.9)

Smagorinsky subgrid-scale model for large eddy simulations (LES) with the
[`CompressibleLES3D`](@ref), e.g.,
```julia
equations_parabolic = CompressibleLES3D(equations; mu, Prandtl = 0.72,
                                        subgrid_model = Smagorinsky())
```
The eddy viscosity
```math
\mu_t = \rho (C_s \Delta)^2 |S|, \quad |S| = \sqrt{2 S_{ij} S_{ij}},
```
with the strain rate tensor ``S`` and the Smagorinsky constant ``C_s`` is added to the
dynamic viscosity in the viscous stresses, and the eddy conductivity
``\kappa_t = c_p \mu_t / \mathrm{Pr}_t`` with the turbulent Prandtl number
`Prandtl_turbulent` to the thermal conductivity. The filter width
``\Delta = V^{1/3} / (N + 1)`` is computed from the volume ``V`` of each element and the
polynomial degree ``N`` of the solver.

!!! note "Limitations"
    The model is only implemented for [`P4estMesh`](@ref)es in 3D on the CPU. The
    parabolic time step estimate ([`max_diffusivity`](@ref)) only accounts for the
    dynamic viscosity `mu`.

References:
- Joseph Smagorinsky (1963)
  General circulation experiments with the primitive equations: I. The basic experiment
  [DOI: 10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2](https://doi.org/10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2)
- Douglas K. Lilly (1967)
  The representation of small-scale turbulence in numerical simulation experiments
  Proceedings of the IBM Scientific Computing Symposium on Environmental Sciences, 195-210
"""
struct Smagorinsky{RealT <: Real} <: AbstractSubgridModel
    C_s::RealT
    Prandtl_turbulent::RealT
end

function Smagorinsky(; C_s = 0.17, Prandtl_turbulent = 0.9)
    RealT = promote_type(typeof(C_s), typeof(Prandtl_turbulent))
    return Smagorinsky(convert(RealT, C_s), convert(RealT, Prandtl_turbulent))
end
end # @muladd
