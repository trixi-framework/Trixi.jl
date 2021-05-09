"""
    struct ModalESDG{F1,F2,F3} 
        rd::RefElemData
        volume_flux::F1 
        interface_flux::F2
        interface_dissipation::F3
    end

    `volume_flux`,`interface_flux`,`interface_dissipation` expect arguments `flux(orientation)(u_ll,u_rr)`. 
    Convenience constructors using Trixi's flux interfaces are provided. 

    Note: requires `entropy2cons` and `cons2entropy` to be defined for the entropy projection. 

    Example: 
    ```julia
    equations = CompressibleEulerEquations2D(1.4)
    @inline volume_flux(orientation) = let equations = equations
        (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,equations)
    end
    ```
"""
struct ModalESDG{DIM,ElemType,F1,F2,F3} 
    rd::RefElemData{DIM,ElemType} 
    volume_flux::F1 
    interface_flux::F2
    interface_dissipation::F3
end

"""
    function ModalESDG(rd::RefElemData,
        trixi_volume_flux::F1,
        trixi_interface_flux::F2,
        trixi_interface_dissipation::F3,
        equations) where {F1,F2,F3}
    
Initialize a ModalESDG solver with Trixi fluxes as arguments, where trixi_*_flux has the form of
    trixi_*_flux(u_ll,u_rr,orientation,equations)

"""
function ModalESDG(rd::RefElemData,
                   trixi_volume_flux::F1,
                   trixi_interface_flux::F2,
                   trixi_interface_dissipation::F3,
                   equations) where {F1,F2,F3}

    volume_flux, interface_flux, interface_dissipation = let equations=equations
        volume_flux(orientation) = (u_ll,u_rr)->trixi_volume_flux(u_ll,u_rr,orientation,equations)
        interface_flux(orientation) = (u_ll,u_rr)->trixi_interface_flux(u_ll,u_rr,orientation,equations)
        interface_dissipation(orientation) = (u_ll,u_rr)->trixi_interface_dissipation(u_ll,u_rr,orientation,equations)
        volume_flux,interface_flux,interface_dissipation
    end
    return ModalESDG(rd,volume_flux,interface_flux,interface_dissipation)
end

function Base.show(io::IO, solver::ModalESDG{DIM}) where {DIM}
    println("Modal ESDG solver in $DIM dimension with ")
    println("   volume flux           = $(solver.volume_flux.trixi_volume_flux)")
    println("   interface flux        = $(solver.interface_flux.trixi_interface_flux)")    
    println("   interface dissipation = $(solver.interface_dissipation.trixi_interface_dissipation)")        
end

Base.real(solver::ModalESDG) = Float64 # is this for DiffEq.jl?
Trixi.ndofs(mesh::UnstructuredMesh, solver::ModalESDG, cache) = length(solver.rd.r)*cache.md.K

