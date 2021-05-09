include("flux_differencing.jl")

function Trixi.allocate_coefficients(mesh::UnstructuredMesh,
                                     equations, solver::NodalESDG, cache)
    @unpack md = cache
    nvars = nvariables(equations) 
    return StructArray([SVector{nvars}(zeros(nvars)) for i in axes(md.xq,1), j in axes(md.xq,2)])
end

function Trixi.compute_coefficients!(u::StructArray, initial_condition, t,
                                     mesh::UnstructuredMesh, equations, solver::NodalESDG, cache)
    # @show typeof(u), size(u)
    # @show length(cache.md.xq)
    for i in eachindex(u) # loop over quadrature (sbp) nodes
        xyz_i = getindex.(cache.md.xyzq,i)
        u[i] = initial_condition(xyz_i,t,equations) # interpolate - add projection later?
    end
end

# ======================= 2D rhs codes =============================

function Trixi.create_cache(mesh::UnstructuredMesh{2}, equations, solver::NodalESDG, RealT, uEltype)

    @unpack VXYZ,EToV = mesh
    @unpack rd = solver
    md = MeshData(VXYZ...,EToV,rd)
    md = make_periodic(md,rd)

    @unpack sbp_operators = solver
    Qr,Qs = sbp_operators.Qrst
    QrskewTr = -Matrix(.5*(Qr-Qr'))
    QsskewTr = -Matrix(.5*(Qs-Qs'))
    invm = 1 ./ sbp_operators.wq

    # face storage
    nvars = nvariables(equations)
    Uf = StructArray{SVector{nvars,Float64}}(ntuple(_->similar(md.xq),nvars))

    # tmp cache for threading
    structarray_zeros = StructArray{SVector{nvars,Float64}}(ntuple(_->similar(md.xq[:,1]),nvars))
    rhse_threads = [structarray_zeros for _ in 1:Threads.nthreads()]

    cache = (;md,
             QrskewTr,QsskewTr,invm,Uf,
             rhse_threads)

    return cache
end


function Trixi.rhs!(dQ, Q::StructArray, t, mesh::UnstructuredMesh{2}, equations,
                    initial_condition, boundary_conditions, source_terms,
                    solver::NodalESDG, cache)

    @unpack md = cache
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    @unpack volume_flux, interface_flux, interface_dissipation = solver
    @unpack sbp_operators = solver
    @unpack Fmask,wf = sbp_operators
    @unpack QrskewTr,QsskewTr,invm,Uf = cache

    @timeit Trixi.timer() "extract Uf" begin
        for e = 1:size(Q,2)
            for (i,vol_id) in enumerate(Fmask)
                Uf[i,e] = Q[vol_id,e]
            end
        end
        #Uf .= Q[Fmask,:]
    end

    #@batch 
    for e = 1:md.K
        rhse = cache.rhse_threads[Threads.threadid()]

        fill!(rhse,zero(eltype(rhse)))
        
        Ue = view(Q,:,e)   
        QxTr = LazyArray(@~ @. 2 *(rxJ[1,e]*QrskewTr + sxJ[1,e]*QsskewTr))
        QyTr = LazyArray(@~ @. 2 *(ryJ[1,e]*QrskewTr + syJ[1,e]*QsskewTr))

        hadsum_ATr!(rhse, QxTr, volume_flux(1), Ue) 
        hadsum_ATr!(rhse, QyTr, volume_flux(2), Ue) 

        for (i,vol_id) = enumerate(Fmask)
            UM, UP = Uf[i,e], Uf[mapP[i,e]]
            Fx = interface_flux(1)(UP,UM)
            Fy = interface_flux(2)(UP,UM)
            normal = SVector{2}(nxJ[i,e],nyJ[i,e])/sJ[i,e]
            diss = interface_dissipation(normal)(UM,UP)
            val = (Fx * nxJ[i,e] + Fy * nyJ[i,e] + diss*sJ[i,e]) * wf[i]
            rhse[vol_id] += val
        end

        @timeit Trixi.timer() "broadcast + store" begin
            @. rhse = -rhse / J[1,e] # split up broadcasts into short statements
            @. dQ[:,e] = invm * rhse 
        end
    end

    return nothing
end

## ==============================================================================

function calc_source_terms(du::StructArray,u,t,source_terms::Nothing,
                           equations,solver::NodalESDG,cache)
    return nothing
end

"""
    function calc_source_terms(du::StructArray,u,t,source_terms,
                               equations,solver::ModalESDG,cache,element_index)

Adds source terms on an element. Independent of dimension or equation.
"""
function calc_source_terms(du::StructArray,u,t,source_terms,
                           equations,solver::NodalESDG,cache)
    @unpack md = cache
    @unpack rd = solver

    # add source terms to quadrature points
    for e = 1:md.K
        for i = 1:length(rd.rq)
            du[i,e] += source_terms(u[i],getindex.(md.xyzq,i,e),t,equations)
        end
    end
end