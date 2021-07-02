const DGFluxDiff{Dim, Elem} = DG{<:RefElemData{Dim, Elem}, 
                            M, S, <: VolumeIntegralFluxDifferencing} where {Dim, Elem, M, S}
const PolyDGFluxDiff{Dim, Elem} = DG{<:RefElemData{Dim, Elem, Polynomial}, 
                            M, S, <: VolumeIntegralFluxDifferencing} where {Dim, Elem, M, S}
const SBPDGFluxDiff{Dim, Elem} = DG{<:RefElemData{Dim, Elem, <:SBP}, 
                            M, S, <: VolumeIntegralFluxDifferencing} where {Dim, Elem, M, S}

"""
  function hadamard_sum_ATr!(du, ATr, volume_flux, u, skip_index=(i,j)->false)

Computes the flux difference ∑_j Aij * f(u_i, u_j) and accumulates the result into `du`. 
- `du`, `u` are vectors
- `ATr` is the transpose of the flux differencing matrix `A`. The transpose is used for 
faster traversal since matrices are column major in Julia. 
"""
function hadamard_sum_ATr!(du, ATr, volume_flux::VF, u, skip_index=(i,j)->false) where {VF}
  for i in axes(ATr,2)
    ui = u[i]
    val_i = du[i]
    for j in axes(ATr,1)
      if skip_index(i,j) != true
        val_i += ATr[j,i] * volume_flux(ui, u[j])
      end
    end
    du[i] = val_i
  end
end

function calc_volume_integral!(du, u::StructArray, volume_integral::VolumeIntegralFluxDifferencing,
                 mesh::VertexMappedMesh, equations, dg::DG, cache) where {DG <: SBPDGFluxDiff}

  rd = dg.basis  
  @unpack local_values_threaded = cache

  volume_flux_oriented(i) = let i=i, equations=equations 
    (u_ll, u_rr)->volume_integral.volume_flux(u_ll, u_rr, i, equations)
  end

  # Todo: simplices. Dispatch on curved/non-curved mesh types, this code only works for affine meshes (accessing rxJ[1,e],...)
  @threaded for e in eachelement(mesh, dg, cache)
    u_local = view(u, :, e)
    rhs_local = local_values_threaded[Threads.threadid()]
    fill!(rhs_local, zero(eltype(rhs_local)))
    for i in eachdim(mesh) 
      Qi_skew_Tr = build_lazy_physical_derivative(e, i, mesh, dg, cache)
      hadamard_sum_ATr!(rhs_local, Qi_skew_Tr, volume_flux_oriented(i), u_local)
    end
    view(du, :, e) .+= rhs_local ./ rd.wq
  end
end

function build_lazy_physical_derivative(elem::Int, orientation::Int, mesh::AbstractMeshData{2}, dg, cache) 
  @unpack Qrst_skew_Tr = cache
  @unpack rxJ, sxJ, ryJ, syJ = mesh.md
  QrskewTr, QsskewTr = Qrst_skew_Tr
  if orientation == 1
    return LazyArray(@~ @. 2 * (rxJ[1,elem]*QrskewTr + sxJ[1,elem]*QsskewTr))
  else
    return LazyArray(@~ @. 2 * (ryJ[1,elem]*QrskewTr + syJ[1,elem]*QsskewTr))
  end
end

function build_lazy_physical_derivative(elem::Int, orientation::Int, mesh::AbstractMeshData{3}, dg, cache) 
  @unpack Qrst_skew_Tr = cache
  @unpack rxJ, sxJ, txJ, ryJ, syJ, tyJ, rzJ, szJ, tzJ = mesh.md
  QrskewTr, QsskewTr, QtskewTr = Qrst_skew_Tr
  if orientation == 1
    return LazyArray(@~ @. 2 * (rxJ[1,elem]*QrskewTr + sxJ[1,elem]*QsskewTr) + txJ[1,elem]*QtskewTr)
  elseif orientation == 2
    return LazyArray(@~ @. 2 * (ryJ[1,elem]*QrskewTr + syJ[1,elem]*QsskewTr) + tyJ[1,elem]*QtskewTr)
  else 
    return LazyArray(@~ @. 2 * (rzJ[1,elem]*QrskewTr + szJ[1,elem]*QsskewTr) + tzJ[1,elem]*QtskewTr)
  end
end