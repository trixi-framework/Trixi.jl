using MAT

parsevec(type, str) = str |>
  (x -> split(x, ", ")) |>
  (x -> map(y -> parse(type, y), x))
  
"node_type = :kubatko or :hicken"
function sbp_diagE_nodes(elem::Tri,N;quadrature_strength=2*N-1)
    if quadrature_strength==2*N-1
        vars = matread("./sbp_nodes/KubatkoQuadratureRules.mat");
        rs = vars["Q_GaussLobatto"][N]["Points"]
        r,s = (rs[:,i] for i = 1:size(rs,2))
        w = vec(vars["Q_GaussLobatto"][N]["Weights"])
    elseif quadrature_strength==2*N
        lines = readlines("sbp_nodes/tri_diage_p$N.dat")
        r = parsevec(Float64,lines[11])
        s = parsevec(Float64,lines[12])
        w = parsevec(Float64,lines[13])
    else
        error("No nodes found for N=$N and quadrature_strength = $quadrature_strength")
    end
    return r,s,w
end


