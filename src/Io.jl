module IoMod

using ..Jul1dge
using ..DgMod
using Plots

export plot2file

function plot2file(dg, filename)
  x = dg.nodecoordinate[:]
  y = zeros(length(x), nvars(dg))
  nnodes = polydeg(dg) + 1
  for v = 1:nvars(dg)
    for c in 1:dg.ncells
      for i = 1:nnodes
        y[(c - 1) * nnodes + i, v] = dg.u[v, i, c]
      end
    end
    plot(x, y)
  end
  savefig(filename)
end

end
