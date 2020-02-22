module Mortar

include("interpolation.jl")
using .Interpolation


function main()
  N = 4
  nnodes = N + 1
  nodes, weights = gauss_lobatto_nodes_weights(nnodes)
  wbary = barycentric_weights(nodes)

  # Small to large -> projection
  pl2r_upper = zeros(nnodes, nnodes)
  for j in 1:nnodes
    poly = lagrange_interpolating_polynomials(1/2 * (nodes[j] + 1), nodes, wbary)
    for i in 1:nnodes
      pl2r_upper[i, j] = 1/2 * poly[i] * weights[j]/weights[i]
    end
  end

  pl2r_lower = zeros(nnodes, nnodes)
  for j in 1:nnodes
    poly = lagrange_interpolating_polynomials(1/2 * (nodes[j] - 1), nodes, wbary)
    for i in 1:nnodes
      pl2r_lower[i, j] = 1/2 * poly[i] * weights[j]/weights[i]
    end
  end

  # Large to small -> interpolation
  pr2l_upper = zeros(nnodes, nnodes)
  for j in 1:nnodes
    poly = lagrange_interpolating_polynomials(1/2 * (nodes[j] + 1), nodes, wbary)
    for i in 1:nnodes
      pr2l_upper[j, i] = poly[i]
    end
  end

  pr2l_lower = zeros(nnodes, nnodes)
  for j in 1:nnodes
    poly = lagrange_interpolating_polynomials(1/2 * (nodes[j] - 1), nodes, wbary)
    for i in 1:nnodes
      pr2l_lower[j, i] = poly[i]
    end
  end

  # Checks
  println()
  println("-"^80)
  println("Equation 18a")
  for i = 1:nnodes
    s_upper = 0.0
    s_lower = 0.0
    for j = 1:nnodes
      s_upper += pr2l_upper[i, j]
      s_lower += pr2l_lower[i, j]
    end
    println("i = $i: sum upper = $s_upper; sum lower = $s_lower")
  end
  println("-"^80)

  println()
  println("-"^80)
  println("Equation 18b")
  for i = 1:nnodes
    s = 0.0
    for j = 1:nnodes
      s += pl2r_upper[i, j]
      s += pl2r_lower[i, j]
    end
    println("i = $i: sum = $s")
  end
  println("-"^80)

  println()
  println("-"^80)
  println("Equation 6 (M-compatibility)")
  for i = 1:nnodes
    for j = 1:nnodes
      res_upper = pr2l_upper[j, i] * weights[j] - 2 * pl2r_upper[i, j] * weights[i]
      res_lower = pr2l_lower[j, i] * weights[j] - 2 * pl2r_lower[i, j] * weights[i]
      println("(i,j) = ($i,$j): res_upper = $res_upper, res_lower = $res_lower")
    end
  end
  println("-"^80)

  println()
  println("-"^80)
  println("pr2l_upper")
  for i = 1:nnodes
    for j = 1:nnodes
      println("pr2l_upper($i,$j) = $(pr2l_upper[i,j])")
    end
  end
  println()
  println("pr2l_lower")
  for i = 1:nnodes
    for j = 1:nnodes
      println("pr2l_lower($i,$j) = $(pr2l_lower[i,j])")
    end
  end
  println()
  println("pl2r_upper")
  for i = 1:nnodes
    for j = 1:nnodes
      println("pl2r_upper($i,$j) = $(pl2r_upper[i,j])")
    end
  end
  println()
  println("pl2r_lower")
  for i = 1:nnodes
    for j = 1:nnodes
      println("pl2r_lower($i,$j) = $(pl2r_lower[i,j])")
    end
  end
  println("-"^80)
end

main()

end # module Mortar
