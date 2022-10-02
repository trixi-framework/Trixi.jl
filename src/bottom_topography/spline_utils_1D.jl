# Helpfunctions for 1D spline interpolation

# Sorting the inputs that the x values are ascending
function sort_data(x::Vector{Float64},y::Vector{Float64})
      
  original_data = hcat(x,y)
  sorted_data = original_data[sortperm(original_data[:,1]), :]
  
  x_sorted = sorted_data[:,1]
  y_sorted = sorted_data[:,2]
  
  return x_sorted,y_sorted
end

# Cubic spline smoothing
# Based on https://en.wikipedia.org/wiki/Smoothing_spline
function spline_smoothing(lambda, h, y)

  n = length(y)

  h_vec = repeat([h], n-2)

  Delta_ii   =  1 ./ h_vec
  Delta_iip1 = -2 ./ h_vec
  Delta_iip2 =  1 ./ h_vec
  
  Delta             =  zeros(n-2, n)
  Delta[:, 1:(n-2)] =  diagm(Delta_ii)
  Delta[:, 2:(n-1)] += diagm(Delta_iip1)
  Delta[:, 3: n   ] += diagm(Delta_iip2)

  W_im1i =  h_vec[1:end-1] ./ 6
  W_ii   = (2*h_vec) ./ 3
  W      = SymTridiagonal(W_ii, W_im1i)

  K = transpose(Delta) * inv(W) * Delta

  return inv(diagm(ones(n)) + lambda*K) * y
end