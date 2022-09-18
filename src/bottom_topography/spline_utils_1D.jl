# Helpfunctions for 1D spline interpolation

# Sorting the inputs that the x values are ascending
function sort_data(x::Vector{Float64},y::Vector{Float64})
      
  orig_data = hcat(x,y)
  sort_data = orig_data[sortperm(orig_data[:,1]), :]
  
  u = sort_data[:,1]
  t = sort_data[:,2]
  
  return u,t
end

# Cubic spline smoothing
# Based on https://en.wikipedia.org/wiki/Smoothing_spline
function spline_smoothing(smoothing_factor, h, y)

  n = length(y)

  h_vec = repeat([h], n-1)

  delta_ii   =  1 ./ h_vec[1:end-1]
  delta_iip1 = -1 ./ h_vec[1:end-1] - 1 ./ h_vec[2:end]
  delta_iip2 =  1 ./ h_vec[2:end  ]
  
  delta             =  zeros(n-2, n)
  delta[:, 1:(n-2)] =  diagm(delta_ii)
  delta[:, 2:(n-1)] += diagm(delta_iip1)
  delta[:, 3: n   ] += diagm(delta_iip2)

  W_im1i =  h_vec[2:end-1]             ./ 6
  W_ii   = (h_vec[1:end-1] + h_vec[2:end]) ./ 3
  W      = SymTridiagonal(W_ii, W_im1i)

  a = transpose(delta) * inv(W) * delta

  return inv(diagm(ones(n)) + smoothing_factor*a) * y
end