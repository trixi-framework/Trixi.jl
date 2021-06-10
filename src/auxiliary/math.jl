# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See TODO: link-to-my-blog-post
@muladd begin


# Computes the logarithmic mean: (aR-aL)/(LOG(aR)-LOG(aL)) = (aR-aL)/LOG(aR/aL)
# Problem: if aL~= aR, then 0/0, but should tend to --> 0.5*(aR+aL)
#
# introduce xi=aR/aL and f=(aR-aL)/(aR+aL) = (xi-1)/(xi+1)
# => xi=(1+f)/(1-f)
# => Log(xi) = log(1+f)-log(1-f), and for small f (f^2<1.0E-02) :
#
#    Log(xi) ~=     (f - 1/2 f^2 + 1/3 f^3 - 1/4 f^4 + 1/5 f^5 - 1/6 f^6 + 1/7 f^7)
#                  +(f + 1/2 f^2 + 1/3 f^3 + 1/4 f^4 + 1/5 f^5 + 1/6 f^6 + 1/7 f^7)
#             = 2*f*(1           + 1/3 f^2           + 1/5 f^4           + 1/7 f^6)
#  (aR-aL)/Log(xi) = (aR+aL)*f/(2*f*(1 + 1/3 f^2 + 1/5 f^4 + 1/7 f^6)) = (aR+aL)/(2 + 2/3 f^2 + 2/5 f^4 + 2/7 f^6)
#  (aR-aL)/Log(xi) = 0.5*(aR+aL)*(105/ (105+35 f^2+ 21 f^4 + 15 f^6)
@inline function ln_mean(value1, value2)
  epsilon_f2 = 1.0e-4
  ratio = value2 / value1
  # f2 = f^2
  f2 = (ratio * (ratio - 2) + 1) / (ratio * (ratio + 2) + 1)
  if f2 < epsilon_f2
    return (value1 + value2) * 52.5 / (105 + f2 * (35 + f2 * (21 + f2 * 15)))
  else
    return (value2 - value1) / log(ratio)
  end
end


end # @muladd
