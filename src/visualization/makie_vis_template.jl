struct MyType end
@Makie.recipe(MyPlot, mytype) do scene
  Makie.Theme(;)
end
function Makie.plot!(myplot::MyPlot{Tuple{<:MyType}})
  x = LinRange(-1,1,100)
  Makie.lines!(myplot, sin.(pi*x))
  Makie.plot!(myplot, cos.(pi*x))
  myplot
end
Makie.plottype(::MyType) = MyPlot # makes plot(::MyType) = myplot(::MyType)


struct MyOtherType end
function Makie.plot(x::MyOtherType)
  fig = Makie.Figure()
  axes = [Axis(fig[i, j]) for j = 1:3, i = 1:2]
  for ax in axes
    myplot!(ax, MyType())
  end
  fig
end

# 1. create recipe for UnstructuredPlotDataSeries2D
# 2. use type conversion to overload Makie.plot(UnstructuredPlotDataSeries2D)
# 3. overload Makie.plot(UnstructuredPlotData2D) to create layout. check if this lets you use Makie.plot() too?
