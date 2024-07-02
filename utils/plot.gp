# set term pdfcairo color enhanced font ",8" fontscale 1.0 lw 0.5 size 7cm,5cm 
set term pdfcairo color enhanced font ",8" fontscale 1.0 lw 0.5 size 14cm,10cm 

# Set defaults
if (!exists("infile")) infile="solution_000000000.txt"
if (!exists("eqn")) eqn="linear_scalar_advection"
ext = ".pdf"

# Legend
set grid

# Axes and labels
set ylabel ""
set xlabel "x"

# Line styles
set style line 2 lw 3 lc rgb "black" 
set style line 3 lw 3 lc rgb "black" dt 3 pt 4 ps 0.6
set style line 4 lw 3 lc rgb "black" dt (12,12) pt 2 ps 0.6
set style line 5 lw 3 lc rgb "black" dt (6,9) pt 6 ps 0.6
set style line 6 lw 3 lc rgb "black" dt (18,6,3,6) pt 8 ps 0.6
set style line 7 lw 3 lc rgb "black" dt (9,8,3,8) pt 10 ps 0.6

# Skip first line in data file...
set key autotitle columnhead

set out infile.ext
if (eqn eq "linear_scalar_advection") {
  plot infile u 1:2 w l lw 3
}
if (eqn eq "euler") {
  plot infile u 1:2 w l lw 3, \
       ''     u 1:3 w l lw 3, \
       ''     u 1:4 w l lw 3
}
