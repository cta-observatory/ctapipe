set terminal png notransparent nocrop enhanced size 800,600 font "arial,10"
set output '/tmp/analysis/4-PMAP/gamma_20deg_0deg_run9408___cta-prod3-merged_desert-2150m-Paranal-subarray-4_cone10_y.png'
unset key
set xlabel "x"
set ylabel "y"
plot "tmpPHistogram.txt" using 1:2 title '' with steps
