#!/bin/bash -l

basepath=.
operators=sincos
dates=2023-01-13
datasource=${operators}_nv5_nt58

metric=neg_mse
python $basepath/result/parse_results.py --fp $basepath/result/$datasource/$dates/prog_ \
--metric $metric \
--true_program_file $basepath/data/$datasource/prog_ \
--dso_basepath $basepath/dso_classic \
--noise_std 0.0
