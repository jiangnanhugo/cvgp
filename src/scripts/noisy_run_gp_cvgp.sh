#!/bin/bash -l

basepath=.
metric=neg_mse
noise_std=0.1
thispath=$basepath/src/ctrl_var_gp

nvar=5
nt=58
for dataname in inv sincos sincosinv; do
	datasource=${dataname}_nv${nvar}_nt${nt}

	for pgn in {0..9}; do
		prog=prog_$pgn.data
		echo "submit $prog"
		dump_dir=$basepath/result/noisy${noise_std}_$datasource/$(date +%F)
		if [ ! -d "$dump_dir" ]; then
			echo "create dir: $dump_dir"
			mkdir -p $dump_dir
		fi
		log_dir=$basepath/log/$(date +%F)
		if [ ! -d "$log_dir" ]; then
			echo "create dir: $log_dir"
			mkdir -p $log_dir
		fi
		python3 $thispath/main.py $nvar \
			$basepath/data/$datasource/$prog $metric --noise_std ${noise_std} \
			>$dump_dir/$prog.metric_${metric}.gp.out
		python3 $thispath/main.py $nvar \
			$basepath/data/$datasource/$prog $metric --expand_gp --noise_std ${noise_std} \
			>$dump_dir/$prog.metric_${metric}.egp.out
	done
done
