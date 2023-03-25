#!/usr/bin/bash
nvar=5
nt=58
basepath=~/Desktop/CVGP_code_implementation
datasource=inv_nv${nvar}_nt${nt}

for metric in inv_nrmse; do
	for pgn in {0..9}; do
		echo "submit $pgn"
		dump_dir=$basepath/result/$datasource/$(date +%F)
		if [ ! -d "$dump_dir" ]; then
			echo "create output dir: $dump_dir"
			mkdir -p $dump_dir
		fi
		log_dir=$basepath/log/$(date +%F)
		if [ ! -d "$log_dir" ]; then
			echo "create log dir: $log_dir"
			mkdir -p $log_dir
		fi
		for bsl in DSR PQT VPG GPMELD; do
			echo $bsl
			python3 -m dso.run $basepath/dso/dataset/$datasource/prog_${pgn}_${bsl}_${metric}.json >$dump_dir/prog_$pgn.data.metric_${metric}.${bsl}.out

		done

	done
done
