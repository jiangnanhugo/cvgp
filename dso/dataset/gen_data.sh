#!/usr/bin/zsh
nvar=5
nt=58
seed=42
basepath=./
nsamples=50000

noise_std=0.0
for dataset_prefix in inv sincos sincosinv; do
	datasource=${dataset_prefix}_nv${nvar}_nt${nt}

	for pgn in {0..9}; do
		for bsl in DSR PQT VPG GPMELD; do
			echo "submit $pgn"
			output_csv_dir=$basepath/dso/dataset/$datasource
			if [ ! -d "$output_csv_dir" ]; then
				echo "create dir: $output_csv_dir"
				mkdir -p $output_csv_dir
			fi
			python3 $basepath/dso/dataset/dataset_config_generator.py $basepath/dso/config/config_regression_${dataset_prefix}_${bsl}.json \
				$basepath/data/$datasource/prog_$pgn.data \
				$output_csv_dir/prog_$pgn \
				$seed \
				$bsl \
				$nsamples $noise_std
		done
	done
done

