# README： Symbolic Regression via Control Variable Genetic Programming #

This is the code implemetation of paper accepted at ECML-PKDD 2023 with title: [
Symbolic Regression via Control Variable Genetic Programming](https://link.springer.com/chapter/10.1007/978-3-031-43421-1_11).

## 0. Prerequisites

### 0.1 Dependency packages
```bash
pip install deap
```


### 0.2 Directory 

- data: the generated dataset. Every file represent a ground-truth expression.
- dso: public code implementation from https://github.com/brendenpetersen/deep-symbolic-optimization.
- plots: the jupter notebook to generate our figures in the paper.
- result: contains all the output of all the programs, the training logs.
- src: the inplemenattion of the our proposed control variable genetic programming algorithm and the classic genetic programming algorithm.


## 1. Run Control Variable Generic Programming (CVGP) and generic prgoramming (GP)

### 1.1 configure the `basepath`
Run the **CVGP, GP** model on the **Noiseless** **[inv, sincos, sincosinv]** dataset with configurations *(5,5,8)*.

```bash
./src/scripts/run_gp_cvgp.sh
```


Assume we want to run **CVGP, GP** the **Noisy** **[inv, sincos, sincosinv]** dataset with configurations *(5,5,8)*.

```bash
./src/scripts/noisy_run_gp_cvgp.sh
```


## 2. Run DSR, PQT, VPG, GPMeld

### 2.0 prequisites
1. install python environment 3.6.13: `conda create -n py3613 python=3.6.13`. 
2. use the enviorment `conda env py3613`.
3. install `dso`
```cmd
cd ./dso
pip install --upgrade setuptools pip
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS"
pip install -e ./dso
```

3. create the `.csv` data file and `.json` model configuration file

If you 
```bash
# generate the **Noiseless** **[inv, sincos, sincosinv]** dataset with configurations *(5,5,8)*.
./dso/dataset/gen_data.sh
# generate the **Noisy** **[inv, sincos, sincosinv]** dataset with configurations *(5,5,8)*.
./dso/dataset/noisy_gen_data.sh
```


4. run DSR, PQT, VPG, GPMeld models by
If you want to run DSR, PQT, VPG, GPMeld on **Noiseless** datasets.
```bash
./dso/scripts/run_dsr_pqt_vpg_gpmeld.sh
```

If you want to run DSR, PQT, VPG, GPMeld on **Noisy** datasets.
```bash
./dso/scripts/noisy_run_dsr_pqt_vpg_gpmeld.sh
```

## 3. Look at the summarized result 
Just open the `plots` folder

## 4. Parse your own results
```bash
./result/summary_output.sh
```
# Cite

If you want to reuse this material, please considering citing the following:
```bib
 @InProceedings{10.1007/978-3-031-43421-1_11,
 author="Jiang, Nan
 and Xue, Yexiang",
 title="Symbolic Regression via Control Variable Genetic Programming",
 booktitle="Machine Learning and Knowledge Discovery in Databases: Research Track",
 year="2023",
 publisher="Springer Nature Switzerland",
 address="Cham",
 pages="178--195",
 isbn="978-3-031-43421-1"
 }
```

