import argparse
import os

import pandas as pd

from compute_dso_all_metrics import compute_dso_all_metrics, compute_eureqa_all_metrics


def read_until_line_starts_with(inp, line):
    l = inp.readline()
    while l != "" and not l.startswith(line):
        l = inp.readline()
    return l


def create_all_metrics_dict(inp):
    l = inp.readline()
    # print(l)
    l = inp.readline()
    # print(l)
    val_dict = {}
    while l != "" and not l.startswith("%%%%%"):
        spl = l.split(" ")
        val_dict[spl[0]] = float(spl[1].strip())
        l = inp.readline()
        # print(l)
    # print(val_dict)
    return val_dict


def parse_gp_file(filename):
    # print('filename=', filename)
    inp = open(filename, 'r')
    l = read_until_line_starts_with(inp, 'final hof')
    # print('l=', l)
    rs = []
    l = read_until_line_starts_with(inp, 'validate r=')
    while l != "":
        # print('l=', l.strip())
        tt = l[:-1].split()
        val_dict = create_all_metrics_dict(inp)
        rs.append([float(tt[2]), val_dict])
        l = read_until_line_starts_with(inp, 'validate r=')

    inp.close()
    # print(rs)
    rs.sort(key=lambda x: x[0], reverse=True)  # changes the list in-place (and returns None)
    # print(rs[0])
    r = rs[0]
    return r


def parse_dso_file(dso_log_filename, true_program_file, basepath, noise_std=0.1):
    if not os.path.isfile(dso_log_filename):
        print(dso_log_filename, 'does not exist')
    # print(dso_log_filename)
    inp = open(dso_log_filename, 'r')
    l = read_until_line_starts_with(inp, 'Source path______')
    # print(l)
    data_frame_basepath = l.strip().split('/')[-1]
    print(data_frame_basepath)
    csv_expr_dir = os.path.join(basepath, 'scripts/log', data_frame_basepath)

    if not os.path.isdir(csv_expr_dir):
        print(csv_expr_dir, 'does not exists')
        return None
    else:
        print(csv_expr_dir, 'exists')
    csv_expr_path = None
    for root, dirs, files in os.walk(csv_expr_dir):
        for name in files:
            if name.endswith("hof.csv"):
                csv_expr_path = os.path.join(root, name)
                break
    if not os.path.isfile(csv_expr_path):
        print("cannot find hof file")
    print(csv_expr_path)
    print(true_program_file)
    return compute_dso_all_metrics(true_program_file, csv_expr_path, testset_size=256, noise_std=noise_std)


def parse_exp_set(file_prefix, metric_name, file_suffix, start, end, true_program_file, dso_basepath, noise_std):
    all_dso_r, all_gp_r, all_egp_r = {}, {}, {}
    for baseline_name in ['VPG', 'PQT', 'DSR', 'GPMELD']:
        all_dso_r[baseline_name] = {}
    for i in range(start, end):
        # all_dso_r[i], all_gp_r[i], all_egp_r[i] = {}, {}, {}

        for baseline_name in ['VPG', 'PQT', 'DSR', 'GPMELD']:
            try:
                dso_file = file_prefix + str(i) + '.data.metric_inv_nrmse.' + baseline_name + file_suffix
                print(dso_file)
                dso_r = parse_dso_file(dso_file, true_program_file + str(i) + '.data', dso_basepath, noise_std)
                if dso_r != None:
                    all_dso_r[baseline_name][i] = dso_r
                    # print('dso', dso_r)
            except:
                print(i, "cannot process with", baseline_name)
        try:
            gp_file = file_prefix + str(i) + '.data.metric_' + metric_name + '.gp' + file_suffix
            print(gp_file)
            if not os.path.isfile(gp_file):
                raise FileExistsError(gp_file, 'does not exists!')
            gp_r = parse_gp_file(gp_file)
            all_gp_r[i] = gp_r[-1]
            print('gp', gp_r[-1])
        except:
            print(i, "gp cannot process")
        try:
            egp_file = file_prefix + str(i) + '.data.metric_' + metric_name + '.egp' + file_suffix
            if not os.path.isfile(egp_file):
                raise FileExistsError(egp_file, 'does not exists!')
            egp_r = parse_gp_file(egp_file)
            all_egp_r[i] = egp_r[-1]
            print('egp', egp_r[-1])
        except:
            print(i, "egp cannot process")

    return all_dso_r, all_gp_r, all_egp_r


def parse_eureqa_solutions(eureqa_basepath, true_program_file, noise_std, start=0, end=10):
    df = pd.read_csv(eureqa_basepath)
    all_eureqa_r = {}
    result_dict = {}
    for _, row in df.iterrows():
        prog = row['benchmark']
        idx = int(prog.split('_')[-1])
        predicted = row['solution']
        result_dict[idx] = predicted
    for i in range(start, end):
        try:
            eureqa_ri = compute_eureqa_all_metrics(true_program_file + str(i) + '.data', result_dict[i], testset_size=256,
                                                    noise_std=noise_std)
            print(eureqa_ri)
            all_eureqa_r[i] = eureqa_ri
        except:
            print(i, "eureqa cannot process")

    return all_eureqa_r


def pretty_print_dso_family(all_rs):
    for key in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse', 'neg_mse', 'neg_rmse', 'neglog_mse', 'inv_mse']:
        # print('{}\ndata idx, gp, expand_gp, dso'.format(key))
        print('{}, VPG, PQT, DSR, GPMELD'.format(key))
        for idx in range(start, end):
            print(idx, end=", ")
            for baseline_name in ['VPG', 'PQT', 'DSR', 'GPMELD']:
                if idx in all_rs[baseline_name]:
                    print(all_rs[baseline_name][idx][key], end=", ")
                else:
                    print(",", end=" ")
            print()
        print()


def pretty_print_pair(all_gp_rs, all_egp_rs):
    for key in ['neg_nmse',  'neg_nrmse', 'inv_nrmse', 'inv_nmse', 'neg_mse', 'neg_rmse', 'neglog_mse', 'inv_mse']:

        print(key, ", GP, CVGP")
        for idx in range(start, end):
            print(idx, end=", ")
            if idx in all_gp_rs:
                print(all_gp_rs[idx][key], end=", ")
            else:
                print(",", end=" ")
            if idx in all_egp_rs:
                print(all_egp_rs[idx][key])
            else:
                print()
        print()





if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--fp', type=str, required=True)
    parser.add_argument('--metric', type=str, default='inv_nrmse', required=True)
    parser.add_argument('--true_program_file', type=str, required=True)
    parser.add_argument('--dso_basepath', type=str, required=True, default='None')
    parser.add_argument('--eureqa_path', type=str, required=False, default="None")
    parser.add_argument('--noise_std', type=float, default=0.1)
    #
    # Parse the argument
    args = parser.parse_args()
    file_suffix = '.out'
    start = 0
    end = 1
    all_dso_r, all_gp_r, all_egp_r = parse_exp_set(args.fp, args.metric, file_suffix, start, end, args.true_program_file, args.dso_basepath,
                                                   args.noise_std)

    if len(all_dso_r) != 0:
        pretty_print_dso_family(all_dso_r)
    print("GP & CVGP")
    if len(all_gp_r) != 0 or len(all_egp_r):
        pretty_print_pair(all_gp_r, all_egp_r)
