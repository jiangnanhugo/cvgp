import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import sys
from dso.program import Program
import json
# used to load true program for data generation
from dso.library import Library, Token, PlaceholderConstant
import dso.functions as functions
from dso.config import load_config
import pickle
import numpy as np

import click


class DataGen(object):
    def __init__(self, true_program_file, n_samples=40000, noise_std=0.1):
        self.true_program, self.n_input = load_true_program(true_program_file)
        for expr in self.true_program.pretty():
            print(expr)
        self.n_samples = n_samples
        self.noise_std=noise_std

    def data_gen(self):
        self.X = np.random.rand(self.n_samples, self.n_input)*9.5+0.5
        y_true = self.true_program.execute(self.X) + np.random.normal(0.0, scale=self.noise_std, size=self.n_samples)
        self.y = y_true.reshape(self.n_samples, 1)

    def to_csv(self, filename):
        d = np.concatenate((self.X, self.y), axis=1)
        np.savetxt(filename, d, delimiter=",")

def load_true_program(true_program_file, allow_change_const=0):
    def get_library(nvar=5):
        # get all the functions and variables ready
        var_x = []
        for i in range(nvar):
            xi = Token(None, 'X_' + str(i), 0, 0., i)
            var_x.append(xi)

        ops = [
                # Binary operators
                functions.unprotected_ops[0],
                functions.unprotected_ops[1],
                functions.unprotected_ops[2],
                functions.unprotected_ops[3],
                functions.unprotected_ops[4],
                functions.unprotected_ops[5],
                functions.protected_ops[0],  # 'div'
                functions.protected_ops[5]  # 'inv' '1/x'
                ]
        named_const = [PlaceholderConstant(1.0)]
        protected_library = Library(ops + var_x + named_const)
        return protected_library

    prog = pickle.load(open(true_program_file, 'rb'))

    vars = set([x for x in prog["preorder"] if "X_" in x])
    protected_library = get_library(len(vars))
    # relevant hyper parameters

    # get program ready
    Program.library = protected_library

    Program.set_execute(True)  # protected = True

    # set const_optimizer
    # Program.const_optimizer = ScipyMinimize()

    preorder_actions = protected_library.actionize(prog['preorder'])
    # true_pr_allow_change = allow_change_const * np.ones(len(prog['preorder']), dtype=np.int32)
    true_pr = Program(tokens=preorder_actions)
    for loc, c in zip(prog['const_loc'], prog['consts']):
        true_pr.traversal[loc] = PlaceholderConstant(c)
    print("true expression is:")
    # print(true_pr.print_expression())

    return true_pr, len(vars)


@click.command()
@click.argument('config_template', default="")
@click.argument('program_file', default="", type=str)
@click.argument('output_base', default="dataset", type=str)
@click.argument('seed', default=10086, type=int)
@click.argument('baseline_name', default='default', type=str)
@click.argument('n_samples', default=100000, type=int)
@click.argument('noise_std', default=0.1, type=float)
def main(config_template, program_file, output_base, seed, baseline_name, n_samples, noise_std):
    dgen = DataGen(program_file, n_samples=n_samples, noise_std=noise_std)
    dgen.data_gen()
    dgen.to_csv(output_base + '.csv')

    # Load the experiment config
    config_template = config_template if config_template != "" else None
    config = load_config(config_template)
    # Overwrite config seed, if specified
    config["experiment"]["seed"] = seed
    config["training"]["n_samples"] = n_samples
    config["training"]["n_cores_batch"] = 4

    # Save starting seed and run command
    config["experiment"]["starting_seed"] = config["experiment"]["seed"]
    # set metric and true program file
    config['task']['dataset'] = output_base + '.csv'
    metrics = ["inv_nrmse", "inv_mse", "inv_nmse", "inv_rmse", "inv_nrmse"]
    for metric in metrics:
        config['task']['metric'] = metric
        with open(output_base + '_' + baseline_name + "_" + metric + '.json', 'w') as fw:
            json.dump(config, fw, indent=4)


if __name__ == "__main__":
    main()
