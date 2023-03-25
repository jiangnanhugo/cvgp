import argparse
import random
import pickle
import os
import numpy as np

from program import Program
from library import Library, Token, PlaceholderConstant


def is_copy(term1, term2):
    if len(term1) != len(term2):
        return False
    for t1, t2 in zip(term1, term2):
        if t1 != t2:
            return False
    return True


def is_duplicate(term, term_list):
    for old_term in term_list:
        # print('old_term=', old_term)
        # print('term=', term)
        if is_copy(term, old_term):
            return True
    return False


def sample_one_decr(decr_arity):
    t = random.random() 
    if t < 0.5:
        return 0  # no decr
    else:
        t = (t - 0.5) * 20. / (10./decr_arity)
        return int(t)+1
        #return 0  # no decr


def gen_terms(n_vars, vc, n_term, decr_arity, int_coef=False):
    # print('n_vars=', n_vars, 'vc=', vc, 'n_term=', n_term)
    term_list = []
    decr_list = []
    const_list = []
    for i in range(n_term):
        term = random.sample(range(n_vars), vc)
        term.sort()
        # print('term=', term)
        while is_duplicate(term, term_list):
            term = random.sample(range(n_vars), vc)
            term.sort()
            # print('term=', term)
        term_list.append(term)
        decr_list.append([sample_one_decr(decr_arity) for j in range(vc)])
        # change from random real number to random integer so that it would be compatiable with the DSO output
        if int_coef:
            const_list.append(random.randint(-10, 10))
        else:
            const_list.append(round(random.uniform(-1., 1.), 4))

    return term_list, decr_list, const_list


def gen_program_one_term(term, decr, decr_list):
    preorder = ['const']
    for var, de in zip(term, decr):
        if de == 0:
            preorder = ['mul'] + preorder + ['X_' + str(var)]
        else:
            preorder = ['mul'] + preorder + [decr_list[de-1], 'X_' + str(var)]
    return preorder


def gen_one_program(param):
    """
    generate one program (its preorder operator list) and the constants,
    return the preorder list, and the constant loc list, constant list
    """
    preorder = ['const']
    # change from random real number to random integer so that it would be compatiable with the DSO output
    consts = []
    if param.int_coef:
        consts.append(random.randint(-10, 10))
    else:
        consts.append(round(random.uniform(-1., 1.), 4))

    for vc1, n_term in enumerate(param.n_terms):
        vc = vc1 + 1
        term_list, decr_list, const_list = gen_terms(param.n_vars, vc, param.n_terms[vc1], len(param.decor), int_coef=param.int_coef)
        for j in range(param.n_terms[vc1]):
            preorder_term = gen_program_one_term(term_list[j], decr_list[j], param.decor)
            preorder = ['add'] + preorder + preorder_term
            consts.append(const_list[j])

    const_loc = [i for i, term in enumerate(preorder) if term.startswith('const')]
    return preorder, const_loc, consts


def read_true_program(filename):
    inp = open(filename, 'rb')
    prog = pickle.load(inp)
    inp.close()
    return prog


def build_program(prog, library, allow_change_const):
    print('preorder=', prog['preorder'])
    print('const_loc=', prog['const_loc'])
    print('consts=', prog['consts'])

    preorder_actions = library.actionize(prog['preorder'])
    true_pr_allow_change = allow_change_const * np.ones(len(prog['preorder']), dtype=np.int32)
    true_pr = Program(preorder_actions, true_pr_allow_change)
    for loc, c in zip(prog['const_loc'], prog['consts']):
        true_pr.traversal[loc] = PlaceholderConstant(c)
    print("true expression is:")
    print(true_pr.print_expression())
    print("return true program")
    return true_pr


class Param:
    pass


def main(param):
    if not os.path.isdir(param.folder):
        print("creating", param.folder)
        os.mkdir(param.folder)

    n_programs = 50

    for i in range(n_programs):
        preorder, const_loc, consts = gen_one_program(param)
        print('preorder=', preorder)
        print('const_loc=', const_loc)
        print('consts=', consts)

        prog = {'preorder': preorder,
                'const_loc': const_loc,
                'consts': consts}

        oup = open(param.folder + '/prog_' + str(i) + '.data', 'wb')
        print(param.folder + '/prog_' + str(i) + '.data')
        pickle.dump(prog, oup)
        oup.close()


if __name__ == '__main__':
    param = Param()
    param.int_coef=False

    basepath = "~/Desktop/CVGP_code_implementation/data"
    
    param.n_vars = 5
    param.n_terms = [5, 8]
    param.decor = [ 'inv' ]
    param.folder = basepath + '/inv_nv{}_nt{}{}'.format(param.n_vars, param.n_terms[0], param.n_terms[1])

    main(param)
