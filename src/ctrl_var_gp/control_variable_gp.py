import random
import time
import numpy as np
from operator import attrgetter
import copy

from program import Program


def print_prs(prs):
    for pr in prs:
        print('        ' + str(pr.__getstate__()))


def create_geometric_generations(n_generations, nvar):
    gens = [0] * nvar
    for it in range(nvar - 1, 0, -1):
        gens[it] = n_generations // 2
        n_generations -= gens[it]
    gens[0] = n_generations
    for it in range(0, nvar):
        if gens[it] < 50:
            gens[it] = 50
    print('generation #:', gens, 'sum=', sum(gens))
    return gens


def create_uniform_generations(n_generations, nvar):
    gens = [0] * nvar
    each_gen = n_generations // nvar
    for it in range(nvar - 1, 0, -1):
        gens[it] = each_gen
        n_generations -= each_gen
    gens[0] = n_generations
    print('generation #:', gens, 'sum=', sum(gens))
    return gens


class ControlVariableGeneticProgram(object):
    # the main idea.
    """
    Parameters
    ----------
    cxpb: probability of mate
    mutpb: probability of mutations
    maxdepth: the maxdepth of the tree during mutation
    population_size: the size of the selected populations (at the end of each generation)
    tour_size: the size of the tournament for selection
    hof_size: the size of the best programs retained
    nvar: number of variables

    Variables
    ---------
    population: the current list of programs
    hof: list of the best programs
    timer_log: list of times
    gen_num: number of generations, starting from 0.
    """

    # static variables
    library = None
    gp_helper = None

    def __init__(self, cxpb, mutpb, maxdepth, population_size, tour_size, hof_size,
                 n_generations, nvar):
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.maxdepth = maxdepth
        self.population_size = population_size
        self.tour_size = tour_size
        self.hof_size = hof_size

        # self.n_generations = create_geometric_generations(n_generations, nvar)
        self.n_generations = create_uniform_generations(n_generations, nvar)

        self.hof = []
        self.timer_log = []
        self.gen_num = 0

        self.nvar = nvar
        assert self.library != None
        assert Program.task != None

        start_allowed_input_tokens = np.zeros(nvar, dtype=np.int32)
        start_allowed_input_tokens[0] = 1
        self.library.set_allowed_input_tokens(start_allowed_input_tokens)

        Program.task.set_allowed_inputs(start_allowed_input_tokens)

        self.create_init_population()

    def run(self):
        # for
        most_recent_timestamp = time.perf_counter()
        for var in range(self.nvar):

            if var == self.nvar - 1:
                # XYX on Jan 5: in the last iteration there is no constants
                #               that can be changed among different experiments.
                Program.task.batchsize *= Program.opt_num_expr
                Program.opt_num_expr = 1

            # consider one variable at a time.
            for pr in self.population:
                # once we fix it, we want to remove r so that it won't change.
                pr.remove_r_evaluate()
            for pr in self.hof:
                pr.remove_r_evaluate()

            for pr in self.population:
                # a cached property in python (evaluated once)
                # forth the function to evaluate a new r
                thisr = pr.r  # how good you fit. the inverse of the residual.
            for pr in self.hof:
                thisr = pr.r

            # for fixed variable, do n generation
            for i in range(self.n_generations[var]):
                print('++++++++++++ VAR {0} ITERATION {1} ++++++++++++'.format(var, i))
                self.one_generation()

                print('hof (VAR {0} ITERATION {1})='.format(var, i))
                print_prs(self.hof)
                print("")

                now_time_stamp = time.perf_counter()
                if now_time_stamp - most_recent_timestamp >= 900:  # 15 min
                    print('print hof (VAR {0} ITERATION {1})='.format(var, i))
                    self.print_hof()
                    print("")
                    most_recent_timestamp = now_time_stamp

            for pr in self.population:
                # evaluate r again, just incase it has not been evaluated.
                this_r = pr.r
                if len(pr.const_pos) == 0 or pr.num_changing_consts == 0:
                    # only expand at those constant node. if there are no constant node,then we are done
                    # if we do not want num_changing_consts, then we also quit.
                    # print('pr.r=', pr.r)
                    pass
                else:
                    if not ("expr_objs" in pr.__dict__ and "expr_consts" in pr.__dict__):
                        print('WARNING: pr.expr_objs NOT IN DICT: pr=' + str(pr.__getstate__()))
                        pr.remove_r_evaluate()
                        this_r = pr.r
                        print('pr.expr_objs=', pr.expr_objs)
                        print('pr.expr_consts=', pr.expr_consts)
                # whether you get very different value for different constant.
                pr.freeze_equation()
                print('pr=', (pr.__getstate__()))
                print('pr.r=', pr.r)

            if var < self.nvar - 1:
                # previous we only change x0,
                # the next round, we are not allow to change x0.
                self.library.set_allowed_input_token(var, 0)  # XYX commented out this on Jan 16; trying to run noisy experiments.
                self.library.set_allowed_input_token(var + 1, 1)
                Program.task.set_allowed_input(var + 1, 1)

    def one_generation(self, iter=None):
        """
        One step of the genetic algorithm. 
        This wraps selection, mutation, crossover and hall of fame computation
        over all the individuals in the population for this epoch/step.  
        
        Parameters
        ----------            
        iter : int
            The current iteration used for logging purposes.

        """
        t1 = time.perf_counter()

        # Select the next generation individuals
        offspring = self.selectTournament(self.population_size, self.tour_size)

        print('offspring after select=')
        print_prs(offspring)
        print("")

        # Vary the pool of individuals
        offspring = self._var_and(offspring)

        print('offspring after _var_and=')
        print_prs(offspring)
        print("")

        # Replace the current population by the offspring
        self.population = offspring + self.hof

        # Update hall of fame
        self.update_hof()

        timer = time.perf_counter() - t1

        self.timer_log.append(timer)
        self.gen_num += 1

    def update_hof(self):
        # pop = [copy.deepcopy(pr) for pr in self.population]
        # new_hof = sorted(self.hof + self.population, reverse=True, key=attrgetter('r'))
        new_hof = sorted(self.population, reverse=True, key=attrgetter('r'))
        # XYX: remove duplicates?
        # self.hof = new_hof[:self.hof_size]
        self.hof = []
        for i in range(self.hof_size):
            # new_hofi = copy.deepcopy(new_hof[i])
            # if "expr_objs" in new_hof[i].__dict__:
            #     new_hofi.expr_objs = np.copy(new_hof[i].expr_objs)
            # if "expr_consts" in new_hof[i].__dict__:
            #     new_hofi.expr_consts = np.copy(new_hof[i].expr_consts)
            new_hofi = new_hof[i].clone()
            self.hof.append(new_hofi)

    def selectTournament(self, population_size, tour_size):
        offspring = []
        for pp in range(population_size):
            spr = random.sample(self.population, tour_size)
            maxspr = max(spr, key=attrgetter('r'))
            # maxspri = copy.deepcopy(maxspr)
            # if "expr_objs" in maxspr.__dict__:
            #     maxspri.expr_objs = np.copy(maxspr.expr_objs)
            # if "expr_consts" in maxspr.__dict__:
            #     maxspri.expr_consts = np.copy(maxspr.expr_consts)
            maxspri = maxspr.clone()
            offspring.append(maxspri)
        return offspring

    def _var_and(self, offspring):
        """
        Apply crossover AND mutation to each individual in a population 
        given a constant probability. 
        """

        # Apply crossover on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < self.cxpb:
                self.gp_helper.mate(offspring[i - 1],
                                    offspring[i])

        # Apply mutation on the offspring
        for i in range(len(offspring)):
            if random.random() < self.mutpb:
                self.gp_helper.multi_mutate(offspring[i], self.maxdepth)

        return offspring

    def create_init_population(self):
        """
           create the initial population; look for every token in library, fill in
           the leaves with constants or inputs.
        """
        self.population = []
        for i, t in enumerate(self.library.tokens):
            if self.library.allowed_tokens[i]:
                # otherwise (not allowed) do not need to do anything
                tree = [i]
                for j in range(t.arity):
                    t_idx = np.random.choice(self.library.tokens_of_arity[0])
                    while self.library.allowed_tokens[t_idx] == 0:
                        t_idx = np.random.choice(self.library.tokens_of_arity[0])
                    tree.append(t_idx)
                tree = np.array(tree)

                pr = Program(tree, np.ones(tree.size, dtype=np.int32))
                self.population.append(pr)

        self.hof = []
        for pr in self.population:
            # new_pr = copy.deepcopy(pr)
            # if "expr_objs" in pr.__dict__:
            #     new_pr.expr_objs = np.copy(pr.expr_objs)
            # if "expr_consts" in pr.__dict__:
            #     new_pr.expr_consts = np.copy(pr.expr_consts)
            new_pr = pr.clone()
            self.hof.append(new_pr)

    def print_population(self):
        for pr in self.population:
            print(pr.__getstate__())

    def print_hof(self):
        for pr in self.hof:
            print(pr.__getstate__())
            pr.task.rand_draw_data()
            print('validate r=', pr.task.reward_function_fixed_data(pr))
            pr.task.reward_function_fixed_data_all_metrics(pr)

            print(pr.print_expression())


class GeneticProgram(object):
    """
    Parameters
    ----------
    cxpb: probability of mate
    mutpb: probability of mutations
    maxdepth: the maxdepth of the tree during mutation
    population_size: the size of the selected populations (at the end of each generation)
    tour_size: the size of the tournament for selection
    hof_size: the size of the best programs retained

    Variables
    ---------
    population: the current list of programs
    hof: list of the best programs
    timer_log: list of times
    gen_num: number of generations, starting from 0.

    """

    # static variables
    library = None
    gp_helper = None

    def __init__(self, cxpb, mutpb, maxdepth, population_size, tour_size, hof_size,
                 n_generations):
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.maxdepth = maxdepth
        self.population_size = population_size
        self.tour_size = tour_size
        self.hof_size = hof_size
        self.n_generations = n_generations

        self.hof = []
        self.timer_log = []
        self.gen_num = 0
        self.create_init_population()

    def run(self):
        # run for n generations
        most_recent_timestamp = time.perf_counter()
        for i in range(self.n_generations):
            print('++++++++++++++++++ ITERATION {0} ++++++++++++++++++'.format(i))
            self.one_generation()

            now_time_stamp = time.perf_counter()
            if now_time_stamp - most_recent_timestamp >= 900:  # 15 min
                print('print hof (ITERATION {0})='.format(i))
                self.print_hof()
                print("")
                most_recent_timestamp = now_time_stamp

    def one_generation(self, iter=None):
        """
        One step of the genetic algorithm. 
        This wraps selection, mutation, crossover and hall of fame computation
        over all the individuals in the population for this epoch/step.  
        
        Parameters
        ----------            
        iter : int
            The current iteration used for logging purposes.

        """
        t1 = time.perf_counter()

        # Selection the next generation individuals
        offspring = self.selectTournament(self.population_size, self.tour_size)

        print('offspring after select=')
        print_prs(offspring)
        print("")

        # Vary the pool of individuals
        # the crossover and mutation.
        offspring = self._var_and(offspring)

        print('offspring after _var_and=')
        print_prs(offspring)
        print("")

        # Replace the current population by the offspring
        self.population = offspring + self.hof

        # Update hall of fame
        self.update_hof()

        timer = time.perf_counter() - t1

        self.timer_log.append(timer)
        self.gen_num += 1

    def update_hof(self):
        new_hof = sorted(self.population, reverse=True, key=attrgetter('r'))
        self.hof = []
        for i in range(self.hof_size):
            new_hofi = new_hof[i].clone()
            self.hof.append(new_hofi)

    def selectTournament(self, population_size, tour_size):
        offspring = []
        # higher fitness score has higher chance to be survive in the next generation.
        for pp in range(population_size):
            # random sample  tor_size number of individual
            spr = random.sample(self.population, tour_size)
            # select the guys has the highest fit
            maxspr = max(spr, key=attrgetter('r'))
            maxspri = maxspr.clone()
            offspring.append(maxspri)
            # offspring may have duplicates,
        return offspring

    def _var_and(self, offspring):
        """
        Apply crossover AND mutation to each individual in a population 
        given a constant probability. 
        """
        # offspring = [copy.deepcopy(pr) for pr in self.population]

        # Apply crossover on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < self.cxpb:
                self.gp_helper.mate(offspring[i - 1],
                                    offspring[i])

        # Apply mutation on the offspring
        for i in range(len(offspring)):
            if random.random() < self.mutpb:
                # for everyone you randomly mutate them.
                self.gp_helper.multi_mutate(offspring[i], self.maxdepth)

        return offspring

    def create_init_population(self):
        """
           create the initial population; look for every token in library, fill in
           the leaves with constants or inputs.
        """
        self.population = []
        for i, t in enumerate(self.library.tokens):
            if self.library.allowed_tokens[i]:
                # otherwise (not allowed) do not need to do anything
                tree = [i]
                for j in range(t.arity):
                    t_idx = np.random.choice(self.library.tokens_of_arity[0])
                    while self.library.allowed_tokens[t_idx] == 0:
                        t_idx = np.random.choice(self.library.tokens_of_arity[0])
                    tree.append(t_idx)
                tree = np.array(tree)

                pr = Program(tree, np.ones(tree.size, dtype=np.int32))
                self.population.append(pr)

        self.hof = []
        for pr in self.population:
            new_pr = pr.clone()
            self.hof.append(new_pr)

    def print_population(self):
        for pr in self.population:
            print(pr.__getstate__())

    def print_hof(self):
        for pr in self.hof:
            print(pr.__getstate__())
            pr.task.rand_draw_data()

            print('validate r=', pr.task.reward_function_fixed_data(pr))
            pr.task.reward_function_fixed_data_all_metrics(pr)

            print(pr.print_expression())


class GPHelper(object):
    """
    Function class for genetic programming.

    Parameters
    ----------

    Methods
    -------
    mate(a, b): apply cross over of two program trees; find two subtrees 
       within allowed_change_tokens then swap them
    
    """

    # static variables
    library = None

    def mate(self, a, b):
        """
            a and b are two program objects.
        """
        a_allowed = a.allow_change_pos()
        b_allowed = b.allow_change_pos()

        if len(a_allowed) == 0 or len(b_allowed) == 0:
            return

        a_start = random.sample(a_allowed, 1)[0]
        b_start = random.sample(b_allowed, 1)[0]

        a_end = a.subtree_end(a_start)
        b_end = b.subtree_end(b_start)


        na_tokens = np.concatenate((a.tokens[:a_start],
                                    b.tokens[b_start:b_end],
                                    a.tokens[a_end:]))
        nb_tokens = np.concatenate((b.tokens[:b_start],
                                    a.tokens[a_start:a_end],
                                    b.tokens[b_end:]))

        na_allow = np.concatenate((a.allow_change_tokens[:a_start],
                                   b.allow_change_tokens[b_start:b_end],
                                   a.allow_change_tokens[a_end:]))
        nb_allow = np.concatenate((b.allow_change_tokens[:b_start],
                                   a.allow_change_tokens[a_start:a_end],
                                   b.allow_change_tokens[b_end:]))

        a.__init__(na_tokens, na_allow)
        b.__init__(nb_tokens, nb_allow)
        a.remove_r_evaluate()
        b.remove_r_evaluate()

    def gen_full(self, maxdepth):
        """
            generate a full program tree recursively (represented in token indicies in library)
        """
        if maxdepth == 1:

            # more efficient implementation
            allowed_pos = [t for t in self.library.tokens_of_arity[0] \
                           if self.library.allowed_tokens[t] > 0]
            t_idx = random.choice(allowed_pos)
            return [t_idx]
        else:
            # more efficient implementation
            allowed_pos = self.library.allowed_tokens_pos()
            t_idx = random.choice(allowed_pos)

            arity = self.library.tokens[t_idx].arity
            tree = [t_idx]
            for i in range(arity):
                tree.extend(self.gen_full(maxdepth - 1))
            return tree

    def multi_mutate(self, individual, maxdepth):
        """Randomly select one of four types of mutation."""

        v = np.random.randint(0, 4)

        if v == 0:
            self.mutUniform(individual, maxdepth)
        elif v == 1:
            self.mutNodeReplacement(individual)
        elif v == 2:
            self.mutInsert(individual, maxdepth)
        elif v == 3:
            self.mutShrink(individual)

    def mutUniform(self, p, maxdepth):
        """
            find a leaf node (which allow_change_tokens == 1), replace the node with a gen_full
            tree of maxdepth.
        """
        leaf_set = []
        for i, token in enumerate(p.traversal):
            if p.allow_change_tokens[i] > 0 and token.arity == 0:
                leaf_set.append(i)
        if len(leaf_set) == 0:
            return
        t_idx = np.random.choice(np.array(leaf_set))

        new_tree = np.array(self.gen_full(maxdepth))

        np_tokens = np.concatenate((p.tokens[:t_idx], new_tree, p.tokens[(t_idx + 1):]))
        np_allow = np.insert(p.allow_change_tokens, t_idx, \
                             np.ones(len(new_tree) - 1, dtype=np.int32))

        p.__init__(np_tokens, np_allow)
        p.remove_r_evaluate()

    def mutNodeReplacement(self, p):
        """
            find a node and replace it with a node of the same arity
        """
        allowed_pos = p.allow_change_pos()
        if len(allowed_pos) == 0:
            return
        a_idx = allowed_pos[random.randint(0, len(allowed_pos) - 1)]
        arity = p.traversal[a_idx].arity


        allowed_pos = [t for t in self.library.tokens_of_arity[arity] \
                       if self.library.allowed_tokens[t] > 0]
        t_idx = random.choice(allowed_pos)

        p.tokens[a_idx] = t_idx
        p.__init__(p.tokens, p.allow_change_tokens)
        p.remove_r_evaluate()

    def mutInsert(self, p, maxdepth):
        """
            insert a node at a random position, the original subtree at the location 
            becomes one of its subtrees.
        """
        insert_pos = random.randint(0, len(p.tokens) - 1)
        subtree_start = insert_pos
        subtree_end = p.subtree_end(subtree_start)

        non_term_allowed = self.library.allowed_non_terminal_tokens_pos()
        t_idx = random.choice(non_term_allowed)

        root_arity = self.library.tokens[t_idx].arity

        which_old_tree = random.randint(0, root_arity - 1)

        # generate other subtrees
        np_tokens = np.concatenate((p.tokens[:subtree_start], np.array([t_idx])))
        np_allow = np.concatenate((p.allow_change_tokens[:subtree_start], np.array([1])))

        for i in range(root_arity):
            if i == which_old_tree:
                np_tokens = np.concatenate((np_tokens,
                                            p.tokens[subtree_start:subtree_end]))
                np_allow = np.concatenate((np_allow,
                                           p.allow_change_tokens[subtree_start:subtree_end]))
            else:
                subtree = np.array(self.gen_full(maxdepth - 1))
                np_tokens = np.concatenate((np_tokens, subtree))
                np_allow = np.insert(np_allow, -1, np.ones(len(subtree), dtype=np.int32))

        # add the rest
        np_tokens = np.concatenate((np_tokens, p.tokens[subtree_end:]))
        np_allow = np.concatenate((np_allow, p.allow_change_tokens[subtree_end:]))

        p.__init__(np_tokens, np_allow)
        p.remove_r_evaluate()

    def mutShrink(self, p):
        """
            delete a node (which allow_change_tokens == 1), use one of its child to replace
            its position.
        """
        allowed_pos = p.allow_change_pos()
        if len(allowed_pos) == 0:
            return
        a_idx = allowed_pos[random.randint(0, len(allowed_pos) - 1)]
        arity = p.traversal[a_idx].arity

        # print('arity=', arity)
        if arity == 0:
            # replace it with another node arity == 0 (perhaps not this node; but it is OK).
            self.mutNodeReplacement(p)
        else:
            a_end = p.subtree_end(a_idx)

            # get all the subtrees
            subtrees_start = []
            subtrees_end = []
            k = a_idx + 1
            while k < a_end:
                k_end = p.subtree_end(k)
                subtrees_start.append(k)
                subtrees_end.append(k_end)
                k = k_end

            # pick one of the subtrees, and re-assemble
            sp = random.randint(0, len(subtrees_start) - 1)

            np_tokens = np.concatenate((p.tokens[:a_idx],
                                        p.tokens[subtrees_start[sp]:subtrees_end[sp]],
                                        p.tokens[a_end:]))
            np_allow = np.concatenate((p.allow_change_tokens[:a_idx],
                                       p.allow_change_tokens[subtrees_start[sp]:subtrees_end[sp]],
                                       p.allow_change_tokens[a_end:]))
            p.__init__(np_tokens, np_allow)
            p.remove_r_evaluate()
