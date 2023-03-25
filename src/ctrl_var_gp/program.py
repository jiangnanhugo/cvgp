"""Class for symbolic expression object or program."""

import array
import warnings
from textwrap import indent

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty

from functions import PlaceholderConstant
from const import make_const_optimizer
from utils import cached_property
import utils as U

from scipy.optimize import minimize

def _finish_tokens(tokens):
    """
    Complete a possibly unfinished string of tokens.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal.

    Returns
    _______
    tokens : list of ints
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    XYX: is this what the function is doing?? I doubt...

    """

    n_objects = Program.n_objects

    arities = np.array([Program.library.arities[t] for t in tokens])
    # Number of dangling nodes, returns the cumsum up to each point
    # Note that terminal nodes are -1 while functions will be >= 0 since arities - 1
    dangling = 1 + np.cumsum(arities - 1)

    if -n_objects in (dangling - 1):
        # Chop off tokens once the cumsum reaches 0, This is the last valid point in the tokens
        expr_length = 1 + np.argmax((dangling - 1) == -n_objects)
        tokens = tokens[:expr_length]
    else:
        # Extend with valid variables until string is valid
        # NOTE: This only appends onto the end of a set of tokens, even in the multi-object case!
        assert n_objects == 1, "Is max length constraint turned on? Max length constraint required when n_objects > 1."
        tokens = np.append(tokens, np.random.choice(Program.library.input_tokens, size=dangling[-1]))

    return tokens


def from_str_tokens(str_tokens, skip_cache=False):
    """
    Memoized function to generate a Program from a list of str and/or float.
    See from_tokens() for details.

    Parameters
    ----------
    str_tokens : str | list of (str | float)
        Either a comma-separated string of tokens and/or floats, or a list of
        str and/or floats.

    skip_cache : bool
        See from_tokens().

    Returns
    -------
    program : Program
        See from_tokens().
    """

    # Convert str to list of str
    if isinstance(str_tokens, str):
        str_tokens = str_tokens.split(",")

    # Convert list of str|float to list of tokens
    if isinstance(str_tokens, list):
        traversal = []
        constants = []
        for s in str_tokens:
            if s in Program.library.names:
                t = Program.library.names.index(s.lower())
            elif U.is_float(s):
                assert "const" not in str_tokens, "Currently does not support both placeholder and hard-coded constants."
                t = Program.library.const_token
                constants.append(float(s))
            else:
                raise ValueError("Did not recognize token {}.".format(s))
            traversal.append(t)
        traversal = np.array(traversal, dtype=np.int32)
    else:
        raise ValueError("Input must be list or string.")

    # Generate base Program (with "const" for constants)
    p = from_tokens(traversal, skip_cache=skip_cache)

    # Replace any constants
    p.set_constants(constants)

    return p


def from_tokens(tokens, skip_cache=False, on_policy=True, finish_tokens=True):
    """
    Memoized function to generate a Program from a list of tokens.

    Since some tokens are nonfunctional, this first computes the corresponding
    traversal. If that traversal exists in the cache, the corresponding Program
    is returned. Otherwise, a new Program is returned.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    skip_cache : bool
        Whether to bypass the cache when creating the program (used for
        previously learned symbolic actions in DSP).
        
    finish_tokens: bool
        Do we need to finish this token. There are instances where we have
        already done this. Most likely you will want this to be True. 

    Returns
    _______
    program : Program
        The Program corresponding to the tokens, either pulled from memoization
        or generated from scratch.
    """

    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''

    if finish_tokens:
        tokens = _finish_tokens(tokens)

    # For stochastic Tasks, there is no cache; always generate a new Program.
    # For deterministic Programs, if the Program is in the cache, return it;
    # otherwise, create a new one and add it to the cache.
    if skip_cache or Program.task.stochastic:
        p = Program(tokens, on_policy=on_policy)
    else:
        key = tokens.tostring()
        try:
            p = Program.cache[key]
            if on_policy:
                p.on_policy_count += 1
            else:
                p.off_policy_count += 1
        except KeyError:
            p = Program(tokens, on_policy=on_policy)
            Program.cache[key] = p

    return p


class Program(object):
    """
    The executable program representing the symbolic expression.

    The program comprises unary/binary operators, constant placeholders
    (to-be-optimized), input variables, and hard-coded constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. "Dangling"
        programs are completed with repeated "x1" until the expression
        completes.

    Attributes
    ----------
    traversal : list
        List of operators (type: Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarry (dtype: int)
        Array of integers whose values correspond to indices

    allow_change_tokens: np.ndarry (dtype: int)
        if each token allows to be changed during GP. 

    const_pos : list of int
        A list of indicies of constant placeholders along the traversal.

    num_changing_const: int
        number of changing constant.

    float_pos : list of float
        A list of indices of constants placeholders or floating-point constants
        along the traversal.

    sympy_expr : str
        The (lazily calculated) SymPy expression corresponding to the program.
        Used for pretty printing _only_.

    complexity : float
        The (lazily calcualted) complexity of the program.

    r : float
        The (lazily calculated) reward of the program.

    expr_objs: array of floats
        The objective functions done with opt_num_expr experiments during optimization.

    expr_consts: 2-d array of floats
        The optimized constant values with opt_num_expr experiments during optimization. 

    count : int
        The number of times this Program has been sampled.

    str : str
        String representation of tokens. Useful as unique identifier.
    """
    # data_used = {'X':[], 'y_true': []}
    # Static variables
    task = None  # Task
    library = None  # Library
    const_optimizer = None  # Function to optimize constants
    cache = {}
    n_objects = 1  # Number of executable objects per Program instance

    opt_num_expr = 32  # number of experiments done for optimization

    expr_obj_thres = 1e-2
    expr_consts_thres = 1e-3

    # Cython-related static variables
    have_cython = None  # Do we have cython installed
    execute = None  # Link to execute. Either cython or python
    cyfunc = None  # Link to cyfunc lib since we do an include inline

    noise_std = 0.0

    def __init__(self, tokens=None, allow_change_tokens=None):
        """
        Builds the Program from a list of of integers corresponding to Tokens.
        """

        # Can be empty if we are unpickling 
        if tokens is not None:
            self._init(tokens, allow_change_tokens)

    def _init(self, tokens, allow_change_tokens):
        # pre-order of the program. the most important thing.
        self.traversal = [Program.library[t] for t in tokens]
        # added part: which token is allowed to be token. 1 means allowed
        self.allow_change_tokens = allow_change_tokens
        # position of the constant
        self.const_pos = [i for i, t in enumerate(self.traversal) if isinstance(t, PlaceholderConstant)]
        self.num_changing_consts = 0
        for pos in self.const_pos:  # compute num_changing_consts
            if self.allow_change_tokens[pos]:
                self.num_changing_consts += 1
        self.len_traversal = len(self.traversal)

        if self.have_cython and self.len_traversal > 1:
            self.is_input_var = array.array('i', [t.input_var is not None for t in self.traversal])

        self.invalid = False  # always false.
        self.str = tokens.tostring()
        self.tokens = tokens

        if Program.n_objects > 1:  # only 1 function, output is one :y=f(x1, x2,...).
            # XYX: this part is useless; for our application, n_objects == 1.
            # Fill list of multi-traversals
            danglings = -1 * np.arange(1, Program.n_objects + 1)
            self.traversals = []  # list to keep track of each multi-traversal
            i_prev = 0
            arity_list = []  # list of arities for each node in the overall traversal
            for i, token in enumerate(self.traversal):
                arities = token.arity
                arity_list.append(arities)
                dangling = 1 + np.cumsum(np.array(arity_list) - 1)[-1]
                if (dangling - 1) in danglings:
                    trav_object = self.traversal[i_prev:i + 1]
                    self.traversals.append(trav_object)
                    i_prev = i + 1
                    """
                    Keep only what dangling values have not yet been calculated. Don't want dangling to go down and up (e.g hits -1, goes back up to 0 before hitting -2)
                    and trigger the end of a traversal at the wrong time
                    """
                    danglings = danglings[danglings != dangling - 1]

    def clone(self):
        new_me = Program(self.tokens, self.allow_change_tokens)
        
        for i in range(len(self.traversal)):
            if isinstance(self.traversal[i], PlaceholderConstant):
                new_me.traversal[i] = PlaceholderConstant(self.traversal[i].value)

        new_me.allow_change_tokens = np.copy(self.allow_change_tokens)
        new_me.tokens = np.copy(self.tokens)

        if 'r' in self.__dict__:
            new_me.r = self.r
        if 'expr_objs' in self.__dict__:
            new_me.expr_objs = np.copy(self.expr_objs)
        if 'expr_consts' in self.__dict__:
            new_me.expr_consts = np.copy(self.expr_consts)

        return new_me
        

    def __getstate__(self):
        # for printing purpose
        have_r = "r" in self.__dict__
        have_evaluate = "evaluate" in self.__dict__
        possible_const = have_r or have_evaluate
        # muliplie_rewards=[]
        # for i in range(10):
        #     self.task.rand_draw_data()
        #     muliplie_rewards.append(self.task.reward_function_fixed_data(self))

        state_dict = {'tokens': self.tokens,  # string rep comes out different if we cast to array, so we can get cache misses.
                      'allow_change_tokens': self.allow_change_tokens,
                      'have_r': bool(have_r),
                      'r': float(self.r) if have_r else float(-np.inf),
                      # 'multiply_r':muliplie_rewards,
                      'fixed_column': self.task.fixed_column,
                      'have_evaluate': bool(have_evaluate),
                      'evaluate': self.evaluate if have_evaluate else float(-np.inf),
                      'const': array.array('d', self.get_constants()) if possible_const else float(-np.inf),
                      'invalid': bool(self.invalid),
                      'error_node': array.array('u', "" if not self.invalid else self.error_node),
                      'error_type': array.array('u', "" if not self.invalid else self.error_type)}

        # In the future we might also return sympy_expr and complexity if we ever need to compute in parallel 

        return state_dict

    def __setstate__(self, state_dict):

        # Question, do we need to init everything when we have already run, or just some things?
        self._init(state_dict['tokens'], state_dict['allow_change_tokens'])

        have_run = False

        if state_dict['have_r']:
            setattr(self, 'r', state_dict['r'])
            have_run = True

        if state_dict['have_evaluate']:
            setattr(self, 'evaluate', state_dict['evaluate'])
            have_run = True

        if have_run:
            self.set_constants(state_dict['const'].tolist())
            self.invalid = state_dict['invalid']
            self.error_node = state_dict['error_node'].tounicode()
            self.error_type = state_dict['error_type'].tounicode()

    def allow_change_pos(self):
        # the place the token can be changed
        return [i for i, t in enumerate(self.allow_change_tokens) if t == 1]

    def subtree_end(self, subtree_start):
        # subtree_start arbitraty
        # the END point of that subtree in preorder
        k = subtree_start
        s = self.traversal[k].arity
        k += 1
        while k < self.len_traversal and s > 0:
            s -= 1 - self.traversal[k].arity
            k += 1
        return k

    def remove_r_evaluate(self):
        # remove  r
        if 'r' in self.__dict__:
            del self.__dict__['r']
        if 'evaluate' in self.__dict__:
            del self.__dict__['evaluate']
        if 'expr_objs' in self.__dict__:
            del self.__dict__['expr_objs']
        if 'expr_consts' in self.__dict__:
            del self.__dict__['expr_consts']

    def execute(self, X):
        # execute the program with input X and obtain output.
        """
        Execute program on input X.

        Parameters
        ==========

        X : np.array
            Input to execute the Program over.

        Returns
        =======

        result : np.array or list of np.array
            In a single-object Program, returns just an array. In a multi-object Program, returns a list of arrays.
        """
        if Program.n_objects > 1:
            # XYX: this part is useless. For our application, n_objects == 1.
            if not Program.protected:
                result = []
                invalids = []
                for trav in self.traversals:
                    val, invalid, self.error_node, self.error_type = Program.execute_function(trav, X)
                    result.append(val)
                    invalids.append(invalid)
                self.invalid = any(invalids)
            else:
                result = [Program.execute_function(trav, X) for trav in self.traversals]
            return result
        else:
            if not Program.protected:
                # return some weired error.
                result, self.invalid, self.error_node, self.error_type = Program.execute_function(self.traversal, X)
            else:
                result = Program.execute_function(self.traversal, X)
                # always protected. 1/div
            return result

    def optimize(self):
        """
        Optimizes PlaceholderConstant tokens against the reward function. The
        optimized values are stored in the traversal.
        """
        # find the best constant value and fit the equation.
        if len(self.const_pos) == 0 or self.num_changing_consts == 0:
            return

        # Define the objective function: negative reward
        def f(consts):
            # replace all the constant in self.travasal with the given constant.
            self.set_constants(consts)

            # r = self.task.reward_function(self)
            # evaluate the diffferent betwen predited y and the groundtruth y
            r = self.task.reward_function_fixed_data(self)
            # self.data_used["X"].append(self.task.X)
            # self.data_used["y_true"].append(self.task.y_true)
            # minimize the objective function
            obj = -r  # Constant optimizer minimizes the objective function

            # Need to reset to False so that a single invalid call during
            # constant optimization doesn't render the whole Program invalid.
            self.invalid = False

            return obj

        optimized_constants = []
        optimized_obj = []
        # how many time
        # do more than one experiment, so that we can set x2-x4 with different constant value.
        for expr in range(self.opt_num_expr):
            # Do the optimization
            # x0 = np.ones(self.num_changing_consts) # Initial guess
            x0 = np.random.rand(self.num_changing_consts) * 10
            # print('c0=', x0)

            # self.task.rand_draw_X_fixed()
            self.task.rand_draw_data()
            # the returned constant, and the objective function.
            # t_optimized_constants, t_optimized_obj = Program.const_optimizer(f, x0)
            if Program.noise_std > 0:
                opt_result = minimize(f, x0, method='BFGS', options={'eps':Program.noise_std})
            else:
                opt_result = minimize(f, x0, method='BFGS')
                
            t_optimized_constants = opt_result['x']
            t_optimized_obj = opt_result['fun']

            optimized_constants.append(t_optimized_constants)

            # add validated data as the obj
            self.task.rand_draw_X_nonfixed()
            validate_obj = -self.task.reward_function_fixed_data(self)
            optimized_obj.append(validate_obj)

        optimized_obj = np.array(optimized_obj)
        optimized_constants = np.array(optimized_constants)

        # print('optimized_obj=', optimized_obj)
        # print('optimized_consts=', optimized_constants)
        # if obj close to zero, we get a good expression.
        # rember all the objective function and constant across all the iterations, so that we know which one is a real constant, which one is a variable.
        self.expr_objs = optimized_obj
        self.expr_consts = optimized_constants

        assert self.expr_objs.shape[0] == self.opt_num_expr
        assert len(self.expr_objs.shape) == 1

        # print('expr_objs=', self.expr_objs)
        # print('expr_consts=', self.expr_consts)

        # Set the optimized constants
        # set the value of optimized constants with the last optimized constants (the values of the constants may change, so only the last one makes sense; the mean does not make sense).
        self.set_constants(t_optimized_constants)

    def freeze_equation(self):
        if len(self.const_pos) == 0 or self.num_changing_consts == 0:
            assert "r" in self.__dict__
            if self.r >= -self.expr_obj_thres:
                for pos, t in enumerate(self.traversal):
                    self.allow_change_tokens[pos] = 0
            print("allow_change_tokens: {}".format(self.allow_change_tokens))
            return

        assert 'expr_objs' in self.__dict__
        # fitted objective  <= thereshold (residual is 0.01)
        #

        # the optimized result of negated reward should be smaller than the threshold
        # if use neg_mse as reward: max{(y-y_pred)^2} < threshold, 
        #     expr_obj_thres = 0.01
        # if use inv_mse as reward: max{-1/(1+(y-y_pred)^2)} < threshold 
        #     expr_obj_thres = - 0.99
        print("np.max(self.expr_objs) <= self.expr_obj_thres: {} {} {}".format(np.max(self.expr_objs) <= self.expr_obj_thres,
                                                                               self.expr_objs, self.expr_obj_thres))
        if np.max(self.expr_objs) <= self.expr_obj_thres:
            consts_tp = 0
            for pos, t in enumerate(self.traversal):
                # if t is a constant
                if isinstance(t, PlaceholderConstant):
                    if self.allow_change_tokens[pos] and \
                            np.std(self.expr_consts[consts_tp]) <= self.expr_consts_thres:
                        # std of x. 
                        # freeze it.
                        # the last step is allow to change (everything) x1 to x5 and every part of equation.
                        #
                        self.allow_change_tokens[pos] = 0
                else:
                    # residual is within threshold and is not a constant, freeze it.
                    self.allow_change_tokens[pos] = 0

            self.num_changing_consts = 0  # compute  num_changing_consts
            for pos in self.const_pos:
                if self.allow_change_tokens[pos]:
                    self.num_changing_consts += 1

    def get_constants(self):
        """Returns the values of a Program's constants."""

        return [t.value for t in self.traversal if isinstance(t, PlaceholderConstant)]

    def set_constants(self, consts):
        """Sets the program's constants to the given values"""
        consts_tp = 0
        for i, pos in enumerate(self.const_pos):
            # Create a new instance of PlaceholderConstant instead of changing
            # the "values" attribute, otherwise all Programs will have the same
            # instance and just overwrite each other's value.
            if self.allow_change_tokens[pos]:
                assert U.is_float(consts[consts_tp]), "Input to program constants must be of a floating point type"
                self.traversal[pos] = PlaceholderConstant(consts[consts_tp])
                consts_tp += 1

    @classmethod
    def set_n_objects(cls, n_objects):
        Program.n_objects = n_objects

    @classmethod
    def clear_cache(cls):
        """Clears the class' cache"""

        cls.cache = {}

    @classmethod
    def set_task(cls, task):
        """Sets the class' Task"""

        Program.task = task
        Program.library = task.library

    @classmethod
    def set_const_optimizer(cls, name, **kwargs):
        """Sets the class' constant optimizer"""

        const_optimizer = make_const_optimizer(name, **kwargs)
        Program.const_optimizer = const_optimizer

    @classmethod
    def set_complexity(cls, name):
        """Sets the class' complexity function"""

        all_functions = {
            # No complexity
            None: lambda p: 0.0,

            # Length of sequence
            "length": lambda p: len(p.traversal),

            # Sum of token-wise complexities
            "token": lambda p: sum([t.complexity for t in p.traversal]),

        }

        assert name in all_functions, "Unrecognzied complexity function name."

        Program.complexity_function = lambda p: all_functions[name](p)

    @classmethod
    def set_execute(cls, protected):
        """Sets which execute method to use"""

        # Check if cython_execute can be imported; if not, fall back to python_execute
        try:
            from dso import cyfunc
            from execute import cython_execute
            execute_function = cython_execute
            Program.have_cython = True
        except ImportError:
            from execute import python_execute
            execute_function = python_execute
            Program.have_cython = False

        if protected:
            Program.protected = True
            Program.execute_function = execute_function
        else:
            Program.protected = False

            class InvalidLog():
                """Log class to catch and record numpy warning messages"""

                def __init__(self):
                    self.error_type = None  # One of ['divide', 'overflow', 'underflow', 'invalid']
                    self.error_node = None  # E.g. 'exp', 'log', 'true_divide'
                    self.new_entry = False  # Flag for whether a warning has been encountered during a call to Program.execute()

                def write(self, message):
                    """This is called by numpy when encountering a warning"""

                    if not self.new_entry:  # Only record the first warning encounter
                        message = message.strip().split(' ')
                        self.error_type = message[1]
                        self.error_node = message[-1]
                    self.new_entry = True

                def update(self):
                    """If a floating-point error was encountered, set Program.invalid
                    to True and record the error type and error node."""

                    if self.new_entry:
                        self.new_entry = False
                        return True, self.error_type, self.error_node
                    else:
                        return False, None, None

            invalid_log = InvalidLog()
            np.seterrcall(invalid_log)  # Tells numpy to call InvalidLog.write() when encountering a warning

            # Define closure for execute function
            def unsafe_execute(traversal, X):
                """This is a wrapper for execute_function. If a floating-point error
                would be hit, a warning is logged instead, p.invalid is set to True,
                and the appropriate nan/inf value is returned. It's up to the task's
                reward function to decide how to handle nans/infs."""

                with np.errstate(all='log'):
                    y = execute_function(traversal, X)
                    invalid, error_node, error_type = invalid_log.update()
                    return y, invalid, error_node, error_type

            Program.execute_function = unsafe_execute

    @cached_property
    def r(self):
        """Evaluates and returns the reward of the program"""
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore")

            # print('===before optimize===')

            # Optimize any PlaceholderConstants
            self.optimize()

            # print('===after optimize===')

            # XYX: may need to average over multiple runs. 
            # Return final reward as the mean of multiple experiment runs.
            # Using another reward_function() evaluation seems not make sense.
            if 'expr_objs' in self.__dict__:
                return -np.mean(self.expr_objs)
            else:
                # this means there is no constants to be optimized.
                # return self.task.reward_function(self)
                self.expr_objs = []
                for expr in range(self.opt_num_expr):
                    self.task.rand_draw_data()
                    self.expr_objs.append(self.task.reward_function_fixed_data(self))
                self.expr_objs = np.array(self.expr_objs)
                return np.mean(self.expr_objs)

    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_function(self)

    @cached_property
    def evaluate(self):
        """Evaluates and returns the evaluation metrics of the program."""

        # Program must be optimized before computing evaluate
        if "r" not in self.__dict__:
            print("WARNING: Evaluating Program before computing its reward. Program will be optimized first.")
            self.optimize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return self.task.evaluate(self)

    @cached_property
    def sympy_expr(self):
        """
        Returns the attribute self.sympy_expr.

        This is actually a bit complicated because we have to go: traversal -->
        tree --> serialized tree --> SymPy expression
        """

        if Program.n_objects == 1:
            tree = self.traversal.copy()
            tree = build_tree(tree)
            tree = convert_to_sympy(tree)
            try:
                expr = parse_expr(tree.__repr__())  # SymPy expression
            except:
                expr = tree.__repr__()
            return [expr]
        else:
            exprs = []
            for i in range(len(self.traversals)):
                tree = self.traversals[i].copy()
                tree = build_tree(tree)
                tree = convert_to_sympy(tree)
                try:
                    expr = parse_expr(tree.__repr__())  # SymPy expression
                except:
                    expr = tree.__repr__()
                exprs.append(expr)
            return exprs

    def pretty(self):
        """Returns pretty printed string of the program"""
        return [pretty(self.sympy_expr[i]) for i in range(Program.n_objects)]

    def print_expression(self):
        print("\tExpression {}: {}".format(0, self.traversal))
        print("{}\n".format(self.pretty()[0]))


    def print_stats(self):
        """Prints the statistics of the program
        
            We will print the most honest reward possible when using validation.
        """

        print("\tReward: {}".format(self.r))
        print("\tOriginally on Policy: {}".format(self.originally_on_policy))
        print("\tInvalid: {}".format(self.invalid))
        print("\tTraversal: {}".format(self))

        if Program.n_objects == 1:
            print("\tExpression:")
            print("{}\n".format(indent(self.pretty()[0], '\t  ')))
        else:
            for i in range(Program.n_objects):
                print("\tExpression {}:".format(i))
                print("{}\n".format(indent(self.pretty()[i], '\t  ')))

    def __repr__(self):
        """Prints the program's traversal"""
        return ','.join([repr(t) for t in self.traversal])


###############################################################################
# Everything below this line is currently only being used for pretty printing #
###############################################################################


# Possible library elements that sympy capitalizes
capital = ["add", "mul", "pow"]


class Node(object):
    """Basic tree class supporting printing"""

    def __init__(self, val):
        self.val = val
        self.children = []

    def __repr__(self):
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return self.val  # Avoids unnecessary parantheses, e.g. x1()
        return "{}({})".format(self.val, children_repr)


def build_tree(traversal):
    """Recursively builds tree from pre-order traversal"""

    op = traversal.pop(0)
    n_children = op.arity
    val = repr(op)
    if val in capital:
        val = val.capitalize()

    node = Node(val)

    for _ in range(n_children):
        node.children.append(build_tree(traversal))

    return node


# this function is used for pretty print the expression
def convert_to_sympy(node):
    """Adjusts trees to only use node values supported by sympy"""

    if node.val == "div":
        node.val = "Mul"
        new_right = Node("Pow")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "sub":
        node.val = "Add"
        new_right = Node("Mul")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "inv":
        node.val = Node("Pow")
        node.children.append(Node("-1"))

    elif node.val == "neg":
        node.val = Node("Mul")
        node.children.append(Node("-1"))

    elif node.val == "n2":
        node.val = "Pow"
        node.children.append(Node("2"))

    elif node.val == "n3":
        node.val = "Pow"
        node.children.append(Node("3"))

    elif node.val == "n4":
        node.val = "Pow"
        node.children.append(Node("4"))

    for child in node.children:
        convert_to_sympy(child)

    return node
