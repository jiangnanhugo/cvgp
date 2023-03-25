import numpy as np
import scipy

class RegressTaskV1(object):
    """
    input parameters:

    batchsize:
    allowed_input: 1 if the input can be in the approximated expr. 0 cannot.
    n_input: num of vars in X
    true_program: the program to map from X to Y

    reward_function(self, p) # in reward function need to decide on 
                             # non-varying parameters

    evaluate(self, p)        # this is the inference task 
                               (evaluate the program on the test set).

    NOTE: nexpr should be left to program.optimize() (nexpr: number of experiments)
    """

    def __init__(self, batchsize, allowed_input, true_program, noise_std=0.0, metric='neg_mse', metric_params=(1.0,)):
        self.batchsize = batchsize
        self.allowed_input = allowed_input
        self.n_input = allowed_input.size
        self.true_program = true_program

        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]
        self.X_fixed = np.random.rand(self.n_input)
        self.metric_name = metric
        self.metric = make_regression_metric(metric, *metric_params)

        self.noise_std = noise_std

    def set_allowed_inputs(self, allowed_inputs):
        self.allowed_input = np.copy(allowed_inputs)
        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]

    def set_allowed_input(self, i, flag):
        self.allowed_input[i] = flag
        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]

    def rand_draw_X_fixed(self):
        self.X_fixed = np.random.rand(self.n_input)*9.5 + 0.5

    def rand_draw_data(self):
        self.X_fixed = np.random.rand(self.n_input)*9.5 + 0.5
        self.X = np.random.rand(self.batchsize, self.n_input)*9.5 + 0.5
        self.X[:, self.fixed_column] = self.X_fixed[self.fixed_column]

    def rand_draw_X_nonfixed(self):
        self.X = np.random.rand(self.batchsize, self.n_input)*9.5 + 0.5
        self.X[:, self.fixed_column] = self.X_fixed[self.fixed_column]        

    def reward_function_fixed_data(self, p):
        y_true = self.true_program.execute(self.X) + \
            np.random.normal(0.0, scale=self.noise_std, size=self.batchsize)
        self.y_true_out = y_true
        y_hat = p.execute(self.X)

        # print('X=', self.X.shape,self.X[0,:10])
        # print('y_true=', y_true.shape, y_true[:10])
        # print('y_hat=', y_hat)

        # return -np.mean((y_true - y_hat)**2)
        # print("VAR Y:", np.var(y_true))
        # Compute metric
        if self.metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            r = self.metric(y_true, y_hat, np.var(y_true))
        else:
            r = self.metric(y_true, y_hat)
        return r

    def reward_function_fixed_data_all_metrics(self, p):
        y_true = self.true_program.execute(self.X) + \
            np.random.normal(0.0, scale=self.noise_std, size=self.batchsize)
        self.y_true_out = y_true
        y_hat = p.execute(self.X)

        # print('X=', self.X.shape,self.X[0,:10])
        # print('y_true=', y_true.shape, y_true[:10])
        # print('y_hat=', y_hat)

        # return -np.mean((y_true - y_hat)**2)
        # print("VAR Y:", np.var(y_true))
        # Compute metric
        print('%' * 30)
        dict_of_result={}
        dict_of_result['y_true']=y_true
        dict_of_result['y_hat']=y_hat
        dict_of_result['diff'] = y_true-y_hat
        for metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            metric_params = (1.0,)
            metric = make_regression_metric(metric_name, *metric_params)
            r = metric(y_true, y_hat, np.var(y_true))
            dict_of_result[metric_name] = r
            print('{} {}'.format(metric_name, r))

        for metric_name in ['neg_mse', 'neg_rmse', 'neglog_mse', 'inv_mse']:
            metric_params = (1.0,)
            metric = make_regression_metric(metric_name, *metric_params)
            r = metric(y_true, y_hat)
            dict_of_result[metric_name] = r
            print('{} {}'.format(metric_name, r))
        # return r
        print('%' * 30)
        return dict_of_result

    def reward_function(self, p):
        # our program need the data generator.
        # p is a program.
        #
        X = np.random.rand(self.batchsize, self.n_input)*9.5 + 0.5
        # fixec colum coresponds to the fixed random variables. every time you use the same value
        X[:, self.fixed_column] = self.X_fixed[self.fixed_column]
        # the ground-truth program
        y_true = self.true_program.execute(X) + \
            np.random.normal(0.0, scale=self.noise_std, size=self.batchsize)
        # the model's
        y_hat = p.execute(X)


        if self.metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            r = self.metric(y_true, y_hat, np.var(y_true))
        else:
            r = self.metric(y_true, y_hat)
        return r


def make_regression_metric(name, *args):
    """
    Factory function for a regression metric. This includes a closures for
    metric parameters and the variance of the training data.

    Parameters
    ----------

    name : str
        Name of metric. See all_metrics for supported metrics.

    args : args
        Metric-specific parameters

    Returns
    -------

    metric : function
        Regression metric mapping true and estimated values to a scalar.
    """

    all_metrics = {

        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        "neg_mse": (lambda y, y_hat: -np.mean((y - y_hat) ** 2), 0),

        # Negative root mean squared error
        # Range: [-inf, 0]
        # Value = -sqrt(var(y)) when y_hat == mean(y)
        "neg_rmse": (lambda y, y_hat: -np.sqrt(np.mean((y - y_hat) ** 2)), 0),

        # Negative normalized mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nmse": (lambda y, y_hat, var_y: -np.mean((y - y_hat) ** 2) / var_y,
                     0),

        # Negative normalized root mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nrmse": (lambda y, y_hat, var_y: -np.sqrt(np.mean((y - y_hat) ** 2) / var_y), 0),

        # (Protected) negative log mean squared error
        # Range: [-inf, 0]
        # Value = -log(1 + var(y)) when y_hat == mean(y)
        "neglog_mse": (lambda y, y_hat: -np.log(1 + np.mean((y - y_hat) ** 2)), 0),

        # (Protected) inverse mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]*var(y)) when y_hat == mean(y)
        "inv_mse": (lambda y, y_hat: 1 / (1 + args[0] * np.mean((y - y_hat) ** 2)), 1),

        # (Protected) inverse normalized mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nmse": (lambda y, y_hat, var_y: 1 / (1 + args[0] * np.mean((y - y_hat) ** 2) / var_y), 1),

        # (Protected) inverse normalized root mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nrmse": (lambda y, y_hat, var_y: 1 / (1 + args[0] * np.sqrt(np.mean((y - y_hat) ** 2) / var_y)),
                      1),

        # Fraction of predicted points within p0*abs(y) + p1 band of the true value
        # Range: [0, 1]
        "fraction": (lambda y, y_hat: np.mean(abs(y - y_hat) < args[0] * abs(y) + args[1]),
                     2),

        # Pearson correlation coefficient
        # Range: [0, 1]
        "pearson": (lambda y, y_hat: scipy.stats.pearsonr(y, y_hat)[0],
                    0),

        # Spearman correlation coefficient
        # Range: [0, 1]
        "spearman": (lambda y, y_hat: scipy.stats.spearmanr(y, y_hat)[0],
                     0)
    }

    assert name in all_metrics, "Unrecognized reward function name."
    # assert len(args) == all_metrics[name][1], "For {}, expected {} reward function parameters; received {}.".format(name,all_metrics[name][1], len(args))
    metric = all_metrics[name][0]

    return metric
