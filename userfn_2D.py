import numpy as np
import GPy


def f(x):
    data = np.load('./model_2D.npz')
    dim = data['X'].shape[1]
    # ARD : One lengthscale parameter per dimension
    kernel = GPy.kern.StdPeriodic(input_dim=dim, ARD1=True, ARD2=True)
    mean_func = GPy.mappings.Constant(dim, 1)
    model = GPy.models.GPRegression(data['X'],
                                    data['Y'],
                                    kernel=kernel,
                                    mean_function=mean_func)
    model[:] = data['params']
    model.fix()
    model.parameters_changed()
    true_min = -203012.374
    return 23.061*model.predict(np.atleast_2d(x))[0] - true_min
