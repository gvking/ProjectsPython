import numpy as np
from sgd import SGDSolver

def load_data(filename):
    """
    arg: filename - filename of file you want to load data from
            e.g. red_train.npy 
    Return: x values (numpy array: n x n)
            y values (numpy array: n x 1)
    """
    data = np.load(filename)
    data_x  =  data[:, [0,1,2,3,4,5,6,7,8,9,10]]
    data_y  =  data[:, [11]]
    # TODO: Finish this function here.
    return data_x, data_y

def red_wine_run(train_red_x, train_red_y, test_red_x, test_red_y):
    # Red wine data
    print('---------------\nRed Wine Data\n---------------')

    # Training Phase
    # values for 2D-grid search
    lam     = [0,0.01]        # regularization weight [min, max]
    alpha   = [0, 0.3]        # learning rate [min, max]
    nepochs = 50      # sample # of epochs
    epsilon = 0.0       # epsilon value
    param   = []
    # end TODO

    # using this alpha and lambda values run the training
    print(f"alpha: {alpha}, lambda:{lam}")
    print("Running Training phase")
    # return param and optimal values for alpha and lambda from SGDSolver
    param, alpha, lam = SGDSolver('Training', train_red_x, train_red_y, alpha, lam, nepochs, epsilon, param)

    # optimal values from 2-D search
    print(f"optimal values\nalpha: {alpha}, lambda: {lam}")

    # Note: validation and testing phases only take a single value for (alpha, lam) and not a list. 
    # Validation Phase
    x_mse_val = SGDSolver('Validation', test_red_x, test_red_y, alpha, lam, nepochs, epsilon, param)
    print(f"Current Red Wine Data MSE is: {x_mse_val}.")

    # Testing Phase
    red_wine_predicted = SGDSolver('Testing', test_red_x, test_red_y, alpha, lam, nepochs, epsilon, param)

    for i in range(0, 50):
        print(f"Predicted: {red_wine_predicted[i]}, Real: {test_red_y[i]}")

def white_wine_run(train_white_x, train_white_y, test_white_x, test_white_y):
    # White wine data
    print('---------------\nWhite Wine Data\n---------------')

    # TODO: Change hyperparameter values here as needed 
    # similar to red_wine_run
    # values for 2D-grid search
    lam     = [0,0.01]        # regularization weight [min, max]
    alpha   = [0, 0.3]        # learning rate [min, max]
    nepochs = 50        # sample # of epochs
    epsilon = 0.0       # epsilon value
    param   = []
    # end TODO

    # Training Phase
    print(f"alpha: {alpha}, lambda:{lam}")
    print("Running Training phase")
    # return param and optimal values for alpha and lambda from SGDSolver
    param, alpha, lam = SGDSolver('Training', train_white_x, train_white_y, alpha, lam, nepochs, epsilon, param)

    # optimal values from 2-D search
    print(f"optimal values\nalpha: {alpha}, lambda: {lam}")

    # Note: validation and testing phases only take a single value for (alpha, lam) and not a list. 
    # Validation Phase
    x_mse_val = SGDSolver('Validation', test_white_x, test_white_y, alpha, lam, nepochs, epsilon, param)
    print(f"Current White Wine Data MSE is: {x_mse_val}.")

    # Testing Phase
    white_wine_predicted = SGDSolver('Testing', test_white_x, test_white_y, alpha, lam, nepochs, epsilon, param)

    for i in range(0, 50):
        print(f"Predicted: {white_wine_predicted[i]}, Real: {test_white_y[i]}")

def main():
    # import all the data
    # TODO: call the load_data() function here and load data from file
    train_red_x, train_red_y        = load_data('hw2_winequality-red_train.npy')
    test_red_x, test_red_y          = load_data('hw2_winequality-red_test.npy')
    train_white_x, train_white_y    = load_data('hw2_winequality-white_train.npy')
    test_white_x, test_white_y      = load_data('hw2_winequality-white_test.npy')
    
    # Tests
    red_wine_run(train_red_x, train_red_y, test_red_x, test_red_y)
    white_wine_run(train_white_x, train_white_y, test_white_x, test_white_y)

if __name__ == "__main__":
    main()

    
