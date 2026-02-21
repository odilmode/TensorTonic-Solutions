import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    errors = []
    for i, y in enumerate(y_pred):
        errors.append(np.pow((y - y_true[i]), 2))
    mse = np.sum(errors) / len(errors)
    return mse