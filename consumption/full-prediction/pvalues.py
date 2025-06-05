import torch
import numpy as np
import scipy.stats as stats


def get_pvalues(model, X, y_true, y_pred):
    """
    Calculate p-values for the model predictions.

    Args:
        model: The trained model.
        x: Input features.
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        p_values: A tensor of p-values for each prediction.
    """
    n_samples = len(X)
    n_features = len(X[0][0])

    residuals = y_true - y_pred

    residual_var = (residuals**2).sum() / (n_samples - n_features - 1)

    # X is torch dataset, convert to numpy array
    X_np = X.convert_to_numpy()
    XtX_inv = np.linalg.inv(X_np.T @ X_np)

    se = np.sqrt(np.diagonal(residual_var * XtX_inv))

    weights = model.linear.weight.detach().numpy().flatten()
    t_vals = weights / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=n_samples - n_features - 1))

    return p_vals
