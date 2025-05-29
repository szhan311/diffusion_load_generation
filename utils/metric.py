import numpy as np

def calculate_quantile_forecasts(X_hat, quantiles):
    """
    Calculate the quantile forecasts from the scenarios.
    
    Parameters:
    - X_hat: The forecasted values with shape (730, 20, 96).
    - quantiles: A list of quantiles to calculate (e.g., [0.1, 0.5, 0.9]).
    
    Returns:
    - A dictionary of quantile forecasts with each key being the quantile and each value
      being the forecasted values with shape (730, 96).
    """
    quantile_forecasts = {}
    n_scenarios = X_hat.shape[1]
    scenario_indices = np.linspace(0, n_scenarios - 1, n_scenarios).astype(int)
    
    for q in quantiles:
        # Calculate the index of the scenario that represents the quantile
        scenario_index = int(np.ceil(q * (n_scenarios - 1)))
        sorted_scenarios = np.sort(X_hat, axis=1)
        quantile_forecasts[q] = sorted_scenarios[:, scenario_index, :]
    
    return quantile_forecasts

def calculate_pinball_loss(X, quantile_forecasts, quantiles):
    """
    Calculate the Pinball Loss for given quantile forecasts.
    
    Parameters:
    - X: The actual values with shape (730, 96).
    - quantile_forecasts: The forecasted quantile values.
    - quantiles: A list of quantiles for which the forecasts are calculated.
    
    Returns:
    - A dictionary of Pinball Losses with each key being the quantile.
    """
    pinball_losses = []
    for q in quantiles:
        forecasts = quantile_forecasts[q]
        errors = X - forecasts
        pinball_loss = np.where(errors >= 0, q * errors, (1 - q) * (-errors))
        pinball_losses.append(np.mean(pinball_loss))
    
    return pinball_losses

def pinball_Loss(y_true, y_pred):
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantile_forecasts = calculate_quantile_forecasts(y_pred, quantiles)
    pinball_loss = calculate_pinball_loss(y_true[:, 0, :], quantile_forecasts, quantiles)
    return np.array(pinball_loss).mean()

def gaussian_kernel(X, Y, sigma=1.0):
    """
    Computes the Gaussian (RBF) kernel between two sets of vectors.
    
    Parameters:
    - X, Y: numpy arrays of shapes (n_samples_X, n_features) and (n_samples_Y, n_features).
    - sigma: Bandwidth of the RBF kernel.
    
    Returns:
    - Kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    
    X_sqnorms = np.diagonal(XX)
    Y_sqnorms = np.diagonal(YY)
    
    K = np.exp(-0.5 * (X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY) / sigma**2)
    
    return K

def mmd_squared(X, Y, sigma=1.0):
    """
    Computes the squared Maximum Mean Discrepancy (MMD^2) between two samples.
    
    Parameters:
    - X, Y: numpy arrays of shapes (n_samples_X, n_features) and (n_samples_Y, n_features), samples from two distributions.
    - sigma: Bandwidth of the RBF kernel.
    
    Returns:
    - The squared MMD value.
    """
    K_XX = gaussian_kernel(X, X, sigma)
    K_YY = gaussian_kernel(Y, Y, sigma)
    K_XY = gaussian_kernel(X, Y, sigma)
    
    m = X.shape[0]
    n = Y.shape[0]
    
    mmd2 = (np.sum(K_XX) - np.sum(np.diag(K_XX))) / (m * (m - 1))
    mmd2 += (np.sum(K_YY) - np.sum(np.diag(K_YY))) / (n * (n - 1))
    mmd2 -= 2 * np.sum(K_XY) / (m * n)
    
    return mmd2
