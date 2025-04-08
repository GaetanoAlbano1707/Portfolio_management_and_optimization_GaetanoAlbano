import numpy as np

def validate_against_frontier(weights, mu, cov_matrix):
    expected_return = np.dot(weights, mu)
    volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    return expected_return * 100, volatility * 100
