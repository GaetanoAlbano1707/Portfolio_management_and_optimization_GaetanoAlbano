import numpy as np
from cvxopt import matrix, solvers

def optimize_portfolio(mu, sigma, w_prev, gamma=1.0, cost_rate=0.001):
    n = len(mu)
    I = np.identity(n)

    # QP formulation: min 0.5 x^T P x + q^T x
    P = matrix(sigma)
    q = matrix(-gamma * mu + cost_rate * np.sign(w_prev - mu))  # approx cost

    # Constraints Gx <= h
    G = matrix(np.vstack([-I, I]))
    h = matrix(np.hstack([np.zeros(n), np.ones(n)]))

    # Equality: Ax = b (sum to 1)
    A = matrix(np.ones((1, n)))
    b = matrix(1.0)

    sol = solvers.qp(P, q, G, h, A, b)
    return np.array(sol['x']).flatten()
