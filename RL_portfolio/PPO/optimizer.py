import numpy as np
from cvxopt import matrix, solvers

def optimize_portfolio(mu, sigma, w_prev, gamma=1.0, cost_rate=0.001):
    n = len(mu)
    I = np.eye(n)

    P = matrix(sigma)
    q = matrix(-gamma * mu + cost_rate * np.abs(w_prev - mu))  # penalità proporzionale

    G = matrix(np.vstack([-I, I]))
    h = matrix(np.hstack([np.zeros(n), np.ones(n)]))

    A = matrix(np.ones((1, n)))
    b = matrix(1.0)

    try:
        sol = solvers.qp(P, q, G, h, A, b)
        return np.array(sol['x']).flatten()
    except Exception as e:
        raise RuntimeError(f"❌ Ottimizzazione base fallita: {str(e)}")
