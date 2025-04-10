import numpy as np
from cvxopt import matrix, solvers

def optimize_portfolio_extended(mu, sigma, w_prev, gamma=1.0, c_minus=0.005, c_plus=0.005):
    n = len(mu)
    solvers.options['show_progress'] = False

    Q = np.zeros((3*n, 3*n))
    Q[:n, :n] = sigma

    r = np.zeros(3*n)
    r[:n] = -gamma * mu
    r[n:2*n] = gamma * c_minus
    r[2*n:] = gamma * c_plus

    A = np.zeros((n+1, 3*n))
    b = np.zeros(n+1)

    A[:n, :n] = np.eye(n)
    A[:n, n:2*n] = np.eye(n)
    A[:n, 2*n:] = -np.eye(n)
    b[:n] = w_prev

    A[-1, :n] = 1
    A[-1, n:2*n] = c_minus
    A[-1, 2*n:] = c_plus
    b[-1] = 1

    G = -np.eye(3*n)
    h = np.zeros(3*n)

    G_up = np.zeros((n, 3*n))
    G_up[:, :n] = np.eye(n)
    h_up = np.ones(n)

    G_total = np.vstack([G, G_up])
    h_total = np.hstack([h, h_up])

    try:
        sol = solvers.qp(matrix(Q), matrix(r), matrix(G_total), matrix(h_total), matrix(A), matrix(b))
        if sol['status'] != 'optimal':
            raise ValueError("QP solution not optimal.")
        x = np.array(sol['x']).flatten()
        return x[:n]
    except Exception as e:
        raise RuntimeError(f"❌ Ottimizzazione fallita: {str(e)}")
