import numpy as np
from cvxopt import matrix, solvers

def optimize_portfolio_extended(mu, sigma, w_prev, gamma=1.0, c_minus=0.005, c_plus=0.005):
    n = len(mu)
    solvers.options['show_progress'] = False

    # === Q matrice (3n x 3n)
    Q = np.zeros((3*n, 3*n))
    Q[:n, :n] = sigma  # solo su w

    # === R vettore lineare
    r = np.zeros(3*n)
    r[:n] = -gamma * mu
    r[n:2*n] = gamma * c_minus
    r[2*n:] = gamma * c_plus

    # === Matrici dei vincoli (Ax = b)
    A = np.zeros((n+1, 3*n))
    b = np.zeros(n+1)

    # Vincolo di conservazione del portafoglio
    # w + Δw^- - Δw^+ = w̃
    A[:n, :n] = np.identity(n)
    A[:n, n:2*n] = np.identity(n)
    A[:n, 2*n:] = -np.identity(n)
    b[:n] = w_prev

    # Vincolo di budget: 1ᵀw + c⁻ᵀΔw⁻ + c⁺ᵀΔw⁺ = 1
    A[-1, :n] = 1
    A[-1, n:2*n] = c_minus
    A[-1, 2*n:] = c_plus
    b[-1] = 1

    # === Bounds: 0 ≤ x ≤ x+
    G = -np.identity(3*n)
    h = np.zeros(3*n)

    # Upper bounds (opzionale, per w ≤ 1)
    G_up = np.zeros((n, 3*n))
    G_up[:, :n] = np.identity(n)
    h_up = np.ones(n)

    # Stack matrici G e h
    G_total = np.vstack([G, G_up])
    h_total = np.hstack([h, h_up])

    # === Risoluzione QP
    Q_cvx = matrix(Q)
    r_cvx = matrix(r)
    G_cvx = matrix(G_total)
    h_cvx = matrix(h_total)
    A_cvx = matrix(A)
    b_cvx = matrix(b)

    sol = solvers.qp(Q_cvx, r_cvx, G_cvx, h_cvx, A_cvx, b_cvx)

    if sol['status'] != 'optimal':
        raise ValueError("⚠️ QP solver failed to find optimal solution.")

    x = np.array(sol['x']).flatten()
    w_opt = x[:n]
    return w_opt
