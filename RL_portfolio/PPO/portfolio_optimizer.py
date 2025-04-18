import numpy as np
import cvxpy as cp
import pandas as pd

def mean_variance_optimizer(mu, cov, gamma=1.0):
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu.T @ w - gamma * cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value

def mean_variance_with_costs(mu, cov, w_prev, c_minus, c_plus,
                              delta_minus=None, delta_plus=None,
                              gamma=1.0, cost_type='linear'):
    n = len(mu)
    w = cp.Variable(n)
    dw = w - w_prev

    if cost_type == 'linear':
        cost = cp.sum(cp.multiply(c_minus, cp.pos(-dw)) + cp.multiply(c_plus, cp.pos(dw)))
    elif cost_type == 'quadratic':
        cost = cp.sum(
            cp.multiply(c_minus, cp.pos(-dw)) + cp.multiply(c_plus, cp.pos(dw)) +
            cp.multiply(delta_minus, cp.square(cp.pos(-dw))) +
            cp.multiply(delta_plus, cp.square(cp.pos(dw)))
        )
    else:
        raise ValueError("Tipo di costo non riconosciuto: usa 'linear' o 'quadratic'")

    objective = cp.Maximize(mu.T @ w - gamma * cp.quad_form(w, cov) - cost)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value

def efficient_frontier(mu, cov, w_prev=None, cost_params=None, gamma_range=np.linspace(0.1, 10, 20), cost_type=None):
    frontier = []
    for gamma in gamma_range:
        if cost_type is None:
            weights = mean_variance_optimizer(mu, cov, gamma)
        else:
            weights = mean_variance_with_costs(
                mu, cov, w_prev=w_prev,
                c_minus=cost_params['c_minus'],
                c_plus=cost_params['c_plus'],
                delta_minus=cost_params.get('delta_minus', np.zeros_like(mu)),
                delta_plus=cost_params.get('delta_plus', np.zeros_like(mu)),
                gamma=gamma,
                cost_type=cost_type
            )
        exp_ret = mu @ weights
        risk = np.sqrt(weights.T @ cov @ weights)
        frontier.append((risk, exp_ret, weights))
    return frontier
