import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def simulate_cost_frontier(cov_matrix, mean_returns, n_points=150, c_plus=0.01, c_minus=0.02, verbose=False):
    num_assets = len(mean_returns)
    results = {
        'Volatility_no_cost': [],
        'Return_no_cost': [],
        'Volatility_with_cost': [],
        'Return_with_cost': []
    }

    # === FUNZIONI UTILI ===
    def port_return(w): return np.dot(w, mean_returns)
    def port_vol(w): return np.sqrt(w.T @ cov_matrix @ w)
    def tc_cost(w):
        base = 1 / num_assets
        return c_plus * np.sum(np.maximum(w - base, 0)) + c_minus * np.sum(np.maximum(base - w, 0))

    def obj_no_tc(w): return port_vol(w)
    def obj_with_tc(w): return port_vol(w) + tc_cost(w)

    bounds = [(0, 1) for _ in range(num_assets)]
    base_weights = np.ones(num_assets) / num_assets

    # === RANGE DEI RENDIMENTI ATTESI ===
    base_ret = port_return(base_weights)
    max_ret = np.max(mean_returns) * 1.2  # stima conservativa
    target_returns = np.linspace(0, max_ret, n_points)

    for i, target in enumerate(target_returns):
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: port_return(w) - target}
        ]

        # === NO COST ===
        result_nc = minimize(obj_no_tc, base_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result_nc.success:
            w = result_nc.x
            results['Volatility_no_cost'].append(port_vol(w))
            results['Return_no_cost'].append(port_return(w))
        elif verbose:
            print(f"[⚠️ NC] Fallito target {target:.4f}")

        # === WITH COST ===
        result_wc = minimize(obj_with_tc, base_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result_wc.success:
            w = result_wc.x
            results['Volatility_with_cost'].append(port_vol(w))
            results['Return_with_cost'].append(port_return(w) - tc_cost(w))
        elif verbose:
            print(f"[⚠️ TC] Fallito target {target:.4f}")

    return results

def plot_efficient_frontier(results, c_plus, c_minus, filename="efficient_frontier_test.png"):
    vol_no_cost = np.array(results['Volatility_no_cost']) * 100
    ret_no_cost = np.array(results['Return_no_cost']) * 100
    vol_tc = np.array(results['Volatility_with_cost']) * 100
    ret_tc = np.array(results['Return_with_cost']) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(vol_no_cost, ret_no_cost, label='Senza Costi', color='blue', linewidth=2)
    plt.plot(vol_tc, ret_tc, label='Con Costi', color='red', linestyle='--', linewidth=2)

    plt.xlabel("Volatilità (%)", fontsize=12)
    plt.ylabel("Rendimento Atteso Netto (%)", fontsize=12)
    plt.title(f"Frontiera Efficiente (c⁺ = {c_plus*100:.2f}%, c⁻ = {c_minus*100:.2f}%)", fontsize=14)

    plt.xlim(left=0)
    plt.ylim(bottom=min(ret_tc.min(), ret_no_cost.min(), 0))

    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Grafico frontiera efficiente salvato in: {filename}")


def generate_multiple_frontiers(cov_matrix, mean_returns, configs, base_filename="efficient_frontier"):
    """
    Genera più grafici della frontiera efficiente per diversi valori di c+ e c-.
    """
    for c_plus, c_minus in configs:
        print(f"🔄 Calcolo frontiera con c⁺={c_plus*100:.1f}%, c⁻={c_minus*100:.1f}%")
        results = simulate_cost_frontier(cov_matrix, mean_returns, c_plus=c_plus, c_minus=c_minus)
        filename = f"{base_filename}_cp{int(c_plus*100)}_cm{int(c_minus*100)}.png"
        plot_efficient_frontier(results, c_plus=c_plus, c_minus=c_minus, filename=filename)
