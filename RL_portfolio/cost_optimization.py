import itertools
import json
from policy_evaluator import evaluate_policy
from portfolio_optimization_env import PortfolioOptimizationEnv


def grid_search_transaction_costs(
    policy_net,
    df,
    cost_grid,
    evaluation_metric="fapv",
    initial_amount=100000,
    device="cpu",
    reward_scaling=1.0,
    model_name="EIIE",
    save_path="results/grid_search_results.json",  # <-- aggiornato
    env_kwargs=None,
):
    """
    Esegue una ricerca a griglia su coefficienti di costo per trovare la combinazione ottimale.

    Args:
        policy_net: modello da valutare
        df: dataframe dei dati di mercato
        cost_grid: dizionario con liste di valori per c⁺, c⁻, δ⁺, δ⁻
        evaluation_metric: metrica da ottimizzare (default: "fapv")
        initial_amount: capitale iniziale
        device: "cpu" o "cuda"
        reward_scaling: fattore di scala per i reward
        model_name: nome del modello per riferimenti
        save_path: path per salvare i risultati della grid search
        env_kwargs: argomenti extra da passare all’ambiente
    """
    best_score = float("-inf")
    best_combo = None
    results = []

    env_kwargs = env_kwargs or {}
    combos = list(itertools.product(
        cost_grid["c_plus"], cost_grid["c_minus"],
        cost_grid["delta_plus"], cost_grid["delta_minus"]
    ))

    for c_plus, c_minus, d_plus, d_minus in combos:
        print(f"Testing: c+={c_plus}, c-={c_minus}, δ+={d_plus}, δ-={d_minus}")

        n_assets_plus_cash = len(df["tic"].unique()) + 1
        cost_args = {
            "cost_c_plus": [c_plus] * n_assets_plus_cash,
            "cost_c_minus": [c_minus] * n_assets_plus_cash,
            "cost_delta_plus": [d_plus] * n_assets_plus_cash,
            "cost_delta_minus": [d_minus] * n_assets_plus_cash,
        }

        metrics = evaluate_policy(
            policy_net=policy_net,
            env_class=PortfolioOptimizationEnv,
            df=df,
            initial_amount=initial_amount,
            device=device,
            reward_scaling=reward_scaling,
            **cost_args,
            **env_kwargs
        )

        result = {
            "c_plus": c_plus,
            "c_minus": c_minus,
            "delta_plus": d_plus,
            "delta_minus": d_minus,
            "metric": metrics[evaluation_metric],
        }
        results.append(result)

        if result["metric"] > best_score:
            best_score = result["metric"]
            best_combo = result

    # ✅ Salva i risultati
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Best combo: {best_combo}")
    return best_combo, results
