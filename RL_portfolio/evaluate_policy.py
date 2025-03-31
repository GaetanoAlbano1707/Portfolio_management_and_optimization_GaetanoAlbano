import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def evaluate_policy(
    policy_net,
    env_class,
    df: pd.DataFrame,
    initial_amount: float = 100000,
    device: str = "cpu",
    plot_results: bool = True,
    results_path: str = "./results/evaluation",
    cost_c_plus=None,
    cost_c_minus=None,
    cost_delta_plus=None,
    cost_delta_minus=None,
    reward_scaling: float = 1.0,
    **env_kwargs,
) -> dict:
    """
    Valuta un modello di policy su un dataset specificato in un ambiente RL.

    Args:
        policy_net: Modello PyTorch della policy.
        env_class: Classe dell'ambiente (es. PortfolioOptimizationEnv).
        df: Dataframe con i dati del test set.
        initial_amount: Capitale iniziale.
        device: Dispositivo su cui gira la rete.
        plot_results: Se True, salva grafici di valore portafoglio e reward.
        results_path: Percorso dove salvare i risultati.
        cost_*: Parametri per i costi di transazione.

    Returns:
        metrics (dict): dizionario con metriche finali dell’episodio.
    """
    env = env_class(
        df=df,
        initial_amount=initial_amount,
        reward_scaling=reward_scaling,
        cost_c_plus=cost_c_plus,
        cost_c_minus=cost_c_minus,
        cost_delta_plus=cost_delta_plus,
        cost_delta_minus=cost_delta_minus,
        print_metrics=False,
        plot_graphs=False,
        **env_kwargs,
    )

    state, _ = env.reset()
    done = False
    last_action = torch.tensor(np.array([1] + [0] * env.portfolio_size)).unsqueeze(0).to(device)

    portfolio_values = []
    rewards = []

    while not done:
        observation = torch.tensor(
            state["state"] if isinstance(state, dict) else state
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            action = policy_net(observation, last_action)
        state, reward, done, _, info = env.step(action.squeeze().cpu().numpy())
        last_action = action

        portfolio_values.append(env._portfolio_value)
        rewards.append(reward)

    # === Plot risultati ===
    results_dir = Path(results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.title("Portfolio Value During Evaluation")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig(results_dir / "portfolio_value.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="Reward")
    plt.title("Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(results_dir / "reward.png")
    plt.close()

    metrics = {
        "final_value": portfolio_values[-1],
        "fapv": portfolio_values[-1] / portfolio_values[0],
        "mean_reward": np.mean(rewards),
        "max_drawdown": max(1 - np.array(portfolio_values) / np.maximum.accumulate(portfolio_values)),
    }

    pd.DataFrame({
        "Portfolio Value": portfolio_values,
        "Reward": rewards
    }).to_csv(results_dir / "evaluation_results.csv", index=False)

    print(f"✅ Valutazione completata. FAPV: {metrics['fapv']:.4f}, Final Value: {metrics['final_value']:.2f}")
    return metrics
