def export_rebalance_weights_csv(env, tickers, filename="portfolio_weights_rebalances.csv"):
    import pandas as pd
    import numpy as np

    weights = np.array(env.rebalance_weights)  # Assicurati che vengano salvati nell'env
    dates = [env.df.index.get_level_values("date")[step] for step in env.rebalance_steps]

    df_weights = pd.DataFrame(weights, columns=tickers)
    df_weights.insert(0, "date", dates)
    df_weights.insert(1, "step", env.rebalance_steps)

    df_weights.to_csv(filename, index=False)
    print(f"✅ Pesi al ribilanciamento salvati in: {filename}")
