import pandas as pd

# Carica i file
monthly = pd.read_csv(r"C:\Users\gaeta\Desktop\PORTFOLIO_MANAGEMENT_AND_OPTIMIZATION_GaetanoAlbano\RL_portfolio\PPO\models\monthly_best_configs\lambda_0.1\lr_0.0005_gamma_0.99_clip_0.2_entropy_0.0_cplus_0.0005_cminus_0.001_deltaplus_0.001_deltaminus_0.0005\results\rebalance_weights_test.csv", index_col=0)
quarterly = pd.read_csv(r"C:\Users\gaeta\Desktop\PORTFOLIO_MANAGEMENT_AND_OPTIMIZATION_GaetanoAlbano\RL_portfolio\PPO\models\quarterly_best_configs\lambda_0.1\lr_0.0005_gamma_0.95_clip_0.1_entropy_0.0_cplus_0.001_cminus_0.0005_deltaplus_0.001_deltaminus_0.0005\results\rebalance_weights_test.csv", index_col=0)

print("📆 Date rebalance mensile:", list(monthly.index))
print("📆 Date rebalance trimestrale:", list(quarterly.index))

# Confronta riga per riga
for date in monthly.index:
    print(f"\n🔍 Ribilanciamento del {date}:")
    print("Mensile:")
    print(monthly.loc[date].sort_values(ascending=False).head(3))
    if date in quarterly.index:
        print("Trimestrale:")
        print(quarterly.loc[date].sort_values(ascending=False).head(3))