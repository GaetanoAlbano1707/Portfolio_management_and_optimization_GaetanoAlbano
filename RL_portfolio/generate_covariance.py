import pickle
from data_loader import compute_rolling_covariance

# Path al file dei log return wide
log_return_path = "./TEST/log_returns_for_covariance.csv"

# Calcola la rolling covariance
cov_matrices = compute_rolling_covariance(log_return_path, window=60)

# Salva in pickle
with open("./TEST/cov_matrices.pkl", "wb") as f:
    pickle.dump(cov_matrices, f)

print("Covariance matrices salvate in ./TEST/cov_matrices.pkl")
