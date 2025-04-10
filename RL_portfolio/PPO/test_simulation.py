from simulate import simulate_portfolio
from data_loader import load_data, get_rebalancing_dates
from logger_weights import PortfolioLogger

def test_portfolio_run():
    df_prices, df_mu, df_sigma = load_data("./main_data_real.csv", "./expected_returns_real.csv", "./forecasting_data_combined.csv")
    rebal_dates = get_rebalancing_dates(sorted(df_prices['date'].unique()), freq='Q')
    logger = PortfolioLogger()
    series = simulate_portfolio(df_prices, df_mu, df_sigma, rebal_dates, gamma=1.0, cost_rate=0.01, logger=logger)
    logger.save()
    assert len(series) > 100, "Serie troppo corta!"
    print("✅ Test passato: portafoglio simulato e salvato correttamente.")

if __name__ == "__main__":
    test_portfolio_run()
