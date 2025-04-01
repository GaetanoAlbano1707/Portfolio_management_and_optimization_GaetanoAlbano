import pandas as pd


def calculate_rolling_correlation(df: pd.DataFrame, window: int = 63) -> dict:
    """
    Calcola la matrice di correlazione rolling ogni `window` giorni.
    Restituisce un dizionario {date: correlation_matrix}

    Args:
        df: DataFrame con MultiIndex (date, ticker) e almeno una colonna 'close'
        window: numero di giorni di trading per il rolling (es: 63 ≈ 3 mesi)

    Returns:
        dict con chiavi date e valori pd.DataFrame delle matrici di correlazione
    """
    # Pivottare i dati: righe = date, colonne = ticker, valori = close
    df_wide = df.reset_index().pivot(index="date", columns="tic", values="close")

    rolling_corr = {}
    for i in range(window - 1, len(df_wide)):
        date = df_wide.index[i]
        window_data = df_wide.iloc[i - window + 1: i + 1]
        corr_matrix = window_data.pct_change().dropna().corr()
        rolling_corr[date] = corr_matrix

    return rolling_corr
