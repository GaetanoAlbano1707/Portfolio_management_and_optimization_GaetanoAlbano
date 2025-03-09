import os
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np

# Imposta il backend 'Agg' per evitare problemi di thread
matplotlib.use('Agg')



def analyze_correlation(tickers, start_date='2007-01-01', end_date='2024-12-23', save_dir='./Correlations/Correlation_Analysis/'):
    """
    Calcola e visualizza le correlazioni tra i rendimenti giornalieri (return) dei ticker.
    """
    # Controlla che la directory esista, altrimenti creala
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_date_dt = pd.to_datetime(start_date)  # Rimosso il parametro `format`
    end_date_dt = pd.to_datetime(end_date)

    # Scarica i dati
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, period='max')
        if data.empty:
            print(f"Errore per il ticker {ticker}: dati non disponibili.")
            continue
        data = data.sort_index()
        data = data.ffill().bfill()

        # Filtra i dati antecedenti alla data specificata
        data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
        data = data[data['Volume'] > 0]

        data['return'] = data['Adj Close'].pct_change() * 100
        all_data[ticker] = data['return']

    # Crea un DataFrame con i rendimenti di tutti i ticker
    return_data = pd.DataFrame(all_data)
    # Filtra i dati per eliminare periodi senza sovrapposizione
    return_data.dropna(axis=0, how='all', inplace=True)  # Rimuove righe senza dati per qualsiasi ticker

    # Matrice di correlazione
    correlation_matrix = return_data.corr()

    # Salva la matrice di correlazione come CSV
    correlation_matrix.to_csv(f"{save_dir}correlation_matrix.csv", float_format='%.4f')

    # Heatmap della matrice di correlazione
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title("Matrice di correlazione tra i rendimenti dei ticker")
    plt.tight_layout()
    plt.savefig(f"{save_dir}correlation_heatmap.png")
    plt.close()

    print(f"Matrice di correlazione salvata in {save_dir}")


def analyze_correlation_by_period(tickers, start_date='2007-01-01', end_date='2024-12-23', save_dir='./Correlations/Correlation_Analysis_Periods/'):
    """
    Calcola le correlazioni annuali e mensili tra i rendimenti giornalieri (return) dei ticker.
    Salva i risultati in file unici per le correlazioni annuali e mensili e genera heatmap annuali.
    """
    # Controlla che la directory esista, altrimenti creala
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Scarica i dati
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, period='max')
        if data.empty:
            print(f"Errore per il ticker {ticker}: dati non disponibili.")
            continue
        data = data.sort_index()
        data = data.ffill().bfill()


        # Filtra i dati antecedenti alla data specificata
        data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
        data = data[data['Volume'] > 0]

        data['return'] = data['Adj Close'].pct_change() * 100
        all_data[ticker] = data['return']

    # Crea un DataFrame con i rendimenti di tutti i ticker
    return_data = pd.DataFrame(all_data)
    return_data.dropna(axis=0, how='all', inplace=True)
    return_data['Year'] = return_data.index.year
    return_data['Month'] = return_data.index.month

    # Liste per memorizzare le correlazioni
    annual_correlations = []
    monthly_correlations = []

    # Correlazioni annuali
    for year, group in return_data.groupby('Year'):
        if group.isnull().values.all():
            continue
        annual_corr = group.drop(columns=['Year', 'Month']).corr()
        # Corregge il problema di correlazione 10000 tra tickers uguali
        for ticker in annual_corr.columns:
            annual_corr.loc[ticker, ticker] = 1.0
        # Salva la heatmap per l'anno
        plt.figure(figsize=(10, 8))
        sns.heatmap(annual_corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
        plt.title(f"Heatmap delle correlazioni annuali - Anno {year}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}heatmap_annual_{year}.png")
        plt.close()

        # Trasforma in formato lungo per il CSV
        annual_corr = annual_corr.stack().reset_index()
        annual_corr.columns = ['Ticker1', 'Ticker2', 'Correlation']
        annual_corr['Year'] = year
        annual_correlations.append(annual_corr)

    # Correlazioni mensili
    for (year, month), group in return_data.groupby(['Year', 'Month']):
        if group.isnull().values.all():
            continue
        monthly_corr = group.drop(columns=['Year', 'Month']).corr()
        # Corregge il problema di correlazione 10000 tra tickers uguali
        for ticker in monthly_corr.columns:
            monthly_corr.loc[ticker, ticker] = 1.0
        # Trasforma in formato lungo per il CSV
        monthly_corr = monthly_corr.stack().reset_index()
        monthly_corr.columns = ['Ticker1', 'Ticker2', 'Correlation']
        monthly_corr['Year'] = year
        monthly_corr['Month'] = month
        monthly_correlations.append(monthly_corr)

    # Combina i risultati in un unico DataFrame per annuali e mensili
    if annual_correlations:
        annual_correlation_df = pd.concat(annual_correlations, ignore_index=True)
        annual_correlation_df.to_csv(f"{save_dir}annual_correlations.csv", index=False, float_format='%.4f')
    if monthly_correlations:
        monthly_correlation_df = pd.concat(monthly_correlations, ignore_index=True)
        monthly_correlation_df.to_csv(f"{save_dir}monthly_correlations.csv", index=False, float_format='%.4f')

    print("Correlazioni annuali e mensili salvate in file CSV unici.")
    print("Heatmap annuali salvate nella directory specificata.")



def summarize_correlation_results(save_dir='./Correlations/Correlation_Analysis_Periods/'):
    """
    Riassume e analizza i risultati delle correlazioni annuali e mensili.
    Genera file per le correlazioni alte (≥ 0.7) e grafici delle medie e deviazioni standard.
    """
    # Percorso del file delle correlazioni annuali
    annual_file = os.path.join(save_dir, 'annual_correlations.csv')

    if os.path.exists(annual_file):
        # Legge i dati delle correlazioni annuali
        annual_correlations = pd.read_csv(annual_file)

        # Filtra le correlazioni alte (≥ 0.7) tra ticker diversi
        high_correlations = annual_correlations[
            (annual_correlations['Correlation'] >= 0.7) &
            (annual_correlations['Ticker1'] != annual_correlations['Ticker2'])
        ]

        # Salva le correlazioni alte in un file CSV
        high_correlations.to_csv(
            f"{save_dir}annual_high_correlations.csv",
            index=False,
            float_format='%.4f',
            sep=';',  # Separatore di colonna (punto e virgola)
            decimal=','  # Separatore decimale (virgola)
        )
        print(f"File 'annual_high_correlations.csv' salvato con le correlazioni ≥ 0.7.")

        # Analizza le medie e deviazioni standard delle correlazioni per anno
        annual_summary = annual_correlations.groupby('Year')['Correlation'].agg(['mean', 'std']).reset_index()

        # Grafico delle medie e deviazioni standard
        plt.figure(figsize=(10, 6))
        plt.plot(annual_summary['Year'], annual_summary['mean'], label='Media', marker='o', color='blue')
        plt.fill_between(
            annual_summary['Year'],
            annual_summary['mean'] - annual_summary['std'],
            annual_summary['mean'] + annual_summary['std'],
            color='blue',
            alpha=0.2,
            label='± Deviazione standard'
        )
        plt.title("Media e deviazione standard delle correlazioni annuali")
        plt.xlabel("Anno")
        plt.ylabel("Correlazione")
        plt.xticks(annual_summary['Year'], rotation=45)  # Mostra tutti gli anni sull'asse x
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{save_dir}annual_mean_std.png")
        plt.close()
        print(f"Grafico 'annual_mean_std.png' salvato.")

    else:
        print(f"File annuale '{annual_file}' non trovato. Verifica che l'analisi sia stata eseguita correttamente.")



def perform_pearson_test(tickers, start_date='2007-01-01', end_date='2024-12-23', save_dir='./Correlations/Correlation_Analysis_Pearson/'):
    """
    Esegue il test di Pearson sui rendimenti giornalieri (return) dei ticker e salva i risultati.
    """
    # Controlla che la directory esista, altrimenti creala
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Scarica i dati
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, period='max')
        if data.empty:
            print(f"Errore per il ticker {ticker}: dati non disponibili.")
            continue
        data = data.sort_index()
        data = data.ffill().bfill()

        # Filtra i dati antecedenti alla data specificata
        data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
        data = data[data['Volume'] > 0]

        data['return'] = data['Adj Close'].pct_change() * 100
        all_data[ticker] = data['return']

    # Crea un DataFrame con i rendimenti di tutti i ticker
    return_data = pd.DataFrame(all_data)
    return_data.dropna(inplace=True)

    # Esegue il test di Pearson per tutte le coppie di ticker
    results = []
    tickers_list = list(return_data.columns)
    for i in range(len(tickers_list)):
        for j in range(i + 1, len(tickers_list)):
            ticker1, ticker2 = tickers_list[i], tickers_list[j]
            r, p_value = pearsonr(return_data[ticker1], return_data[ticker2])
            results.append({
                'Ticker1': ticker1,
                'Ticker2': ticker2,
                'Pearson_Correlation': r,
                'P_Value': p_value
            })

    # Crea un DataFrame dei risultati
    results_df = pd.DataFrame(results)

    # Salva i risultati in un file CSV
    results_df.to_csv(
        f"{save_dir}pearson_test_results.csv",
        index=False,
        float_format='%.4f',
        sep=';',  # Separatore di colonna (punto e virgola)
        decimal=','  # Separatore decimale (virgola)
    )
    print(f"File 'pearson_test_results.csv' salvato in {save_dir}")

    # Heatmap delle correlazioni di Pearson
    pearson_matrix = return_data.corr(method='pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pearson_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title("Matrice di correlazione di Pearson tra i rendimenti dei ticker")
    plt.tight_layout()
    plt.savefig(f"{save_dir}pearson_correlation_heatmap.png")
    plt.close()
    print(f"Heatmap 'pearson_correlation_heatmap.png' salvata in {save_dir}")


def perform_pearson_test_periods(tickers, start_date='2007-01-01', end_date='2024-12-23', save_dir='./Correlations/Correlation_Analysis_Pearson_Periods/'):
    """
    Esegue il test di Pearson per i rendimenti giornalieri dei ticker su base annuale e mensile.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Scarica i dati
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, period='max')
        if data.empty:
            print(f"Errore per il ticker {ticker}: dati non disponibili.")
            continue
        data = data.sort_index()
        data = data.ffill().bfill()
        data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
        data = data[data['Volume'] > 0]

        data['return'] = data['Adj Close'].pct_change() * 100
        all_data[ticker] = data['return']

    return_data = pd.DataFrame(all_data)
    return_data.dropna(axis=0, how='all', inplace=True)

    # Aggiungi colonne per l'anno e il mese
    return_data['Year'] = return_data.index.year
    return_data['Month'] = return_data.index.month

    results_annual = []
    results_monthly = []

    # **Calcolo annuale**
    for year, group in return_data.groupby('Year'):
        tickers_list = group.columns[:-2]  # Escludi le colonne "Year" e "Month"
        for i in range(len(tickers_list)):
            for j in range(i + 1, len(tickers_list)):
                ticker1, ticker2 = tickers_list[i], tickers_list[j]
                combined = pd.concat([group[ticker1], group[ticker2]], axis=1, join='inner').dropna()
                if len(combined) >= 2:  # Verifica che ci siano abbastanza dati
                    r, p_value = pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
                    results_annual.append({
                        'Year': year,
                        'Ticker1': ticker1,
                        'Ticker2': ticker2,
                        'Pearson_Correlation': r,
                        'P_Value': p_value
                    })

    # **Calcolo mensile**
    for (year, month), group in return_data.groupby(['Year', 'Month']):
        tickers_list = group.columns[:-2]  # Escludi le colonne "Year" e "Month"
        for i in range(len(tickers_list)):
            for j in range(i + 1, len(tickers_list)):
                ticker1, ticker2 = tickers_list[i], tickers_list[j]
                combined = pd.concat([group[ticker1], group[ticker2]], axis=1, join='inner').dropna()
                if len(combined) >= 2:  # Verifica che ci siano abbastanza dati
                    r, p_value = pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
                    results_monthly.append({
                        'Year': year,
                        'Month': month,
                        'Ticker1': ticker1,
                        'Ticker2': ticker2,
                        'Pearson_Correlation': r,
                        'P_Value': p_value
                    })

    # Salva i risultati annuali
    results_annual_df = pd.DataFrame(results_annual)
    results_annual_df.to_csv(
        f"{save_dir}pearson_test_annual.csv",
        index=False,
        float_format='%.4f',
        sep=';',
        decimal=','
    )
    print(f"File annuale 'pearson_test_annual.csv' salvato in {save_dir}")

    # Salva i risultati mensili
    results_monthly_df = pd.DataFrame(results_monthly)
    results_monthly_df.to_csv(
        f"{save_dir}pearson_test_monthly.csv",
        index=False,
        float_format='%.4f',
        sep=';',
        decimal=','
    )
    print(f"File mensile 'pearson_test_monthly.csv' salvato in {save_dir}")



def plot_correlation_evolution(tickers, start_date='2007-01-01', end_date='2024-12-23', save_dir='./Correlations/Correlation_Evolution/'):
    """
    Crea un singolo PNG per ogni coppia di ticker, mostrando l'evoluzione
    delle correlazioni annuali. Evidenzia in rosso le coppie con correlazione
    media più alta, in verde quelle con la più bassa, e in blu tutte le altre.
    """

    # Assicura che la cartella esista
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Scarica i dati per ciascun ticker
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, period='max')
        if data.empty:
            print(f"Errore per il ticker {ticker}: dati non disponibili.")
            continue
        data = data.sort_index()
        data = data.ffill().bfill()  # Riempie i dati mancanti
        data = data[(data.index >= start_date_dt) & (data.index <= end_date_dt)]
        data = data[data['Volume'] > 0]
        data['return'] = data['Adj Close'].pct_change() * 100
        all_data[ticker] = data['return']

    # DataFrame con i rendimenti di tutti i ticker
    return_data = pd.DataFrame(all_data)
    return_data.dropna(axis=0, how='all', inplace=True)
    return_data['Year'] = return_data.index.year

    # Calcola le correlazioni annuali
    annual_correlations = []
    for year, group in return_data.groupby('Year'):
        cmat = group.corr()
        annual_correlations.append({'Year': year, 'Correlation_Matrix': cmat})

    # Crea tutte le coppie possibili (ticker1, ticker2)
    pairs = [
        (tickers[i], tickers[j])
        for i in range(len(tickers))
        for j in range(i + 1, len(tickers))
    ]

    # Calcola la correlazione media su tutto il periodo per ogni coppia
    pair_avg_correlation = {}
    for (ticker1, ticker2) in pairs:
        correlations = []
        for entry in annual_correlations:
            cmat = entry['Correlation_Matrix']
            if ticker1 in cmat.columns and ticker2 in cmat.columns:
                correlations.append(cmat.loc[ticker1, ticker2])
        # Gestione dei valori NaN
        if len(correlations) > 0:
            pair_avg_correlation[(ticker1, ticker2)] = np.nanmean(correlations)
        else:
            pair_avg_correlation[(ticker1, ticker2)] = np.nan

    # Ordina le coppie per correlazione media
    valid_pairs = {k: v for k, v in pair_avg_correlation.items() if not pd.isna(v)}
    sorted_pairs = sorted(valid_pairs.items(), key=lambda x: x[1])

    # Numero di coppie da evidenziare in top/bottom
    n_highlight = min(3, len(sorted_pairs) // 2)

    # Coppie con correlazione media più bassa
    lowest_pairs = set([p[0] for p in sorted_pairs[:n_highlight]])
    # Coppie con correlazione media più alta
    highest_pairs = set([p[0] for p in sorted_pairs[-n_highlight:]])

    # Crea un singolo file PNG per ogni coppia
    for (ticker1, ticker2) in pairs:
        # Recuperiamo i valori annuali di correlazione
        years = []
        correlations = []
        for entry in annual_correlations:
            year = entry['Year']
            cmat = entry['Correlation_Matrix']
            if ticker1 in cmat.columns and ticker2 in cmat.columns:
                years.append(year)
                correlations.append(cmat.loc[ticker1, ticker2])

        # Se non ci sono dati validi (correlations vuoto), salta
        if len(correlations) == 0:
            print(f"Nessuna correlazione valida per la coppia {ticker1} - {ticker2}")
            continue

        # Determina il colore in base alla correlazione media
        avg_corr = pair_avg_correlation[(ticker1, ticker2)]
        if (ticker1, ticker2) in highest_pairs:
            line_color = 'red'
        elif (ticker1, ticker2) in lowest_pairs:
            line_color = 'green'
        else:
            line_color = 'blue'

        # Crea la figura per questa coppia
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(years, correlations, marker='o', linestyle='-', color=line_color)
        if pd.isna(avg_corr):
            avg_corr_text = "Dati insufficienti"
        else:
            avg_corr_text = f"{avg_corr:.2f}"
        ax.set_title(f'{ticker1} - {ticker2}\nCorrelazione media: {avg_corr_text}', fontsize=10)
        ax.set_xlabel('Anno')
        ax.set_ylabel('Correlazione')
        ax.grid(True)
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)

        plt.tight_layout()

        # Nome del file PNG
        png_name = f"{ticker1}_{ticker2}.png"
        plt.savefig(os.path.join(save_dir, png_name), dpi=150)
        plt.close(fig)

    print(f"Grafici salvati in {save_dir}")



# Lista dei ticker
tickers = ['XLK', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI', 'NVDA', 'LMT', 'WMT', 'XOM', 'NKE', 'AMZN', 'NFLX', 'AAPL']

# Correlazione generale
analyze_correlation(tickers)

# Correlazione annuale e mensile
analyze_correlation_by_period(tickers)

# Esegui il riassunto
summarize_correlation_results()

#Test di Pearson sull'intero periodo
perform_pearson_test(tickers)

#Test di Pearson annuale e mensile
perform_pearson_test_periods(tickers)

#Grafico le evoluzioni delle correlazioni
plot_correlation_evolution(tickers)


