# %% [markdown]
# # Fase 1: Pulizia e Preparazione del Dataset
# **Obiettivo**: Trasformare i dati grezzi in un dataset strutturato, pulito e affidabile.
# 
# **Perché è importante**: La qualità di qualsiasi analisi dei dati dipende in modo critico dalla qualità dei dati di input. Un preprocessing accurato ci permette di:
# - **Rimuovere il "rumore"**: Eliminare informazioni irrilevanti o errate che potrebbero distorcere i risultati.
# - **Standardizzare i formati**: Garantire che i dati siano coerenti (es. date, valori booleani).
# - **Gestire i dati mancanti**: Decidere strategicamente come trattare le lacune nel dataset per non perdere informazioni preziose.
# - **Creare feature significative**: Arricchire il dataset con nuove variabili (feature) che possono rivelare pattern nascosti.
# 
# In questa fase, ogni passaggio è documentato per spiegare cosa viene fatto e perché, garantendo la trasparenza e la riproducibilità dell'analisi.

# %% [markdown]
# ## 1. Setup dell'Ambiente
# **Obiettivo**: Caricare tutte le librerie necessarie e configurare le variabili globali.
# 
# **Cosa facciamo**:
# - Importiamo librerie fondamentali come `pandas` per la manipolazione dei dati, `matplotlib` e `seaborn` per le visualizzazioni, e `spacy` per il processamento del linguaggio naturale.
# - Definiamo costanti come le directory per i grafici (`PLOTS_DIR`) e il percorso del file di output (`CLEANED_DATA_PATH`) per mantenere il codice pulito e facilmente configurabile.
# - Inizializziamo il modello di linguaggio `en_core_web_sm` di spaCy, che ci servirà per la lemmatizzazione del testo.
# %% [code]
import pandas as pd
import numpy as np
import os
import re
import string
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from typing import List, Dict, Any

# Importa WordCloud e Plotly se disponibili, altrimenti imposta a None
try:
    from wordcloud import WordCloud
except ImportError:
    print("Libreria WordCloud non trovata. La generazione della word cloud sarà saltata.")
    WordCloud = None

try:
    import plotly.express as px
    import plotly.io as pio
    # Imposta un tema predefinito per Plotly per coerenza
    pio.templates.default = "plotly_white"
except ImportError:
    print("Libreria Plotly non trovata. I grafici interattivi saranno saltati.")
    px = None

# --- Costanti ---
PLOTS_DIR = 'plots'
CLEANED_DATA_PATH = 'Datasets/PPPData_EN_cleaned.csv'

# --- Setup Iniziale ---
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print('Downloading language model...')
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# --- Stile Globale Seaborn ---
# Scegli uno stile: "white", "ticks", "darkgrid", etc.
sns.set_style("ticks")  # Rimuove le griglie di sfondo, aggiunge i tick sugli assi
# Imposta una palette colori di default più moderna
sns.set_palette("tab10")  # Palette qualitativa
# Imposta il contesto per font e linee più grandi
sns.set_context("talk")

# %% [markdown]
# ## 2. Preprocessing Testuale
# **Obiettivo**: Creare una funzione standard per pulire e normalizzare qualsiasi campo di testo.
# 
# **Cosa facciamo** (`preprocess_text`):
# 1.  **Lowercase & Rimozione Punteggiatura/Numeri**: Convertiamo tutto in minuscolo e rimuoviamo punteggiatura e numeri per trattare parole come "Costruzione" e "costruzione." allo stesso modo.
# 2.  **Lemmatizzazione**: Utilizzando spaCy, riduciamo le parole alla loro forma base (es. "running" -> "run", "studies" -> "study"). Questo aggrega parole con lo stesso significato, riducendo la dimensionalità del testo.
# 3.  **Rimozione Stop Words**: Eliminiamo parole comuni ma poco informative (es. "the", "a", "is") che non aggiungono significato semantico.
# 
# **Risultato**: Una stringa di testo pulita, composta solo da parole chiave significative, pronta per essere analizzata o trasformata in feature numeriche.

# %% [code]
def preprocess_text(text: str) -> str:
    """Pulisce e normalizza una stringa di testo."""
    if not isinstance(text, str):
        return ""
    text = re.sub(f'[{re.escape(string.punctuation)}0-9]', '', text.lower())
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.strip()]
    return " ".join(lemmas)

# %% [markdown]
# ## 3. La Classe `DataPreprocessor`
# **Obiettivo**: Incapsulare tutta la logica di preprocessing in una classe per un approccio modulare, riutilizzabile e organizzato.
# 
# **Vantaggi**:
# - **Organizzazione**: Raggruppa tutte le funzioni correlate in un unico oggetto.
# - **Stato**: Mantiene lo stato del DataFrame (`self.df`) internamente, evitando di passare il DataFrame come argomento a ogni funzione.
# - **Riusabilità**: La classe può essere facilmente importata e utilizzata in altri script o notebook.

# %% [code]
class DataPreprocessor:
    """Classe per orchestrare il processo di pulizia e preparazione dei dati."""
    
    def __init__(self, file_path: str):
        self.df = self._load_data(file_path)

    def _load_data(self, path: str) -> pd.DataFrame:
        """Carica dati da file Excel o CSV con gestione errori robusta."""
        print(f"Caricamento dati da: {path}")
        try:
            # Prova a leggere come Excel, gestendo diverse estensioni
            if path.lower().endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
                return pd.read_excel(path)
            # Altrimenti prova come CSV
            elif path.lower().endswith('.csv'):
                # Prova diverse codifiche comuni se UTF-8 fallisce
                try:
                    return pd.read_csv(path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        return pd.read_csv(path, encoding='latin1')
                    except UnicodeDecodeError:
                        return pd.read_csv(path, encoding='iso-8859-1')
            else:
                raise ValueError("Formato file non supportato. Usa .xlsx, .xls o .csv")
        except FileNotFoundError:
            print(f"Errore: File non trovato a {path}")
            sys.exit(1)  # Esce dallo script se il file non viene trovato
        except Exception as e:
            print(f"Errore durante il caricamento del file {path}: {e}")
            sys.exit(1)

    def _save_plot(self, figure: plt.Figure, filename: str):
        """Salva il grafico nella directory PLOTS_DIR con qualità migliorata."""
        path = os.path.join(PLOTS_DIR, filename)
        try:
            # Salva con bbox_inches='tight' per evitare tagli e dpi alto per qualità
            figure.savefig(path, bbox_inches='tight', dpi=150)
            print(f"Grafico salvato in: {path}")
        except Exception as e:
            print(f"Errore durante il salvataggio del grafico {filename}: {e}")

    @staticmethod
    def _format_currency_axis(ax, axis='y', scale='auto'):
        """
        Formatta gli assi con valori monetari in modo leggibile.
        
        Parameters:
        - ax: matplotlib axis object
        - axis: 'x' o 'y' 
        - scale: 'auto', 'K' (migliaia), 'M' (milioni), 'B' (miliardi)
        """
        def format_func(value, tick_number):
            if scale == 'auto':
                if abs(value) >= 1e9:
                    return f'€{value/1e9:.1f}B'
                elif abs(value) >= 1e6:
                    return f'€{value/1e6:.1f}M'
                elif abs(value) >= 1e3:
                    return f'€{value/1e3:.0f}K'
                else:
                    return f'€{value:.0f}'
            elif scale == 'K':
                return f'€{value/1e3:.0f}K'
            elif scale == 'M':
                return f'€{value/1e6:.1f}M'
            elif scale == 'B':
                return f'€{value/1e9:.2f}B'
            return f'€{value:.0f}'
        
        if axis == 'y':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        else:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    def inspect_dataframe(self, title: str):
        """Ispeziona il DataFrame mostrando info, valori nulli e prime righe."""
        print(f"\n--- {title} ---")
        print("\n1. Informazioni Generali e Memoria:")
        self.df.info(memory_usage='deep')
        print("\n2. Valori Nulli per Colonna (Top 5):")
        # Mostra solo le colonne con valori nulli, se ce ne sono
        null_counts = self.df.isnull().sum()
        null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
        if not null_counts.empty:
            print(null_counts.head())
        else:
            print("Nessun valore nullo trovato.")
        print("\n3. Prime 5 Righe:")
        # Mostra più colonne se possibile
        with pd.option_context('display.max_columns', None):
            print(self.df.head())

    def analyze_missing_values(self):
        """Visualizza la percentuale di valori mancanti con stile migliorato."""
        missing_percentage = self.df.isnull().sum() * 100 / len(self.df)
        missing_df = pd.DataFrame({'column_name': self.df.columns,
                                   'percent_missing': missing_percentage})
        missing_df = missing_df[missing_df['percent_missing'] > 0].sort_values('percent_missing', ascending=False)

        if missing_df.empty:
            print("Nessuna colonna con valori mancanti da visualizzare.")
            return

        fig, ax = plt.subplots(figsize=(12, max(8, len(missing_df) * 0.5)))  # Altezza dinamica
        sns.barplot(x='percent_missing', y='column_name', data=missing_df,
                    ax=ax, palette='viridis', edgecolor='black', linewidth=0.8)
        ax.set_title('Percentuale di Valori Mancanti per Colonna (>0%)', fontsize=16, pad=20)
        ax.set_xlabel('Percentuale Mancante (%)', fontsize=12)
        ax.set_ylabel('Nome Colonna', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        sns.despine()
        fig.tight_layout()
        self._save_plot(fig, '01_missing_values_percentage.png')
        plt.show()
        plt.close(fig)

    def prune_columns(self, columns_to_drop: List[str]):
        """Rimuove le colonne specificate con gestione robusta."""
        print(f"\nRimuovendo {len(columns_to_drop)} colonne...")
        actual_cols_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        if len(actual_cols_to_drop) != len(columns_to_drop):
            print("Attenzione: Alcune colonne specificate per la rimozione non esistono nel DataFrame.")
        self.df.drop(columns=actual_cols_to_drop, inplace=True, errors='ignore')
        print(f"Colonne effettivamente rimosse: {len(actual_cols_to_drop)}")

    def clean_and_correct(self):
        """Pulisce e corregge dati con gestione errori migliorata."""
        print("\n--- Pulizia e Correzione Dati ---")
        # Conversione sicura a intero per 'Environmental criteria (T/F)'
        if 'Environmental criteria (T/F)' in self.df.columns:
            self.df['Environmental criteria (T/F)'] = pd.to_numeric(self.df['Environmental criteria (T/F)'], errors='coerce').fillna(0).astype(int)
            print("'Environmental criteria (T/F)' convertito a intero (0/1).")

        # Correzione e conversione sicura per 'Published in the EU journal'
        if 'Published in the EU journal' in self.df.columns:
            mapping = {False: 0, 'False': 0, 0: 0, '0': 0,
                       True: 1, 'True': 1, 'TRUE ': 1, 1: 1, '1': 1}
            self.df['Published in the EU journal'] = self.df['Published in the EU journal'].map(mapping)
            print("'Published in the EU journal' mappato a 0/1.")

        # Pulizia 'District'
        if 'District' in self.df.columns:
            self.df['District'] = self.df['District'].astype(str).str.strip()
            print("'District' ripulito da spazi extra.")

        # Rimozione righe incoerenti (se 'District Code' esiste)
        if 'District Code' in self.df.columns and 'District' in self.df.columns:
            initial_rows_district = len(self.df)
            self.df = self.df[~((self.df['District'] == 'Beja') & (self.df['District Code'] == 13))]
            self.df = self.df[~((self.df['District'] == 'Faro') & (self.df['District Code'] == 13))]
            print(f"Rimosse {initial_rows_district - len(self.df)} righe con codici distretto incoerenti.")
            self.df.drop(columns=['District Code'], inplace=True, errors='ignore')
            print("'District Code' rimosso.")

        # Gestione valori nulli in colonne chiave
        key_cols = ['Publication Year', 'Municipality', 'Base Bid Price (€)']
        actual_key_cols = [col for col in key_cols if col in self.df.columns]
        if actual_key_cols:
            initial_rows_nulls = len(self.df)
            self.df.dropna(subset=actual_key_cols, inplace=True)
            print(f"Rimosse {initial_rows_nulls - len(self.df)} righe con valori nulli in colonne chiave ({', '.join(actual_key_cols)}).")
        else:
            print("Attenzione: Nessuna delle colonne chiave specificate trovata per la rimozione dei nulli.")

    def engineer_date_features(self):
        """Crea feature basate sulle date con gestione robusta."""
        print("\nCreazione di feature basate sulle date...")
        processed_cols = []
        date_cols_to_process = ['Signing date', 'Closing date', 'Publication date']

        for date_col in date_cols_to_process:
            if date_col in self.df.columns:
                try:
                    self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce', dayfirst=True, infer_datetime_format=True)
                    year_col_name = f"{date_col.split()[0]} Year"
                    month_col_name = f"{date_col.split()[0]} Month"
                    self.df[year_col_name] = self.df[date_col].dt.year
                    self.df[month_col_name] = self.df[date_col].dt.month
                    processed_cols.append(date_col)
                    print(f"Elaborata colonna data: {date_col} -> {year_col_name}, {month_col_name}")
                except Exception as e:
                    print(f"Errore durante l'elaborazione della colonna data '{date_col}': {e}")
            else:
                print(f"Colonna data '{date_col}' non trovata, saltata.")

        # Rimuovi le colonne data originali se sono state processate
        if processed_cols:
            self.df.drop(columns=processed_cols, inplace=True, errors='ignore')
            print(f"Colonne data originali rimosse: {', '.join(processed_cols)}")

    def engineer_text_features(self, text_col: str, new_col_prefix: str, vectorizer: TfidfVectorizer):
        """Crea feature testuali con TF-IDF e gestione robusta."""
        print(f"\nCreazione di feature testuali da '{text_col}'...")
        if text_col not in self.df.columns:
            print(f"Colonna '{text_col}' non trovata. Impossibile creare feature testuali.")
            return

        cleaned_col = f"{text_col}_cleaned_internal"
        raw_store_col = f"{new_col_prefix}_raw_text"
        clean_store_col = f"{new_col_prefix}_clean_text"

        # Applica preprocessing
        self.df[cleaned_col] = self.df[text_col].fillna('').astype(str).apply(preprocess_text)
        self.df[raw_store_col] = self.df[text_col].astype(str)
        self.df[clean_store_col] = self.df[cleaned_col]

        # Trasformazione TF-IDF
        try:
            X_tfidf = vectorizer.fit_transform(self.df[cleaned_col])
            keywords = vectorizer.get_feature_names_out()
            print(f"Prime 10 keywords TF-IDF identificate: {', '.join(keywords[:10])}")

            # Crea colonne binarie per ogni keyword
            for keyword in keywords:
                safe_keyword = re.sub(r'\W+', '_', keyword).strip('_')
                col_name = f"{new_col_prefix}_keyword_{safe_keyword}"
                self.df[col_name] = self.df[cleaned_col].apply(lambda x: 1 if keyword in x.split() else 0)

            print(f"Create {len(keywords)} colonne keyword binarie da '{text_col}'.")
        except Exception as e:
            print(f"Errore durante la vettorizzazione TF-IDF per '{text_col}': {e}")
            if cleaned_col in self.df.columns:
                self.df.drop(columns=[cleaned_col], inplace=True)
            return

        # Rimuovi colonne originali
        self.df.drop(columns=[text_col, cleaned_col], inplace=True, errors='ignore')

    def process_numerical_features(self):
        """Elabora feature numeriche: outlier, discretizzazione, visualizzazione."""
        print("\n--- Elaborazione Feature Numeriche ---")
        numerical_cols = {
            'Diference between close and signing dates': ['Short', 'Medium', 'Long'],
            'Execution deadline (days)': ['Short', 'Medium', 'Long']
        }

        for col, labels in numerical_cols.items():
            if col not in self.df.columns:
                print(f"Colonna '{col}' non trovata, saltata.")
                continue

            # Conversione robusta a numerico
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            initial_nulls = self.df[col].isnull().sum()
            if initial_nulls > 0:
                print(f"Attenzione: {initial_nulls} valori non numerici in '{col}' convertiti a NaN.")

            # Visualizza distribuzione PRIMA della rimozione outlier
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(f"Distribuzione di '{col}' (Prima della Rimozione Outlier)", fontsize=16)
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[0], color='skyblue', edgecolor='black', alpha=0.7, bins=30)
            sns.boxplot(x=self.df[col].dropna(), ax=axes[1], color='lightcoral', linewidth=1.5)
            axes[0].set_title('Istogramma e KDE', fontsize=14)
            axes[1].set_title('Box Plot', fontsize=14)
            sns.despine(ax=axes[0])
            sns.despine(ax=axes[1], left=True)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_filename = f'02a_distribution_before_outlier_{col.replace(" ", "_").lower()}.png'
            self._save_plot(fig, plot_filename)
            plt.show()
            plt.close(fig)

            # Rimozione Outlier (IQR * 2.5)
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 2.5 * IQR, Q3 + 2.5 * IQR

            initial_rows = len(self.df)
            self.df = self.df[(self.df[col].isnull()) | ((self.df[col] >= lower_bound) & (self.df[col] <= upper_bound))]
            removed_count = initial_rows - len(self.df)
            if removed_count > 0:
                print(f"Rimosse {removed_count} righe considerate outlier per '{col}' (IQR*2.5). Limiti: [{lower_bound:.2f}, {upper_bound:.2f}]")

            # Salva la versione numerica
            numeric_col_name = f'{col}_numeric'
            self.df[numeric_col_name] = self.df[col]

            # Discretizzazione in quantili
            if self.df[col].notna().sum() > len(labels):
                try:
                    self.df[col] = pd.qcut(self.df[col], len(labels), labels=labels, duplicates='drop')
                    print(f"'{col}' discretizzata in categorie: {labels}.")
                except ValueError as e:
                    print(f"Errore durante la discretizzazione di '{col}': {e}. La colonna rimane numerica.")
                    self.df[col] = self.df[numeric_col_name]
            else:
                print(f"Non abbastanza dati validi in '{col}' per la discretizzazione basata su quantili.")
                self.df[col] = self.df[numeric_col_name]

        # Gestione 'Base Bid Price (€)'
        price_col = 'Base Bid Price (€)'
        if price_col in self.df.columns:
            self.df[price_col] = pd.to_numeric(self.df[price_col], errors='coerce')
            if self.df[price_col].notna().sum() > 3:
                try:
                    self.df[f'{price_col}_category'] = pd.qcut(self.df[price_col], 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                    print(f"'{price_col}' discretizzato in 'Low', 'Medium', 'High'.")
                except ValueError as e:
                    print(f"Errore discretizzazione '{price_col}': {e}. Creato '{price_col}_category' basato su mediane.")
                    median_price = self.df[price_col].median()
                    self.df[f'{price_col}_category'] = pd.cut(self.df[price_col], bins=[-np.inf, median_price, np.inf], labels=['Low', 'High'])
            else:
                print(f"Non abbastanza dati validi in '{price_col}' per la discretizzazione.")

        # Gestione 'Difference between the effective and initial price (€)'
        diff_price_col = 'Difference between the effective and initial price (€)'
        if diff_price_col in self.df.columns:
            diff_numeric = pd.to_numeric(self.df[diff_price_col], errors='coerce')
            self.df[f'{diff_price_col}_numeric'] = diff_numeric
            if diff_numeric.notna().sum() > 3:
                try:
                    self.df['Difference between the effective and initial price class'] = pd.qcut(diff_numeric, 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                    print(f"'{diff_price_col}' discretizzato in classi 'Low', 'Medium', 'High'.")
                except ValueError as e:
                    print(f"Errore discretizzazione '{diff_price_col}': {e}. Creato 'Difference... class' basato su zero.")
                    self.df['Difference between the effective and initial price class'] = pd.cut(diff_numeric, bins=[-np.inf, -0.01, 0.01, np.inf], labels=['Decrease', 'Stable', 'Increase'])
            else:
                print(f"Non abbastanza dati validi in '{diff_price_col}' per la discretizzazione.")

    def engineer_financial_features(self):
        """Crea feature finanziarie derivate come 'Price per Day'."""
        print("\n--- Creazione Feature Finanziarie Derivate ---")
        deadline_numeric_col = 'Execution deadline (days)_numeric'
        base_price_col = 'Base Bid Price (€)'
        price_per_day_col = 'Price per Day'

        if deadline_numeric_col not in self.df.columns or base_price_col not in self.df.columns:
            print(f"Errore: Colonne '{deadline_numeric_col}' o '{base_price_col}' mancanti.")
            return

        deadline_numeric = pd.to_numeric(self.df[deadline_numeric_col], errors='coerce')
        base_price = pd.to_numeric(self.df[base_price_col], errors='coerce')

        valid_mask = (deadline_numeric > 0) & deadline_numeric.notna() & base_price.notna()
        self.df[price_per_day_col] = np.nan
        self.df.loc[valid_mask, price_per_day_col] = base_price[valid_mask] / deadline_numeric[valid_mask]

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        num_calculated = valid_mask.sum()
        print(f"Calcolato '{price_per_day_col}' per {num_calculated}/{len(self.df)} righe valide.")

        if self.df[price_per_day_col].isnull().any():
            median_price_per_day = self.df[price_per_day_col].median()
            if pd.notna(median_price_per_day):
                self.df[price_per_day_col].fillna(median_price_per_day, inplace=True)
                print(f"Imputati valori mancanti di '{price_per_day_col}' con mediana: {median_price_per_day:.2f}")
            else:
                self.df[price_per_day_col].fillna(0, inplace=True)
                print(f"Attenzione: Impossibile calcolare mediana per '{price_per_day_col}'. Riempito con 0.")

    def impute_and_finalize(self):
        """Imputa valori mancanti finali e rimuove righe con NaN rimanenti."""
        print("\n--- Imputazione Finale e Finalizzazione ---")
        cols_to_impute_median = ['Submission deadline (days)', 'Classification of the multifactor criteria (%)']
        imputed_count = 0
        for col in cols_to_impute_median:
            if col in self.df.columns and self.df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    median_val = self.df[col].median()
                    if pd.notna(median_val):
                        self.df[col].fillna(median_val, inplace=True)
                        imputed_count += 1
                        print(f"Imputati valori nulli in '{col}' con mediana ({median_val:.2f}).")
                    else:
                        self.df[col].fillna(0, inplace=True)
                        print(f"Attenzione: Mediana per '{col}' è NaN. Imputato con 0.")
                else:
                    print(f"Attenzione: Colonna '{col}' non è numerica.")

        col_impute_mode = 'Published in the EU journal'
        if col_impute_mode in self.df.columns and self.df[col_impute_mode].isnull().any():
            if not self.df[col_impute_mode].mode().empty:
                mode_val = self.df[col_impute_mode].mode()[0]
                self.df[col_impute_mode].fillna(mode_val, inplace=True)
                imputed_count += 1
                print(f"Imputati valori nulli in '{col_impute_mode}' con moda ({mode_val}).")
            else:
                self.df[col_impute_mode].fillna(0, inplace=True)
                print(f"Attenzione: Impossibile trovare moda per '{col_impute_mode}'. Imputato con 0.")

        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        removed_rows = initial_rows - len(self.df)
        if removed_rows > 0:
            print(f"Rimosse {removed_rows} righe con valori NaN rimanenti.")
        else:
            print("Nessun valore NaN rimanente dopo l'imputazione.")

        print(f"Imputazione completata per {imputed_count} colonne.")

    def save_data(self, path: str):
        """Salva il DataFrame pulito in un file CSV."""
        try:
            self.df.to_csv(path, index=False, encoding='utf-8')
            print(f"\nDataset pulito e finale salvato con successo in: {path}")
            print(f"Dimensioni finali del dataset: {self.df.shape}")
        except Exception as e:
            print(f"Errore durante il salvataggio del dataset in {path}: {e}")

    def summarize_data(self):
        """Stampa un riepilogo statistico del DataFrame finale."""
        print("\n--- Riepilogo Statistico del Dataset Pulito ---")
        print(f"Numero totale di righe: {len(self.df)}")
        print(f"Numero totale di colonne: {len(self.df.columns)}")
        print("\nStatistiche Descrittive:")
        with pd.option_context('display.max_columns', None, 'display.width', 1000):
            try:
                print(self.df.describe(include='all').transpose())
            except Exception as e:
                print(f"Errore durante la generazione delle statistiche descrittive: {e}")

        print("\nTipi di Dati Finali delle Colonne:")
        print(self.df.dtypes)

    def compute_semantic_embeddings(self, text_col: str, prefix: str = 'semantic'):
        """Calcola embeddings semantici e riduce a 2D con PCA."""
        print(f"\n--- Calcolo Embedding Semantici per '{text_col}' ---")
        if text_col not in self.df.columns:
            print(f"Errore: Colonna '{text_col}' non trovata.")
            return False

        try:
            from sentence_transformers import SentenceTransformer
            print("Libreria sentence-transformers trovata.")
        except ImportError:
            print("Errore: Libreria 'sentence-transformers' non installata.")
            print("Per installarla, esegui: pip install sentence-transformers")
            return False

        sentences = self.df[text_col].fillna('').astype(str).tolist()
        if not sentences:
            print("Nessun testo trovato nella colonna specificata.")
            return False

        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        print(f"Caricamento modello: {model_name}...")
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Errore durante il caricamento del modello: {e}")
            return False

        print(f"Calcolo embeddings per {len(sentences)} testi...")
        try:
            embeddings = model.encode(sentences, show_progress_bar=True, batch_size=64)
            print(f"Embeddings calcolati. Dimensione: {embeddings.shape}")
        except Exception as e:
            print(f"Errore durante il calcolo degli embeddings: {e}")
            return False

        if embeddings.shape[1] > 2:
            print("Riduzione dimensionale con PCA a 2 componenti...")
            try:
                reducer = PCA(n_components=2, random_state=42)
                components = reducer.fit_transform(embeddings)
                print(f"PCA completata. Varianza spiegata: {reducer.explained_variance_ratio_.sum():.2%}")
                self.df[f'{prefix}_x'] = components[:, 0]
                self.df[f'{prefix}_y'] = components[:, 1]
            except Exception as e:
                print(f"Errore durante la PCA: {e}")
                return False
        elif embeddings.shape[1] == 2:
            print("Gli embeddings sono già a 2 dimensioni.")
            self.df[f'{prefix}_x'] = embeddings[:, 0]
            self.df[f'{prefix}_y'] = embeddings[:, 1]
        else:
            print("Attenzione: Dimensione embeddings < 2. Impossibile ridurre a 2D.")
            return False

        print(f"Colonne '{prefix}_x' e '{prefix}_y' aggiunte al DataFrame.")
        return True

    def perform_text_clustering(self, n_clusters: int = 5, random_state: int = 42, prefix: str = 'cpvs_sem'):
        """Esegue il clustering K-Means sugli embeddings semantici."""
        print(f"\n--- Esecuzione Clustering K-Means ({n_clusters} cluster) ---")
        x_col, y_col = f'{prefix}_x', f'{prefix}_y'
        cluster_col = f"{prefix.split('_')[0]}_cluster"

        if x_col not in self.df.columns or y_col not in self.df.columns:
            print(f"Errore: Colonne embedding '{x_col}' o '{y_col}' non trovate.")
            return

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("Errore: Libreria 'scikit-learn' non installata.")
            return

        embedding_data = self.df[[x_col, y_col]].dropna()
        if embedding_data.empty:
            print("Nessun dato valido per il clustering.")
            return

        print(f"Esecuzione K-Means su {len(embedding_data)} punti dati...")
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding_data)
            self.df[cluster_col] = pd.Series(cluster_labels, index=embedding_data.index)
            self.df[cluster_col] = self.df[cluster_col].astype('category')
            print("Clustering K-Means completato.")
            print(f"Distribuzione dei cluster:\n{self.df[cluster_col].value_counts()}")
        except Exception as e:
            print(f"Errore durante l'esecuzione di K-Means: {e}")

    def visualize_text_clusters(self, prefix: str = 'cpvs_sem'):
        """Visualizza i cluster semantici con Plotly."""
        print("\n--- Visualizzazione Cluster Semantici ---")
        x_col, y_col = f'{prefix}_x', f'{prefix}_y'
        cluster_col = f"{prefix.split('_')[0]}_cluster"
        raw_text_col = f"{prefix.split('_')[0]}_raw_text"

        if px is None:
            print("Plotly non disponibile. Salto visualizzazione cluster.")
            return
        if cluster_col not in self.df.columns or x_col not in self.df.columns or y_col not in self.df.columns:
            print(f"Errore: Colonne necessarie non trovate.")
            return

        plot_data = self.df.dropna(subset=[x_col, y_col, cluster_col]).copy()
        plot_data[cluster_col] = plot_data[cluster_col].astype(str)

        if plot_data.empty:
            print("Nessun dato valido da visualizzare per i cluster.")
            return

        num_clusters = plot_data[cluster_col].nunique()
        color_sequence = px.colors.qualitative.Vivid[:num_clusters] if num_clusters <= len(px.colors.qualitative.Vivid) else px.colors.qualitative.Plotly

        hover_cols = [raw_text_col] if raw_text_col in plot_data.columns else None

        try:
            fig = px.scatter(
                plot_data,
                x=x_col,
                y=y_col,
                color=cluster_col,
                title=f'Cluster Semantici ({cluster_col.split("_")[0].upper()}) - K-Means',
                hover_data=hover_cols,
                category_orders={cluster_col: sorted(plot_data[cluster_col].unique())},
                color_discrete_sequence=color_sequence
            )

            fig.update_layout(
                xaxis_title='Componente Principale 1 (Semantica)',
                yaxis_title='Componente Principale 2 (Semantica)',
                legend_title_text='Cluster ID',
                title_font_size=20,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7))

            cluster_filename = f'13_{cluster_col}_semantic_clusters.html'
            cluster_path = os.path.join(PLOTS_DIR, cluster_filename)
            fig.write_html(cluster_path)
            print(f"Grafico interattivo dei cluster semantici salvato in: {cluster_path}")
            fig.show()

        except Exception as e:
            print(f"Errore durante la creazione del grafico Plotly dei cluster: {e}")

    def analyze_award_criteria_and_price(self):
        """Genera visualizzazioni per analizzare criteri di aggiudicazione e prezzi."""
        print("\n--- Analisi Approfondita: Criteri di Aggiudicazione e Prezzi ---")

        award_col = 'Award criteria class'
        price_col = 'Base Bid Price (€)'
        district_col = 'District'
        deadline_col = 'Execution deadline (days)_numeric'

        required_cols = [award_col, price_col]
        if not all(col in self.df.columns for col in required_cols):
            print(f"Errore: Colonne '{award_col}' o '{price_col}' mancanti.")
            return

        # 1. Box Plot: Prezzo per Criterio
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(
            x=award_col,
            y=price_col,
            data=self.df,
            palette='Set2',
            ax=ax,
            hue=award_col,
            linewidth=1.5,
            legend=False
        )
        ax.set_title('Distribuzione Prezzo Base per Criterio di Aggiudicazione', fontsize=18, pad=15)
        ax.set_ylabel('Prezzo Base (€) (Scala Logaritmica)', fontsize=14)
        ax.set_xlabel('Criterio di Aggiudicazione', fontsize=14)
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=15, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        sns.despine()
        fig.tight_layout()
        self._save_plot(fig, '20_price_distribution_by_award_criteria.png')
        plt.show()
        plt.close(fig)

        # 2. Bar Plot: Conteggio Criteri per Distretto
        if district_col in self.df.columns:
            fig, ax = plt.subplots(figsize=(15, max(10, self.df[district_col].nunique() * 0.4)))
            order = self.df[district_col].value_counts().index
            sns.countplot(
                y=district_col,
                hue=award_col,
                data=self.df,
                order=order,
                palette='viridis',
                edgecolor='grey',
                linewidth=0.5,
                ax=ax
            )
            ax.set_title('Contratti per Criterio di Aggiudicazione e Distretto', fontsize=18, pad=15)
            ax.set_xlabel('Numero di Contratti', fontsize=14)
            ax.set_ylabel('Distretto', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.legend(title='Criterio Aggiudicazione', title_fontsize='13', fontsize='12')
            sns.despine()
            fig.tight_layout()
            self._save_plot(fig, '21_award_criteria_by_district.png')
            plt.show()
            plt.close(fig)

        # 3. KDE Plot: Prezzo vs. Scadenza per Criterio
        if deadline_col in self.df.columns:
            plot_data_kde = self.df[(self.df[price_col] > 0) & (self.df[deadline_col] > 0)].copy()
            if not plot_data_kde.empty:
                g = sns.displot(
                    data=plot_data_kde,
                    x=price_col,
                    y=deadline_col,
                    hue=award_col,
                    kind='kde',
                    fill=True,
                    height=8, aspect=1.2,
                    palette='viridis',
                    log_scale=(True, True)
                )
                g.fig.suptitle('Densità Prezzo vs. Scadenza per Criterio (Scala Log)', y=1.03, fontsize=18)
                g.set_axis_labels('Prezzo Base (€)', 'Scadenza Esecuzione (giorni)', fontsize=14)
                g._legend.set_title('Criterio Aggiudicazione')
                g.fig.tight_layout(rect=[0, 0.03, 1, 0.98])
                self._save_plot(g.fig, '22_price_vs_deadline_density_by_award_criteria.png')
                plt.show()
                plt.close(g.fig)
            else:
                print("Dati insufficienti per grafico densità Prezzo vs Scadenza.")

    def generate_final_visualizations(self):
        print("\n--- Generazione Visualizzazioni Finali ---")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.countplot(y='District', data=self.df, order=self.df['District'].value_counts().index, palette='viridis', ax=ax, hue='District', legend=False)
        ax.set_title('Numero di Contratti per Distretto')
        self._save_plot(fig, '03_district_distribution.png')
        plt.show()
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribuzioni di Feature Chiave', fontsize=16)
        plot_specs = [
            {'col': 'Award criteria class', 'ax': axes[0, 0], 'title': 'Classe Criteri di Aggiudicazione'},
            {'col': 'Base Bid Price (€)_category', 'ax': axes[0, 1], 'title': 'Prezzo Base (Discretizzato)'},
            {'col': 'Execution deadline (days)', 'ax': axes[1, 0], 'title': 'Scadenza Esecuzione (Discretizzato)'},
            {'col': 'Diference between close and signing dates', 'ax': axes[1, 1], 'title': 'Differenza Date (Discretizzata)'}
        ]
        for spec in plot_specs:
            sns.countplot(x=spec['col'], data=self.df, palette='magma', ax=spec['ax'], hue=spec['col'], legend=False)
            spec['ax'].set_title(spec['title'])
            spec['ax'].tick_params(axis='x', rotation=45)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_plot(fig, '04_key_features_distribution.png')
        plt.show()
        plt.close(fig)

    def generate_geospatial_visualizations(self, geojson_path: str = 'Datasets/portugal_districts.geojson'):
        if px is None: return
        geojson_file = Path(geojson_path)
        if not geojson_file.exists(): return
        with geojson_file.open('r', encoding='utf-8') as f:
            districts_geojson = json.load(f)
        geo_metrics = self.df.groupby('District').agg(
            contracts=('District', 'count'),
            base_bid_mean=('Base Bid Price (€)', 'mean')
        ).reset_index()
        fig_map = px.choropleth_mapbox(
            geo_metrics, geojson=districts_geojson, locations='District',
            featureidkey='properties.name', color='base_bid_mean',
            color_continuous_scale='Viridis', mapbox_style='carto-positron',
            zoom=5.5, center={'lat': 39.5, 'lon': -8.0}, opacity=0.7,
            hover_data={'contracts': True, 'base_bid_mean': ':.0f'}
        )
        fig_map.update_layout(title='Valore Medio del Prezzo Base per Distretto')
        fig_map.write_html(os.path.join(PLOTS_DIR, '05_map_base_bid_by_district.html'))
        fig_map.show()
        print("Mappa coropletica salvata.")

    def generate_advanced_visualizations(self):
        """Genera visualizzazioni avanzate professionali e di impatto."""
        print("\n--- Generazione Visualizzazioni Avanzate ---")
        
        award_col = 'Award criteria class'
        price_col = 'Base Bid Price (€)'
        year_col = 'Signing Year'
        
        # 1. Donut Chart Moderno per Criteri
        if award_col in self.df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            award_counts = self.df[award_col].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            
            wedges, texts, autotexts = ax.pie(
                award_counts, labels=None, autopct='%1.1f%%',
                startangle=90, colors=colors[:len(award_counts)],
                pctdistance=0.85, explode=[0.05]*len(award_counts),
                textprops={'weight': 'bold', 'size': 13}
            )
            
            # Donut effect
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax.add_artist(centre_circle)
            
            # Legenda elegante
            ax.legend(wedges, award_counts.index, title="Criteri di Aggiudicazione",
                     loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                     fontsize=11, title_fontsize=13)
            
            ax.set_title('Distribuzione Criteri di Aggiudicazione', 
                        fontsize=20, weight='bold', pad=20)
            
            # Aggiungi totale al centro
            total = len(self.df)
            ax.text(0, 0, f'{total:,}\nContratti', ha='center', va='center',
                   fontsize=18, weight='bold', color='#2C3E50')
            
            plt.tight_layout()
            self._save_plot(fig, '11_award_criteria_donut.png')
            plt.show()
            plt.close(fig)

        # 2. Violin Plot Migliorato con annotazioni
        if award_col in self.df.columns and price_col in self.df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Usa sample per performance se dataset grande
            plot_data = self.df.sample(min(10000, len(self.df)), random_state=42)
            
            parts = ax.violinplot(
                [plot_data[plot_data[award_col] == cat][price_col].dropna().values / 1e6
                 for cat in plot_data[award_col].unique()],
                positions=range(len(plot_data[award_col].unique())),
                showmeans=True, showmedians=True, widths=0.7
            )
            
            # Colora violini
            colors_viol = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_viol[i % len(colors_viol)])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Stile marker
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(2)
            
            ax.set_xticks(range(len(plot_data[award_col].unique())))
            ax.set_xticklabels(plot_data[award_col].unique(), rotation=15, ha='right', fontsize=11)
            ax.set_ylabel('Prezzo Base (Milioni €)', fontsize=14, weight='bold')
            ax.set_title('Distribuzione Prezzi per Criterio di Aggiudicazione', 
                        fontsize=18, weight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Aggiungi annotazioni statistiche
            for i, cat in enumerate(plot_data[award_col].unique()):
                median_val = plot_data[plot_data[award_col] == cat][price_col].median() / 1e6
                ax.text(i, ax.get_ylim()[1] * 0.95, f'Mediana:\n€{median_val:.1f}M',
                       ha='center', va='top', fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                edgecolor='gray', alpha=0.8))
            
            sns.despine()
            plt.tight_layout()
            self._save_plot(fig, '12_price_distribution_violin_enhanced.png')
            plt.show()
            plt.close(fig)

        # 3. Heatmap Correlazione Migliorata
        corr_cols = [price_col, 'Execution deadline (days)_numeric', 
                     'Diference between close and signing dates_numeric', 
                     'Difference between the effective and initial price (€)_numeric', year_col]
        valid_corr_cols = [col for col in corr_cols if col in self.df.columns 
                          and pd.api.types.is_numeric_dtype(self.df[col])]

        if len(valid_corr_cols) > 1:
            corr_matrix = self.df[valid_corr_cols].corr()
            
            # Abbrevia nomi per leggibilità
            col_abbrev = {
                price_col: 'Prezzo Base',
                'Execution deadline (days)_numeric': 'Scadenza',
                'Diference between close and signing dates_numeric': 'Diff. Date',
                'Difference between the effective and initial price (€)_numeric': 'Var. Prezzo',
                year_col: 'Anno'
            }
            corr_matrix.rename(columns=col_abbrev, index=col_abbrev, inplace=True)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Maschera triangolo superiore
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                       center=0, linewidths=2, linecolor='white',
                       cbar_kws={'label': 'Correlazione di Pearson', 'shrink': 0.8},
                       ax=ax, mask=mask, vmin=-1, vmax=1,
                       annot_kws={'size': 12, 'weight': 'bold'})
            
            ax.set_title('Matrice di Correlazione - Feature Numeriche', 
                        fontsize=20, weight='bold', pad=20)
            
            # Ruota etichette
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
            
            plt.tight_layout()
            self._save_plot(fig, '13_correlation_heatmap_enhanced.png')
            plt.show()
            plt.close(fig)

    def generate_price_intensity_plot(self):
        """Genera grafico 2D della relazione prezzo/intensità con annotazioni chiare."""
        print("\n--- Generazione Grafico Intensità Prezzo ---")
        
        price_col = 'Base Bid Price (€)'
        price_day_col = 'Price per Day'
        
        if price_day_col not in self.df.columns or price_col not in self.df.columns:
            print(f"Colonne mancanti. Salto grafico intensità.")
            return

        # Filtra e prepara dati
        plot_data = self.df[
            (self.df[price_col] > 0) & 
            (self.df[price_day_col] > 0)
        ][[price_col, price_day_col]].dropna().copy()

        if plot_data.empty or len(plot_data) < 10:
            print("Dati insufficienti per grafico intensità.")
            return

        # Converti a migliaia/milioni per leggibilità
        plot_data['price_k'] = plot_data[price_col] / 1000
        plot_data['price_day_units'] = plot_data[price_day_col]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Contour plot invece di hexbin per più eleganza
        try:
            # Crea griglia per contour
            from scipy.stats import gaussian_kde
            
            # Sample per performance
            if len(plot_data) > 5000:
                plot_sample = plot_data.sample(5000, random_state=42)
            else:
                plot_sample = plot_data
            
            x = plot_sample['price_k'].values
            y = plot_sample['price_day_units'].values
            
            # Kernel density estimation
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            
            # Griglia
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
            
            # Contour plot
            contour = ax.contourf(xi, yi, zi, levels=15, cmap='YlOrRd', alpha=0.8)
            ax.contour(xi, yi, zi, levels=15, colors='white', linewidths=0.5, alpha=0.3)
            
            # Scatter overlay
            ax.scatter(x, y, c='navy', s=10, alpha=0.1, edgecolors='none')
            
            # Colorbar
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Densità Contratti', fontsize=13, weight='bold')
            
        except Exception as e:
            print(f"Fallback a scatter plot: {e}")
            # Fallback a scatter semplice
            ax.scatter(plot_data['price_k'], plot_data['price_day_units'],
                      c='navy', alpha=0.3, s=20, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('Prezzo Base Contratto (Migliaia €)', fontsize=14, weight='bold')
        ax.set_ylabel('Intensità: Prezzo per Giorno (€/giorno)', fontsize=14, weight='bold')
        ax.set_title('Analisi Intensità Progetto: Valore vs. Costo Giornaliero', 
                    fontsize=18, weight='bold', pad=20)
        
        # Aggiungi griglie di riferimento
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax.set_axisbelow(True)
        
        # Formattazione assi
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:.0f}K'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'€{y:.0f}'))
        
        # Aggiungi annotazioni con quartili
        q1 = plot_data['price_k'].quantile(0.25)
        q3 = plot_data['price_k'].quantile(0.75)
        median = plot_data['price_k'].median()
        
        ax.axvline(median, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mediana: €{median:.0f}K')
        ax.axvline(q1, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Q1: €{q1:.0f}K')
        ax.axvline(q3, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Q3: €{q3:.0f}K')
        
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        # Box informativi
        info_text = f'Progetti analizzati: {len(plot_data):,}\nRange prezzo: €{plot_data["price_k"].min():.0f}K - €{plot_data["price_k"].max():.0f}K'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        sns.despine()
        plt.tight_layout()
        self._save_plot(fig, '18_price_intensity_analysis.png')
        plt.show()
        plt.close(fig)
        print("Grafico intensità prezzo salvato.")
        
    def generate_role_based_visualizations(self):
        print("\n--- Generazione Visualizzazioni per Ruoli Specifici ---")
        financial_df = self.df.groupby(['Signing Year', 'Award criteria class'])['Base Bid Price (€)'].sum().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(14, 8))
        financial_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_title('Valore Totale Contratti per Anno e Criterio')
        ax.set_ylabel('Valore Totale Base Bid Price (€)')
        ax.set_xlabel('Anno di Firma')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Criterio di Aggiudicazione')
        self._save_plot(fig, '14_financial_stacked_bar_value_by_year.png')
        plt.show()
        plt.close(fig)

        if 'Published in the EU journal' in self.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Published in the EU journal', y='Base Bid Price (€)', data=self.df, ax=ax)
            ax.set_title('Confronto Prezzo Base per Pubblicazione in Gazzetta EU')
            ax.set_ylabel('Prezzo Base (€) (Scala Logaritmica)')
            ax.set_xlabel('Pubblicato in Gazzetta EU (1=Sì, 0=No)')
            ax.set_yscale('log')
            self._save_plot(fig, '15_financial_price_vs_eu_publication.png')
            plt.show()
            plt.close(fig)

        manager_df = self.df.groupby('District')['Execution deadline (days)_numeric'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 8))
        manager_df.plot(kind='barh', ax=ax, color=sns.color_palette('coolwarm', len(manager_df)))
        ax.set_title('Scadenza Media di Esecuzione per Distretto')
        ax.set_xlabel('Giorni Medi di Esecuzione')
        ax.set_ylabel('Distretto')
        self._save_plot(fig, '16_manager_avg_deadline_by_district.png')
        plt.show()
        plt.close(fig)

    def generate_additional_geospatial_plot(self, geojson_path: str = 'Datasets/portugal_districts.geojson'):
        if px is None or not Path(geojson_path).exists(): return
        with open(geojson_path, 'r', encoding='utf-8') as f:
            districts_geojson = json.load(f)
        geo_metrics = self.df.groupby('District').size().reset_index(name='contracts')
        fig_map = px.choropleth_mapbox(
            geo_metrics, geojson=districts_geojson, locations='District',
            featureidkey='properties.name', color='contracts',
            color_continuous_scale='Plasma', mapbox_style='carto-positron',
            zoom=5.5, center={'lat': 39.5, 'lon': -8.0}, opacity=0.7,
            hover_data={'contracts': True}
        )
        fig_map.update_layout(title='Numero di Contratti per Distretto')
        map_path = os.path.join(PLOTS_DIR, '17_map_contracts_by_district.html')
        fig_map.write_html(map_path)
        fig_map.show()
        print(f"Mappa coropletica del numero di contratti salvata.")

    def generate_word_cloud(self):
        if WordCloud and 'cpvs_clean_text' in self.df.columns:
            text = ' '.join(self.df['cpvs_clean_text'])
            if text.strip():
                cloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(cloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud delle Keyword CPVS')
                self._save_plot(plt.gcf(), '06_wordcloud_cpvs.png')
                plt.show()
                plt.close()

    def generate_temporal_heatmap(self):
        """Genera heatmap temporale: contratti per mese e anno."""
        print("\n--- Generazione Heatmap Temporale ---")
        year_col = 'Signing Year'
        month_col = 'Signing Month'
        
        if year_col not in self.df.columns or month_col not in self.df.columns:
            print(f"Colonne '{year_col}' o '{month_col}' mancanti. Salto heatmap temporale.")
            return

        # Crea pivot table: anni x mesi
        temporal_data = self.df.groupby([year_col, month_col]).size().reset_index(name='contracts')
        pivot_table = temporal_data.pivot(index=month_col, columns=year_col, values='contracts').fillna(0)
        
        # Ordina mesi correttamente
        month_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 
                       'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
        pivot_table.index = [month_names[int(i)-1] if i <= 12 else str(i) for i in pivot_table.index]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', 
                    linewidths=0.5, linecolor='white', cbar_kws={'label': 'Numero Contratti'},
                    ax=ax, vmin=0)
        ax.set_title('Intensità Temporale: Numero di Contratti per Mese e Anno', 
                     fontsize=20, pad=20, weight='bold')
        ax.set_xlabel('Anno di Firma', fontsize=14, weight='bold')
        ax.set_ylabel('Mese', fontsize=14, weight='bold')
        ax.tick_params(axis='both', labelsize=11)
        plt.tight_layout()
        self._save_plot(fig, '07_temporal_heatmap.png')
        plt.show()
        plt.close(fig)
        print("Heatmap temporale salvata.")

    def generate_stacked_area_chart(self):
        """Genera stacked area chart dell'evoluzione del valore totale dei contratti."""
        print("\n--- Generazione Stacked Area Chart ---")
        year_col = 'Signing Year'
        price_col = 'Base Bid Price (€)'
        award_col = 'Award criteria class'
        
        if not all(c in self.df.columns for c in [year_col, price_col, award_col]):
            print("Colonne necessarie mancanti. Salto stacked area chart.")
            return

        # Aggrega valore per anno e criterio
        temporal_value = self.df.groupby([year_col, award_col])[price_col].sum().reset_index()
        pivot_value = temporal_value.pivot(index=year_col, columns=award_col, values=price_col).fillna(0)
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Stacked area con colori moderni
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        pivot_value.plot(kind='area', stacked=True, ax=ax, alpha=0.8, 
                         color=colors[:len(pivot_value.columns)], linewidth=2)
        
        # Formattazione asse Y per milioni
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1e6:.1f}M'))
        
        ax.set_title('Evoluzione del Valore Totale dei Contratti nel Tempo', 
                     fontsize=22, pad=20, weight='bold')
        ax.set_xlabel('Anno di Firma', fontsize=14, weight='bold')
        ax.set_ylabel('Valore Totale Contratti', fontsize=14, weight='bold')
        ax.legend(title='Criterio Aggiudicazione', title_fontsize=12, fontsize=11,
                  loc='upper left', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=11)
        sns.despine()
        plt.tight_layout()
        self._save_plot(fig, '08_stacked_area_value_evolution.png')
        plt.show()
        plt.close(fig)
        print("Stacked area chart salvato.")

    def generate_treemap_budget_distribution(self):
        """Genera treemap della distribuzione del budget per distretto e criterio."""
        print("\n--- Generazione Treemap Distribuzione Budget ---")
        
        if px is None:
            print("Plotly non disponibile. Salto treemap.")
            return
            
        district_col = 'District'
        award_col = 'Award criteria class'
        price_col = 'Base Bid Price (€)'
        
        if not all(c in self.df.columns for c in [district_col, award_col, price_col]):
            print("Colonne necessarie mancanti. Salto treemap.")
            return

        # Aggrega dati
        treemap_data = self.df.groupby([district_col, award_col])[price_col].sum().reset_index()
        treemap_data.columns = ['District', 'Criterion', 'Total_Value']
        treemap_data = treemap_data[treemap_data['Total_Value'] > 0]
        
        # Crea treemap
        fig = px.treemap(
            treemap_data,
            path=['District', 'Criterion'],
            values='Total_Value',
            title='Distribuzione Budget: Valore Contratti per Distretto e Criterio',
            color='Total_Value',
            color_continuous_scale='Viridis',
            hover_data={'Total_Value': ':,.0f'}
        )
        
        fig.update_layout(
            title_font_size=22,
            title_font_family='Arial Black',
            font=dict(size=13),
            height=800
        )
        
        fig.update_traces(
            textinfo='label+value',
            textfont_size=12,
            marker=dict(line=dict(width=2, color='white'))
        )
        
        treemap_path = os.path.join(PLOTS_DIR, '09_treemap_budget_distribution.html')
        fig.write_html(treemap_path)
        fig.show()
        print(f"Treemap interattivo salvato in: {treemap_path}")

    def generate_kpi_dashboard(self):
        """Genera dashboard con KPI principali e gauge charts."""
        print("\n--- Generazione Dashboard KPI ---")
        
        price_col = 'Base Bid Price (€)'
        deadline_col = 'Execution deadline (days)_numeric'
        
        if not all(c in self.df.columns for c in [price_col, deadline_col]):
            print("Colonne necessarie mancanti. Salto dashboard KPI.")
            return

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # KPI Cards Style
        def create_kpi_card(ax, value, title, subtitle, color):
            ax.axis('off')
            ax.text(0.5, 0.65, value, ha='center', va='center', 
                   fontsize=48, weight='bold', color=color, family='monospace')
            ax.text(0.5, 0.35, title, ha='center', va='center', 
                   fontsize=16, weight='bold', color='#2C3E50')
            ax.text(0.5, 0.15, subtitle, ha='center', va='center', 
                   fontsize=11, color='#7F8C8D', style='italic')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            # Background
            rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                facecolor='white', edgecolor=color, 
                                linewidth=3, transform=ax.transAxes)
            ax.add_patch(rect)
        
        # KPI 1: Totale Contratti
        ax1 = fig.add_subplot(gs[0, 0])
        total_contracts = len(self.df)
        create_kpi_card(ax1, f'{total_contracts:,}', 'Contratti Totali', 
                       'Dataset analizzato', '#3498DB')
        
        # KPI 2: Valore Medio
        ax2 = fig.add_subplot(gs[0, 1])
        avg_value = self.df[price_col].mean()
        create_kpi_card(ax2, f'€{avg_value/1e6:.2f}M', 'Valore Medio', 
                       'Per contratto', '#2ECC71')
        
        # KPI 3: Durata Media
        ax3 = fig.add_subplot(gs[0, 2])
        avg_deadline = self.df[deadline_col].mean()
        create_kpi_card(ax3, f'{avg_deadline:.0f}', 'Giorni Medi', 
                       'Scadenza esecuzione', '#E74C3C')
        
        # Grafico 4: Top 5 Distretti per Valore
        ax4 = fig.add_subplot(gs[1, :2])
        if 'District' in self.df.columns:
            top_districts = self.df.groupby('District')[price_col].sum().nlargest(5)
            colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, 5))
            bars = ax4.barh(range(len(top_districts)), top_districts.values, color=colors_bar)
            ax4.set_yticks(range(len(top_districts)))
            ax4.set_yticklabels(top_districts.index, fontsize=11, weight='bold')
            ax4.set_xlabel('Valore Totale (€)', fontsize=12, weight='bold')
            ax4.set_title('Top 5 Distretti per Valore Contratti', fontsize=14, weight='bold', pad=10)
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1e6:.0f}M'))
            ax4.grid(axis='x', alpha=0.3, linestyle='--')
            # Aggiungi valori sulle barre
            for i, (bar, value) in enumerate(zip(bars, top_districts.values)):
                ax4.text(value + top_districts.max()*0.02, i, f'€{value/1e6:.1f}M',
                        va='center', fontsize=10, weight='bold')
            sns.despine(ax=ax4, left=True)
        
        # Grafico 5: Distribuzione Prezzi (violino compatto)
        ax5 = fig.add_subplot(gs[1, 2])
        if 'Award criteria class' in self.df.columns:
            # Prendi sample per performance
            sample_data = self.df.sample(min(5000, len(self.df)), random_state=42)
            parts = ax5.violinplot(
                [sample_data[sample_data['Award criteria class'] == cat][price_col].dropna().values 
                 for cat in sample_data['Award criteria class'].unique()],
                positions=range(len(sample_data['Award criteria class'].unique())),
                showmeans=True, showmedians=True
            )
            for pc in parts['bodies']:
                pc.set_facecolor('#FF6B6B')
                pc.set_alpha(0.7)
            ax5.set_xticks([])
            ax5.set_ylabel('Prezzo (€)', fontsize=11, weight='bold')
            ax5.set_title('Distribuzione Prezzi', fontsize=13, weight='bold', pad=8)
            ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1e6:.1f}M'))
            ax5.grid(axis='y', alpha=0.3, linestyle='--')
            sns.despine(ax=ax5)
        
        # Grafico 6: Timeline compatta
        ax6 = fig.add_subplot(gs[2, :])
        if 'Signing Year' in self.df.columns:
            yearly = self.df.groupby('Signing Year').agg({
                price_col: 'sum',
                'Signing Year': 'size'
            }).rename(columns={'Signing Year': 'count'})
            
            ax6_twin = ax6.twinx()
            
            line1 = ax6.plot(yearly.index, yearly['count'], marker='o', linewidth=3, 
                            markersize=8, color='#3498DB', label='Numero Contratti')
            ax6_twin.plot(yearly.index, yearly[price_col]/1e9, marker='s', linewidth=3,
                         markersize=8, color='#E74C3C', linestyle='--', label='Valore Totale (€Mld)')
            
            ax6.set_xlabel('Anno', fontsize=12, weight='bold')
            ax6.set_ylabel('Numero Contratti', fontsize=11, weight='bold', color='#3498DB')
            ax6_twin.set_ylabel('Valore Totale (€ Miliardi)', fontsize=11, weight='bold', color='#E74C3C')
            ax6.set_title('Andamento Temporale: Volume e Valore', fontsize=14, weight='bold', pad=10)
            ax6.tick_params(axis='y', labelcolor='#3498DB')
            ax6_twin.tick_params(axis='y', labelcolor='#E74C3C')
            ax6.grid(True, alpha=0.3, linestyle='--')
            
            # Legenda combinata
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
            
            sns.despine(ax=ax6, right=False)
        
        plt.suptitle('Dashboard Riepilogativa - Appalti Pubblici Portoghesi', 
                    fontsize=24, weight='bold', y=0.98)
        
        fig.patch.set_facecolor('#F8F9FA')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_plot(fig, '10_kpi_dashboard.png')
        plt.show()
        plt.close(fig)
        print("Dashboard KPI salvata.")

    def generate_pdf_report(self, report_title: str = 'PPP Portugal - Report di Analisi Narrativa', output_path: str = None):
        """Genera un report PDF contenente i grafici PNG salvati e descrizioni narrative."""
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.lib.pagesizes import landscape, A4
            import textwrap
            from PIL import Image as PILImage
        except ImportError:
            print("\nErrore: Libreria 'reportlab' e/o 'Pillow' non installate.")
            print("Per installarle, esegui: pip install reportlab Pillow")
            return

        if output_path is None:
            output_path = os.path.join(PLOTS_DIR, 'PPP_narrative_report.pdf')
        print(f"\n--- Generazione Report PDF Narrativo: {output_path} ---")

        # Mappatura dei nomi dei file dei grafici a titoli e descrizioni
        plot_info = {
            '01_missing_values_percentage.png': {
                "title": "Analisi dei Valori Mancanti",
                "description": "Questo grafico mostra la percentuale di dati mancanti per ogni colonna. Colonne con un'alta percentuale di valori nulli (es. 'Conclusion of a framework agreement') sono state rimosse perché non avrebbero fornito insight affidabili. Questa fase è cruciale per garantire la robustezza dell'analisi."
            },
            '03_district_distribution.png': {
                "title": "Distribuzione Geografica dei Contratti",
                "description": "Il grafico illustra il numero totale di contratti per distretto. Emerge una chiara concentrazione nelle aree metropolitane di Lisbona e Porto, che sono i principali centri economici del paese. Questo suggerisce che la maggior parte delle attività di costruzione si concentra in queste due regioni."
            },
            '04_key_features_distribution.png': {
                "title": "Distribuzione delle Feature Chiave",
                "description": "Questi grafici mostrano come si distribuiscono alcune delle variabili più importanti. La maggior parte dei contratti usa il 'prezzo più basso' come criterio di aggiudicazione. Le distribuzioni di prezzo e scadenze sono ampie, indicando una grande varietà di progetti, da piccoli lavori di manutenzione a grandi opere infrastrutturali."
            },
            '09_award_criteria_pie.png': {
                "title": "Torta dei Criteri di Aggiudicazione",
                "description": "Questa visualizzazione mostra la proporzione tra i diversi criteri di aggiudicazione. Il 'prezzo più basso' è dominante, ma una fetta significativa di contratti utilizza criteri 'multifattoriali', suggerendo che la qualità e altri fattori sono importanti in un numero non trascurabile di appalti."
            },
            '11_temporal_trends.png': {
                "title": "Andamento Temporale dei Contratti",
                "description": "Questo grafico a doppia scala mostra l'evoluzione del numero di contratti e del prezzo medio nel corso degli anni. Si possono notare picchi in determinati periodi, che potrebbero essere correlati a cicli economici, elezioni o specifici programmi di investimento governativo. Il prezzo medio, d'altra parte, mostra una sua dinamica, non sempre correlata al volume."
            },
            '12_correlation_heatmap.png': {
                "title": "Heatmap di Correlazione",
                "description": "La heatmap visualizza le correlazioni lineari tra le principali variabili numeriche. Si nota una debole correlazione positiva tra prezzo e scadenza, il che è intuitivo. L'assenza di correlazioni forti suggerisce che le relazioni tra le variabili sono complesse e non lineari, richiedendo analisi più approfondite."
            },
            '14_financial_stacked_bar_value_by_year.png': {
                "title": "Valore Contratti per Anno e Criterio (Analisi Finanziaria)",
                "description": "Questo grafico è pensato per un analista finanziario. Mostra come il valore totale dei contratti, suddiviso per criterio di aggiudicazione, è cambiato nel tempo. Si può osservare se c'è stato uno spostamento strategico dal 'prezzo più basso' a criteri 'multifattoriali', indicando un cambiamento nelle priorità di appalto verso la qualità."
            },
            '16_manager_avg_deadline_by_district.png': {
                "title": "Scadenza Media per Distretto (Analisi Gestionale)",
                "description": "Questo grafico è cruciale per un project manager. Evidenzia come la durata media dei progetti vari in modo significativo da un distretto all'altro. Questo può influenzare la pianificazione, l'allocazione delle risorse e la gestione del rischio, poiché i tempi di esecuzione sembrano dipendere da fattori logistici o burocratici locali."
            },
            '20_price_distribution_by_award_criteria.png': {
                "title": "Distribuzione Prezzi per Criterio di Aggiudicazione",
                "description": "Il box plot confronta la distribuzione dei prezzi per i contratti aggiudicati con 'prezzo più basso' rispetto a quelli 'multifattoriali'. Sebbene la mediana dei prezzi per i contratti 'multifattoriali' sia leggermente più alta, la grande dispersione (i 'baffi' del box plot) in entrambi i gruppi indica che il tipo di progetto ha probabilmente un impatto maggiore sul costo rispetto al solo criterio di aggiudicazione."
            },
            '21_award_criteria_by_district.png': {
                "title": "Conteggio Criteri per Distretto",
                "description": "Questo grafico mostra la preferenza per un criterio di aggiudicazione a livello geografico. La maggior parte dei distretti si affida prevalentemente al 'prezzo più basso'. Tuttavia, in centri urbani come Lisbona, la proporzione di contratti 'multifattoriali' è visibilmente più alta, suggerendo una maggiore enfasi sulla qualità e altri fattori oltre al semplice costo."
            },
            '22_price_vs_deadline_density_by_award_criteria.png': {
                "title": "Densità Prezzo vs. Scadenza per Criterio",
                "description": "Questo grafico di densità mostra dove si concentrano i contratti nello spazio prezzo-scadenza. Le aree più luminose indicano una maggiore densità. Si può vedere come i contratti basati sul 'prezzo più basso' (in blu) tendano a concentrarsi su valori più bassi, mentre i contratti 'multifattoriali' (in rosso) sono più sparsi, coprendo anche nicchie di progetti ad alto valore e lunga durata."
            }
        }

        plot_files = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith('.png') and f in plot_info])

        if not plot_files:
            print("Nessun file PNG corrispondente trovato in PLOTS_DIR per generare il report.")
            return

        # Crea documento PDF con ReportLab
        doc = SimpleDocTemplate(output_path, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        story = []

        # Pagina di copertina
        story.append(Paragraph(report_title, styles['h1']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph('Un\'analisi narrativa basata sui dati degli appalti pubblici portoghesi', styles['h3']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f'Data del report: {pd.Timestamp.now().strftime("%d-%m-%Y")}', styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f'Numero totale di contratti analizzati nel dataset finale: {len(self.df):,}', styles['Normal']))
        story.append(PageBreak())

        # Aggiungi pagina per ogni grafico
        for fname in plot_files:
            if fname in plot_info:
                info = plot_info[fname]
                img_path = os.path.join(PLOTS_DIR, fname)

                # Aggiungi titolo e descrizione
                story.append(Paragraph(info['title'], styles['h2']))
                story.append(Spacer(1, 0.1*inch))
                desc_paragraph = Paragraph(textwrap.fill(info['description'], width=100), styles['Normal'])
                story.append(desc_paragraph)
                story.append(Spacer(1, 0.2*inch))

                # Aggiungi immagine
                try:
                    with PILImage.open(img_path) as img_pil:
                        width_px, height_px = img_pil.size
                    max_width = 9*inch
                    max_height = 5.5*inch
                    ratio = min(max_width / width_px, max_height / height_px)
                    img_width = width_px * ratio
                    img_height = height_px * ratio

                    img_reportlab = Image(img_path, width=img_width, height=img_height)
                    story.append(img_reportlab)
                    story.append(PageBreak())
                except FileNotFoundError:
                    print(f"Attenzione: Immagine {fname} non trovata. Sarà saltata nel report.")
                    story.append(Paragraph(f"(Immagine {fname} non trovata)", styles['Italic']))
                    story.append(PageBreak())
                except Exception as e:
                    print(f"Errore durante l'aggiunta dell'immagine {fname} al PDF: {e}")
                    story.append(Paragraph(f"(Errore nel caricamento immagine {fname})", styles['Italic']))
                    story.append(PageBreak())

        # Costruisci il PDF
        try:
            doc.build(story)
            print(f"Report PDF narrativo generato con successo: {output_path}")
        except Exception as e:
            print(f"Errore durante la costruzione del PDF: {e}")
        
# %% [markdown]
# ## Esecuzione della Pipeline di Storytelling
# Si istanzia la classe e si invocano i metodi in sequenza.

# %% [markdown]
# ### Fase 1.1: Caricamento e Ispezione Iniziale
# **Obiettivo**: Caricare il dataset e ottenere una prima comprensione della sua struttura, dei tipi di dati e della presenza di valori nulli.
# 
# **Cosa facciamo**:
# - `_load_data`: Carichiamo il file Excel in un DataFrame pandas.
# - `inspect_dataframe`:
    #   - `df.info()`: Mostra i tipi di dati per colonna e l'uso della memoria. Utile per identificare subito colonne con tipi errati (es. numeri letti come testo).
    #   - `df.isnull().sum()`: Conta i valori mancanti per colonna. Fondamentale per pianificare la strategia di pulizia.
    #   - `df.head()`: Visualizza le prime righe per avere un'idea concreta del contenuto.

# %% [code]
preprocessor = DataPreprocessor('Datasets/PPPData_EN_1.0.xlsx')
preprocessor.inspect_dataframe("Stato Iniziale del DataFrame")

# %% [markdown]
# ### Fase 1.2: Analisi e Pulizia dei Dati
# **Obiettivo**: Rimuovere le colonne inutili, correggere i dati e gestire le righe con informazioni critiche mancanti.
# 
# **Cosa facciamo**:
# - `analyze_missing_values`: Visualizziamo la percentuale di valori mancanti. Questo ci guida nella decisione di quali colonne eliminare.
# - `prune_columns`: Rimuoviamo le colonne con troppi valori mancanti o che sono irrilevanti per l'analisi (es. ID, colonne con un solo valore). **Decisione chiave**: colonne come `Conclusion of a framework agreement` o `Electronic auction` sono quasi vuote e non possono fornire insight affidabili.
# - `clean_and_correct`:
    #   - Correggiamo i tipi di dati (es. da booleano a intero).
    #   - Rimuoviamo spazi bianchi superflui (`strip()`).
    #   - Eliminiamo righe con dati palesemente errati (es. codici distretto incoerenti).
    #   - `dropna(subset=key_cols)`: Rimuoviamo le righe dove mancano informazioni fondamentali come il prezzo o la municipalità, poiché sarebbero inutilizzabili.

# %% [code]
preprocessor.analyze_missing_values()

# %% [code]
columns_to_drop = [
    'Count', 'ID', 'Short Description1', 'Country', 'Award criteria',
    'Involves joint procurement (with several entities) (T/F)',
    'Awarded by a central purchasing body (T/F)',
    'Conclusion of a framework agreement (T/F)', 'Electronic auction (T/F)',
    'Negotiation phase (T/F)', 'Contracting by lots (T/F)', 'Collateral',
    'Contract end type', 'Justification for price change', 'Justification for deadline change'
]
preprocessor.prune_columns(columns_to_drop)
preprocessor.clean_and_correct()
preprocessor.inspect_dataframe("Stato del DataFrame dopo Pulizia Iniziale")

# %% [markdown]
# ### Fase 1.3: Feature Engineering
# **Obiettivo**: Creare nuove feature per arricchire il dataset e migliorare il potenziale predittivo o descrittivo del modello.
# 
# **Cosa facciamo**:
# - `engineer_date_features`: Estraiamo l'anno e il mese dalle colonne di data. Questo ci permette di analizzare andamenti temporali (stagionalità, trend annuali).
# - `engineer_text_features`:
    #   - Applichiamo la nostra `preprocess_text` alla colonna `Cpvs Designation`.
    #   - Usiamo `TfidfVectorizer` per trasformare il testo pulito in feature numeriche. TF-IDF assegna un peso a ogni parola in base alla sua frequenza nel documento e alla sua rarità nell'intero corpus, evidenziando i termini più caratterizzanti.
    #   - Creiamo colonne binarie (0/1) per le 10 parole chiave più importanti, rendendo il loro impatto facilmente interpretabile.

# %% [code]
preprocessor.engineer_date_features()
cpvs_vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.8, ngram_range=(2, 3), max_features=10)
preprocessor.engineer_text_features('Cpvs Designation', 'cpvs', cpvs_vectorizer)
preprocessor.inspect_dataframe("Stato del DataFrame dopo Feature Engineering Testuale")

# %% [markdown]
# ### Fase 1.4: Gestione Outlier e Feature Finanziarie
# **Obiettivo**: Identificare e gestire valori anomali (outlier) e creare nuove feature finanziarie.
# 
# **Cosa facciamo**:
# - `process_numerical_features`:
    #   - Visualizziamo la distribuzione di feature numeriche come `Execution deadline`. **Finding**: Le distribuzioni mostrano una forte asimmetria (coda lunga a destra), indicando la presenza di contratti con durate o valori eccezionalmente alti.
    #   - **Gestione Outlier**: Rimuoviamo gli outlier più estremi usando il metodo dell'IQR (Interquartile Range). Questo previene che pochi valori anomali distorcano le medie e le analisi successive.
    #   - **Discretizzazione**: Convertiamo le variabili numeriche continue in categorie ordinate (es. 'Short', 'Medium', 'Long') usando `pd.qcut`. Questo semplifica l'analisi e la visualizzazione delle relazioni.
# - `engineer_financial_features`:
    #   - Creiamo `Price per Day`, una metrica di intensità che normalizza il costo di un progetto sulla sua durata. Questo permette di confrontare progetti di dimensioni diverse in modo più equo.

# %% [code]
preprocessor.process_numerical_features()

# %% [code]
preprocessor.engineer_financial_features()
preprocessor.inspect_dataframe("Stato del DataFrame dopo Gestione Outlier e Feature Finanziarie")

# %% [markdown]
# ### Fase 1.5: Imputazione Finale e Salvataggio
# **Obiettivo**: Gestire gli ultimi valori mancanti e salvare il dataset pulito per un uso futuro.
# 
# **Cosa facciamo**:
# - `impute_and_finalize`:
    #   - Per le colonne numeriche rimanenti, riempiamo i valori nulli con la **mediana** invece della media, poiché la mediana è meno sensibile agli outlier.
    #   - Per le colonne categoriche, usiamo la **moda** (il valore più frequente).
    #   - `dropna()`: Eseguiamo una pulizia finale per rimuovere qualsiasi riga che possa ancora contenere valori nulli.
# - `save_data`: Salviamo il DataFrame pulito in un file CSV. Questo ci permette di ricaricare direttamente i dati puliti in futuro, saltando tutti i passaggi di preprocessing.

# %% [code]
preprocessor.impute_and_finalize()
preprocessor.save_data(CLEANED_DATA_PATH)
preprocessor.inspect_dataframe("Stato Finale del DataFrame")

# %% [markdown]
# ### Fase 1.6: Riepilogo Statistico
# **Obiettivo**: Eseguire un'analisi descrittiva del dataset finale per avere un riepilogo quantitativo completo.
# 
# **Cosa facciamo**:
# - `summarize_data`: Usiamo `df.describe(include='all')` per ottenere statistiche di base (media, mediana, deviazione standard, conteggi, valori unici) per tutte le colonne, sia numeriche che categoriche. Questo serve come un controllo di qualità finale e fornisce una panoramica immediata del dataset con cui lavoreremo.

# %% [code]
preprocessor.summarize_data()

# %% [markdown]
# # Fase 2: Analisi Esplorativa e Storytelling Visivo
# Con un dataset pulito, si passa all'esplorazione visiva per scoprire pattern e insight.

# %% [markdown]
# ### Fase 2.1: Analisi Semantica e Clustering
# **Obiettivo**: Raggruppare i contratti in cluster tematici basati sul significato delle loro descrizioni.
# 
# **Cosa facciamo**:
# - `compute_semantic_embeddings`: Trasformiamo le descrizioni testuali pulite (`cpvs_clean_text`) in vettori numerici (embeddings) usando un modello pre-addestrato (`all-MiniLM-L6-v2`). Questi vettori catturano il significato semantico del testo. Usiamo PCA per ridurre la dimensionalità a 2D per la visualizzazione.
# - `perform_text_clustering`: Applichiamo l'algoritmo K-Means sugli embedding 2D per raggruppare i contratti in `n_clusters` (in questo caso 5) gruppi distinti.
# - `visualize_text_clusters`: Visualizziamo i cluster su uno scatter plot.
# 
# **Finding**: Il clustering sugli embedding semantici rivela gruppi tematici distinti. Ad esempio, si possono identificare cluster relativi a "lavori stradali", "costruzione di edifici" o "servizi di manutenzione". I colori distinti aiutano a separare visivamente questi gruppi, confermando che il modello ha catturato differenze semantiche reali.

# %% [code]
preprocessor.compute_semantic_embeddings('cpvs_clean_text', prefix='cpvs_sem')
preprocessor.perform_text_clustering(n_clusters=5)
preprocessor.visualize_text_clusters()

# %% [markdown]
# ### Fase 2.2: Visualizzazioni di Base
# **Obiettivo**: Ottenere una panoramica generale della distribuzione dei dati attraverso grafici semplici.
# 
# **Finding**: I grafici confermano la concentrazione geografica dei contratti nelle aree metropolitane di Lisbona e Porto. Emerge che la maggior parte dei contratti viene aggiudicata tramite il criterio del "prezzo più basso". Tuttavia, la distribuzione dei prezzi e delle scadenze è molto ampia, suggerendo una grande eterogeneità nei tipi e nelle dimensioni dei progetti.

# %% [code]
preprocessor.generate_final_visualizations()

# %% [markdown]
# ### Fase 2.3: Visualizzazioni Avanzate
# **Obiettivo**: Esplorare le relazioni tra più variabili e analizzare gli andamenti nel tempo.
# 
# **Finding**:
# - **Correlazione**: La heatmap di correlazione mostra una debole correlazione positiva tra prezzo e scadenza, il che è intuitivo (progetti più costosi tendono a durare di più). L'assenza di correlazioni forti suggerisce che le relazioni sono più complesse.
# - **Andamento Temporale**: Il grafico temporale rivela un picco nel numero di contratti in certi anni. Questo potrebbe essere correlato a cicli economici, elezioni o lanci di specifici programmi di investimento governativi.

# %% [code]
preprocessor.generate_advanced_visualizations()

# %% [markdown]
# ### Fase 2.4: Analisi Geospaziale
# **Obiettivo**: Visualizzare la distribuzione geografica dei contratti e dei loro valori.
# 
# **Finding**: Le mappe coropletiche offrono uno degli insight più interessanti. Sebbene Lisbona e Porto abbiano il maggior *numero* di contratti, il *valore medio* più alto si trova spesso in distretti con meno contratti, come Bragança o Viana do Castelo. Questo pattern suggerisce la presenza di pochi ma grandi progetti infrastrutturali (es. dighe, autostrade) in quelle aree, a differenza della miriade di progetti più piccoli nelle grandi città.

# %% [code]
preprocessor.generate_geospatial_visualizations()

# %% [code]
preprocessor.generate_additional_geospatial_plot()

# %% [markdown]
# ### Fase 2.5: Visualizzazioni per Ruolo
# **Obiettivo**: Creare grafici mirati a rispondere a domande specifiche di diversi stakeholder.
# 
# **Finding per Financial Analyst**: Il grafico a barre impilate mostra come il valore totale dei contratti basati sul "prezzo più basso" sia diminuito negli ultimi anni a favore di criteri multifattoriali. Questo indica un cambiamento strategico nelle politiche di appalto, forse verso una maggiore enfasi sulla qualità oltre che sul costo.
# 
# **Finding per Project Manager**: La scadenza media di esecuzione varia significativamente tra i distretti. Questo è un dato cruciale per la pianificazione strategica e la gestione del rischio, poiché suggerisce che i tempi di progetto possono dipendere fortemente da fattori logistici e burocratici locali.

# %% [code]
preprocessor.generate_role_based_visualizations()

# %% [markdown]
# ### Fase 2.6: Analisi di Intensità e Testo
# **Obiettivo**: Analizzare la relazione tra costo e durata e visualizzare i temi principali del testo.
# 
# **Finding**:
# - **Intensità del Prezzo**: Il grafico di intensità mostra che non c'è una relazione lineare semplice tra il prezzo totale e il "prezzo al giorno". Questo suggerisce che la complessità, i materiali e la manodopera influenzano il costo in modi non banali, e progetti più lunghi non sono necessariamente più costosi su base giornaliera.
# - **Word Cloud**: La nuvola di parole evidenzia "lavori", "costruzione" e "manutenzione" come i termini più frequenti, confermando che il dataset è focalizzato correttamente sul settore delle costruzioni.

# %% [code]
preprocessor.generate_price_intensity_plot()

# %% [code]
preprocessor.generate_word_cloud()

# %% [markdown]
# ### Fase 2.7: Dashboard e Visualizzazioni d'Impatto
# **Obiettivo**: Creare visualizzazioni professionali e moderne che catturino immediatamente l'attenzione e comunichino insights chiave.
# 
# **Finding**: Le visualizzazioni tipo dashboard offrono una panoramica immediata e d'impatto dei KPI principali. La heatmap temporale rivela pattern stagionali nei contratti, mentre lo stacked area chart mostra chiaramente l'evoluzione del valore nel tempo. La treemap permette di identificare rapidamente dove si concentra il budget, e il dashboard KPI fornisce una sintesi visiva perfetta per presentazioni executive.

# %% [code]
preprocessor.generate_temporal_heatmap()

# %% [code]
preprocessor.generate_stacked_area_chart()

# %% [code]
preprocessor.generate_treemap_budget_distribution()

# %% [code]
preprocessor.generate_kpi_dashboard()

# %% [markdown]
# ### Fase 2.8: Creazione del Report PDF
# **Obiettivo**: Consolidare tutte le visualizzazioni statiche in un unico documento portabile.
# 
# **Cosa facciamo**: Si genera un report PDF che raccoglie tutte le visualizzazioni statiche create, fornendo un riassunto completo dell'analisi che può essere facilmente condiviso con stakeholder che non hanno accesso all'ambiente di programmazione.

# %% [code]
preprocessor.generate_pdf_report()

# %% [markdown]
# ### Fase 2.9: Analisi Approfondita dei Criteri di Aggiudicazione
# **Obiettivo**: Indagare come i criteri di aggiudicazione ("prezzo più basso" vs. "multifattoriale") influenzano i costi e come si distribuiscono geograficamente.
# 
# **Finding**:
# - **Distribuzione Prezzi**: Il box plot mostra che, sebbene i contratti "multifattoriali" abbiano una mediana dei prezzi leggermente più alta, la variabilità è enorme in entrambi i casi. Questo suggerisce che il tipo di progetto ha un impatto maggiore sul prezzo rispetto al solo criterio di aggiudicazione.
# - **Distribuzione Geografica**: Il grafico a barre rivela che la maggior parte dei distretti si affida prevalentemente al criterio del "prezzo più basso". Tuttavia, in centri urbani come Lisbona, la proporzione di contratti "multifattoriali" è visibilmente più alta, indicando una maggiore attenzione alla qualità e altri fattori oltre al costo.
# - **Prezzo vs. Scadenza**: Lo scatter plot interattivo conferma che non esiste una regola semplice. Ci sono progetti "multifattoriali" economici e veloci, e progetti basati sul "prezzo più basso" che sono costosi e lunghi. Questo rafforza l'idea che le dinamiche di costo sono complesse e dipendono da molteplici fattori.

# %% [code]
preprocessor.analyze_award_criteria_and_price()