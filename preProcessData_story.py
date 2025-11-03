# %% [markdown]
# %% [markdown]
# ## 1. Setup dell'Ambiente
# **Obiettivo**: Caricare tutte le librerie necessarie e configurare le variabili globali.
#
# **Operazioni**:
# - Si importano librerie fondamentali come `pandas` per la manipolazione dei dati, `matplotlib` e `seaborn` per le visualizzazioni, e `spacy` per il processamento del linguaggio naturale.
# - Si definiscono costanti come le directory per i grafici (`PLOTS_DIR`) e il percorso del file di output (`CLEANED_DATA_PATH`) per mantenere il codice pulito e facilmente configurabile.
# - Si inizializza il modello di linguaggio `en_core_web_sm` di spaCy, che servirà per la lemmatizzazione del testo.
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
from matplotlib.ticker import FuncFormatter
from wordcloud import WordCloud
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import landscape, A4
import textwrap
from PIL import Image as PILImage
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


# --- Costanti ---
PLOTS_DIR = 'plots'
CLEANED_DATA_PATH = 'Datasets/PPPData_EN_cleaned.csv'
GEOJSON_PATH = 'Datasets/portugal_districts.geojson'

# --- Setup Iniziale ---
if not os.path.exists(PLOTS_DIR): os.makedirs(PLOTS_DIR)

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print('Download modello di linguaggio spaCy...')
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# --- Stile Globale Seaborn ---
sns.set_style("ticks")
sns.set_palette("tab10")
sns.set_context("talk")

# %% [markdown]
# ## 2. Preprocessing Testuale
# **Obiettivo**: Creare una funzione standard per pulire e normalizzare qualsiasi campo di testo.
#
# **Operazioni** (`preprocess_text`):
# 1.  **Lowercase & Rimozione Punteggiatura/Numeri**: Il testo viene convertito in minuscolo e si rimuovono punteggiatura e numeri per trattare parole come "Costruzione" e "costruzione." allo stesso modo.
# 2.  **Lemmatizzazione**: Utilizzando spaCy, le parole vengono ridotte alla loro forma base (es. "running" -> "run", "studies" -> "study"). Questo aggrega parole con lo stesso significato.
# 3.  **Rimozione Stop Words**: Si eliminano parole comuni ma poco informative (es. "the", "a", "is") che non aggiungono significato semantico.
#
# **Risultato**: Una stringa di testo pulita, composta solo da parole chiave significative, pronta per essere analizzata.

# %% [code]
def preprocess_text(text: str) -> str:
    """Pulisce e normalizza una stringa di testo."""
    if not isinstance(text, str): return ""
    # Rimuove punteggiatura e numeri, converte in minuscolo
    text = re.sub(f'[{re.escape(string.punctuation)}0-9]', '', text.lower())
    doc = nlp(text)
    # Lemmatizzazione e rimozione stop words/punteggiatura residua
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.strip()]
    return " ".join(lemmas)

# %% [markdown]
# ## 3. La Classe `DataHandler`
# **Obiettivo**: Incapsulare tutta la logica di preprocessing e visualizzazione in una classe per un approccio modulare, riutilizzabile e organizzato, separando la manipolazione dei dati dalla loro visualizzazione.
#
# **Struttura**:
# - `DataPreprocessor`: Sottoclasse per tutte le operazioni di caricamento, pulizia e feature engineering.
# - `DataVisualizer`: Sottoclasse per tutte le operazioni di plotting e generazione di report.
# - `DataHandler`: Classe principale che orchestra le due sottoclassi.

# %% [code]
class DataPreprocessor:
    """Gestisce il caricamento, la pulizia e il feature engineering."""

    def __init__(self, file_path: str):
        self.df = self._load_data(file_path)

    def _load_data(self, path: str) -> pd.DataFrame:
        """Carica dati da file Excel o CSV con gestione errori robusta."""
        print(f"Caricamento dati da: {path}")
        if path.lower().endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
            return pd.read_excel(path)
        elif path.lower().endswith('.csv'):
            try:
                return pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding='latin1')
        else:
            raise ValueError("Formato file non supportato. Usa .xlsx, .xls o .csv")

    def inspect_dataframe(self, title: str):
        """Ispeziona il DataFrame mostrando info, valori nulli e prime righe."""
        print(f"\n--- {title} ---")
        print("\n1. Informazioni Generali e Memoria:")
        self.df.info(memory_usage='deep')
        null_counts = self.df.isnull().sum()
        null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
        print("\n2. Valori Nulli per Colonna (Top 5):")
        if not null_counts.empty:
            print(null_counts.head())
        else:
            print("Nessun valore nullo trovato.")
        print("\n3. Prime 5 Righe:")
        with pd.option_context('display.max_columns', None):
            print(self.df.head())

    def analyze_missing_values(self, visualizer):
        """Visualizza la percentuale di valori mancanti."""
        missing_percentage = self.df.isnull().sum() * 100 / len(self.df)
        missing_df = pd.DataFrame({'column_name': self.df.columns,
                                   'percent_missing': missing_percentage})
        missing_df = missing_df[missing_df['percent_missing'] > 0].sort_values('percent_missing', ascending=False)

        if missing_df.empty:
            print("Nessuna colonna con valori mancanti da visualizzare.")
            return

        fig, ax = plt.subplots(figsize=(12, max(8, len(missing_df) * 0.5)))
        sns.barplot(x='percent_missing', y='column_name', data=missing_df,
                    ax=ax, palette='viridis', edgecolor='black', linewidth=0.8)
        ax.set_title('Percentuale di Valori Mancanti per Colonna (>0%)', fontsize=16, pad=20)
        ax.set_xlabel('Percentuale Mancante (%)', fontsize=12)
        ax.set_ylabel('Nome Colonna', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        sns.despine()
        fig.tight_layout()
        visualizer._save_plot(fig, '01_missing_values_percentage.png')
        plt.show()
        

    def prune_columns(self, columns_to_drop: List[str]):
        """Rimuove le colonne specificate."""
        print(f"\Rimozione di {len(columns_to_drop)} colonne specificate...")
        actual_cols_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        self.df.drop(columns=actual_cols_to_drop, inplace=True, errors='ignore')
        print(f"Colonne effettivamente rimosse: {len(actual_cols_to_drop)}")

    def clean_and_correct(self):
        """Pulisce e corregge dati con gestione errori migliorata."""
        print("\n--- Pulizia e Correzione Dati ---")
        if 'Environmental criteria (T/F)' in self.df.columns:
            self.df['Environmental criteria (T/F)'] = pd.to_numeric(self.df['Environmental criteria (T/F)'], errors='coerce').fillna(0).astype(int)

        if 'Published in the EU journal' in self.df.columns:
            mapping = {False: 0, 'False': 0, 0: 0, '0': 0,
                       True: 1, 'True': 1, 'TRUE ': 1, 1: 1, '1': 1}
            self.df['Published in the EU journal'] = self.df['Published in the EU journal'].map(mapping)

        if 'District' in self.df.columns:
            self.df['District'] = self.df['District'].astype(str).str.strip()

        if 'District Code' in self.df.columns and 'District' in self.df.columns:
            initial_rows = len(self.df)
            self.df = self.df[~((self.df['District'] == 'Beja') & (self.df['District Code'] == 13))]
            self.df = self.df[~((self.df['District'] == 'Faro') & (self.df['District Code'] == 13))]
            print(f"Rimosse {initial_rows - len(self.df)} righe con codici distretto incoerenti.")
            self.df.drop(columns=['District Code'], inplace=True, errors='ignore')

        key_cols = ['Publication Year', 'Municipality', 'Base Bid Price (€)']
        actual_key_cols = [col for col in key_cols if col in self.df.columns]
        if actual_key_cols:
            initial_rows = len(self.df)
            self.df.dropna(subset=actual_key_cols, inplace=True)
            print(f"Rimosse {initial_rows - len(self.df)} righe con valori nulli in colonne chiave.")

    def engineer_date_features(self):
        """Crea feature basate sulle date, inclusa la differenza tra date."""
        print("\nCreazione di feature basate sulle date...")
        date_cols_to_process = ['Signing date', 'Closing date', 'Publication date']
        processed_datetimes = {}

        for date_col in date_cols_to_process:
            if date_col in self.df.columns:
                try:
                    # Converte in datetime
                    dt_series = pd.to_datetime(self.df[date_col], errors='coerce', dayfirst=True, infer_datetime_format=True)
                    self.df[date_col] = dt_series # Salva la colonna convertita
                    processed_datetimes[date_col] = dt_series # Salva per calcoli successivi
                    
                    # Estrae Anno e Mese
                    year_col_name = f"{date_col.split()[0]} Year"
                    month_col_name = f"{date_col.split()[0]} Month"
                    self.df[year_col_name] = dt_series.dt.year
                    self.df[month_col_name] = dt_series.dt.month
                    print(f"Elaborata colonna data: {date_col} -> {year_col_name}, {month_col_name}")
                except Exception as e:
                    print(f"Errore durante l'elaborazione della colonna data '{date_col}': {e}")

        # --- Feature Engineering Aggiuntivo: Calcolo Differenza Date ---
        if 'Closing date' in processed_datetimes and 'Signing date' in processed_datetimes:
            diff_col = 'Diference between close and signing dates'
            if diff_col not in self.df.columns: # Calcola solo se non esiste già
                print(f"Calcolo '{diff_col}'...")
                self.df[diff_col] = (self.df['Closing date'] - self.df['Signing date']).dt.days
                print(f"Creata colonna '{diff_col}'.")
        
        # Rimuove le colonne datetime originali dopo averle usate
        cols_to_drop_dates = list(processed_datetimes.keys())
        if cols_to_drop_dates:
            self.df.drop(columns=cols_to_drop_dates, inplace=True, errors='ignore')
            print(f"Colonne data originali rimosse: {', '.join(cols_to_drop_dates)}")

    def engineer_text_features(self, text_col: str, new_col_prefix: str, vectorizer: TfidfVectorizer):
        """Crea feature testuali con TF-IDF."""
        print(f"\nCreazione di feature testuali da '{text_col}'...")
        if text_col not in self.df.columns:
            print(f"Colonna '{text_col}' non trovata.")
            return

        cleaned_col = f"{text_col}_cleaned_internal"
        raw_store_col = f"{new_col_prefix}_raw_text"
        clean_store_col = f"{new_col_prefix}_clean_text"

        self.df[cleaned_col] = self.df[text_col].fillna('').astype(str).apply(preprocess_text)
        self.df[raw_store_col] = self.df[text_col].astype(str)
        self.df[clean_store_col] = self.df[cleaned_col]

        try:
            X_tfidf = vectorizer.fit_transform(self.df[cleaned_col])
            keywords = vectorizer.get_feature_names_out()
            print(f"Prime 10 keywords TF-IDF: {', '.join(keywords[:10])}")

            for keyword in keywords:
                safe_keyword = re.sub(r'\W+', '_', keyword).strip('_')
                col_name = f"{new_col_prefix}_keyword_{safe_keyword}"
                self.df[col_name] = self.df[cleaned_col].apply(lambda x: 1 if keyword in x.split() else 0)
            print(f"Create {len(keywords)} colonne keyword binarie.")
        except Exception as e:
            print(f"Errore durante la vettorizzazione TF-IDF: {e}")
            if cleaned_col in self.df.columns:
                self.df.drop(columns=[cleaned_col], inplace=True)
            return

        self.df.drop(columns=[text_col, cleaned_col], inplace=True, errors='ignore')

    def process_numerical_features(self, visualizer):
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

            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Visualizza distribuzione PRIMA
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(f"Distribuzione di '{col}' (Prima Rimozione Outlier)", fontsize=16)
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[0], color='skyblue', edgecolor='black', alpha=0.7, bins=30)
            sns.boxplot(x=self.df[col].dropna(), ax=axes[1], color='lightcoral', linewidth=1.5)
            axes[0].set_title('Istogramma e KDE', fontsize=14)
            axes[1].set_title('Box Plot', fontsize=14)
            sns.despine(ax=axes[0])
            sns.despine(ax=axes[1], left=True)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            visualizer._save_plot(fig, f'02a_distribution_before_outlier_{col.replace(" ", "_").lower()}.png')
            plt.show()
            

            # Rimozione Outlier
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 2.5 * IQR, Q3 + 2.5 * IQR
            initial_rows = len(self.df)
            self.df = self.df[(self.df[col].isnull()) | ((self.df[col] >= lower_bound) & (self.df[col] <= upper_bound))]
            print(f"Rimosse {initial_rows - len(self.df)} righe outlier per '{col}'.")

            numeric_col_name = f'{col}_numeric'
            self.df[numeric_col_name] = self.df[col]

            # Discretizzazione
            if self.df[col].notna().sum() > len(labels):
                try:
                    self.df[col] = pd.qcut(self.df[col], len(labels), labels=labels, duplicates='drop')
                except ValueError:
                    self.df[col] = self.df[numeric_col_name] # Fallback a numerico
            else:
                self.df[col] = self.df[numeric_col_name] # Fallback a numerico

        # Gestione 'Base Bid Price (€)'
        price_col = 'Base Bid Price (€)'
        if price_col in self.df.columns:
            self.df[price_col] = pd.to_numeric(self.df[price_col], errors='coerce')
            if self.df[price_col].notna().sum() > 3:
                try:
                    self.df[f'{price_col}_category'] = pd.qcut(self.df[price_col], 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                except ValueError:
                    median_price = self.df[price_col].median()
                    self.df[f'{price_col}_category'] = pd.cut(self.df[price_col], bins=[-np.inf, median_price, np.inf], labels=['Low', 'High'])

        # Gestione 'Difference between the effective and initial price (€)'
        diff_price_col = 'Difference between the effective and initial price (€)'
        if diff_price_col in self.df.columns:
            diff_numeric = pd.to_numeric(self.df[diff_price_col], errors='coerce')
            self.df[f'{diff_price_col}_numeric'] = diff_numeric
            if diff_numeric.notna().sum() > 3:
                try:
                    self.df['Difference between the effective and initial price class'] = pd.qcut(diff_numeric, 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                except ValueError:
                    self.df['Difference between the effective and initial price class'] = pd.cut(diff_numeric, bins=[-np.inf, -0.01, 0.01, np.inf], labels=['Decrease', 'Stable', 'Increase'])

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

        print(f"Calcolato '{price_per_day_col}' per {valid_mask.sum()} righe valide.")

        if self.df[price_per_day_col].isnull().any():
            median_val = self.df[price_per_day_col].median()
            fill_val = median_val if pd.notna(median_val) else 0
            self.df[price_per_day_col].fillna(fill_val, inplace=True)
            print(f"Imputati valori mancanti di '{price_per_day_col}' con: {fill_val:.2f}")

    def impute_and_finalize(self):
        """Imputa valori mancanti finali e rimuove righe con NaN rimanenti."""
        print("\n--- Imputazione Finale e Finalizzazione ---")
        cols_to_impute_median = ['Submission deadline (days)', 'Classification of the multifactor criteria (%)']
        
        for col in cols_to_impute_median:
            if col in self.df.columns and self.df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    median_val = self.df[col].median()
                    fill_val = median_val if pd.notna(median_val) else 0
                    self.df[col].fillna(fill_val, inplace=True)
                    print(f"Imputati valori nulli in '{col}' con mediana ({fill_val:.2f}).")

        col_impute_mode = 'Published in the EU journal'
        if col_impute_mode in self.df.columns and self.df[col_impute_mode].isnull().any():
            if not self.df[col_impute_mode].mode().empty:
                mode_val = self.df[col_impute_mode].mode()[0]
                self.df[col_impute_mode].fillna(mode_val, inplace=True)
                print(f"Imputati valori nulli in '{col_impute_mode}' con moda ({mode_val}).")
            else:
                self.df[col_impute_mode].fillna(0, inplace=True)

        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        print(f"Rimosse {initial_rows - len(self.df)} righe con valori NaN rimanenti.")

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
        print(f"Righe: {len(self.df)}, Colonne: {len(self.df.columns)}")
        print("\nStatistiche Descrittive:")
        with pd.option_context('display.max_columns', None, 'display.width', 1000):
            try:
                print(self.df.describe(include='all').transpose())
            except Exception as e:
                print(f"Errore durante la generazione delle statistiche: {e}")

    def compute_semantic_embeddings(self, text_col: str, prefix: str = 'semantic'):
        """Calcola embeddings semantici e riduce a 2D con PCA."""
        print(f"\n--- Calcolo Embedding Semantici per '{text_col}' ---")
        if SentenceTransformer is None or PCA is None:
            print("Librerie 'sentence-transformers' o 'sklearn' non trovate. Salto.")
            return False
        if text_col not in self.df.columns:
            print(f"Errore: Colonna '{text_col}' non trovata.")
            return False

        sentences = self.df[text_col].fillna('').astype(str).tolist()
        if not sentences:
            print("Nessun testo trovato.")
            return False

        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        print(f"Caricamento modello: {model_name}...")
        try:
            model = SentenceTransformer(model_name)
            print(f"Calcolo embeddings per {len(sentences)} testi...")
            embeddings = model.encode(sentences, show_progress_bar=True, batch_size=64)
            
            print(f"Embeddings calcolati. Dimensione: {embeddings.shape}")
            print("Riduzione dimensionale con PCA a 2 componenti...")
            reducer = PCA(n_components=2, random_state=42)
            components = reducer.fit_transform(embeddings)
            
            self.df[f'{prefix}_x'] = components[:, 0]
            self.df[f'{prefix}_y'] = components[:, 1]
            print(f"PCA completata. Varianza spiegata: {reducer.explained_variance_ratio_.sum():.2%}")
            return True
        except Exception as e:
            print(f"Errore durante il calcolo degli embeddings o PCA: {e}")
            return False

    def perform_text_clustering(self, n_clusters: int = 5, random_state: int = 42, prefix: str = 'cpvs_sem'):
        """Esegue il clustering K-Means sugli embeddings semantici."""
        print(f"\n--- Esecuzione Clustering K-Means ({n_clusters} cluster) ---")
        if KMeans is None:
            print("Libreria 'sklearn' non trovata. Salto clustering.")
            return

        x_col, y_col = f'{prefix}_x', f'{prefix}_y'
        cluster_col = f"{prefix.split('_')[0]}_cluster"
        price_col = 'Base Bid Price (€)' # Per analisi finanziaria

        if x_col not in self.df.columns or y_col not in self.df.columns:
            print(f"Errore: Colonne embedding '{x_col}' o '{y_col}' non trovate.")
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
            
            # --- Feature Engineering Aggiuntivo: Analisi Finanziaria Cluster ---
            if price_col in self.df.columns:
                print("\nAnalisi Finanziaria per Cluster Semantico:")
                cluster_financials = self.df.groupby(cluster_col)[price_col].agg(
                    conteggio='count',
                    valore_medio='mean',
                    valore_mediano='median',
                    valore_totale='sum'
                ).sort_values(by='valore_medio', ascending=False)
                
                # Formattazione per leggibilità
                cluster_financials['valore_medio'] = cluster_financials['valore_medio'].map('€{:,.0f}'.format)
                cluster_financials['valore_mediano'] = cluster_financials['valore_mediano'].map('€{:,.0f}'.format)
                cluster_financials['valore_totale'] = cluster_financials['valore_totale'].map('€{:,.0f}'.format)
                
                print(cluster_financials)

        except Exception as e:
            print(f"Errore durante l'esecuzione di K-Means: {e}")


class DataVisualizer:
    """Gestisce la creazione e il salvataggio di tutte le visualizzazioni."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)

    def _save_plot(self, figure: plt.Figure, filename: str):
        """Salva il grafico nella directory PLOTS_DIR."""
        path = os.path.join(PLOTS_DIR, filename)
        try:
            figure.savefig(path, bbox_inches='tight', dpi=150)
            print(f"Grafico salvato in: {path}")
        except Exception as e:
            print(f"Errore durante il salvataggio del grafico {filename}: {e}")

    def _format_currency_axis(self, ax, axis='y', scale='auto'):
        """Formatta gli assi con valori monetari in modo leggibile."""
        def format_func(value, tick_number):
            scales = {1e9: 'B', 1e6: 'M', 1e3: 'K'}
            if scale == 'auto':
                for s, suffix in scales.items():
                    if abs(value) >= s:
                        return f'€{value/s:.1f}{suffix}'
                return f'€{value:.0f}'
            else:
                s_val = {'K': 1e3, 'M': 1e6, 'B': 1e9}.get(scale, 1)
                fmt = '.0f' if scale == 'K' else '.1f'
                return f'€{value/s_val:{fmt}}{scale}'
        
        formatter = FuncFormatter(format_func)
        if axis == 'y':
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.xaxis.set_major_formatter(formatter)

    def visualize_text_clusters(self, prefix: str = 'cpvs_sem'):
        """Visualizza i cluster semantici con Plotly."""
        print("\n--- Visualizzazione Cluster Semantici ---")
        x_col, y_col = f'{prefix}_x', f'{prefix}_y'
        cluster_col = f"{prefix.split('_')[0]}_cluster"
        raw_text_col = f"{prefix.split('_')[0]}_raw_text"

        if px is None:
            print("Plotly non disponibile.")
            return
        if cluster_col not in self.df.columns:
            print(f"Colonna cluster '{cluster_col}' non trovata.")
            return

        plot_data = self.df.dropna(subset=[x_col, y_col, cluster_col]).copy()
        plot_data[cluster_col] = plot_data[cluster_col].astype(str)

        if plot_data.empty:
            print("Nessun dato valido per visualizzare i cluster.")
            return

        num_clusters = plot_data[cluster_col].nunique()
        color_sequence = px.colors.qualitative.Vivid[:num_clusters]

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
                xaxis_showgrid=False, yaxis_showgrid=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7))

            cluster_path = os.path.join(PLOTS_DIR, f'13_{cluster_col}_semantic_clusters.html')
            fig.write_html(cluster_path)
            print(f"Grafico cluster semantici salvato in: {cluster_path}")
            fig.show()
        except Exception as e:
            print(f"Errore creazione grafico Plotly cluster: {e}")

    def analyze_award_criteria_and_price(self):
        """Genera visualizzazioni per analizzare criteri di aggiudicazione e prezzi."""
        print("\n--- Analisi Approfondita: Criteri di Aggiudicazione e Prezzi ---")
        award_col = 'Award criteria class'
        price_col = 'Base Bid Price (€)'
        district_col = 'District'
        deadline_col = 'Execution deadline (days)_numeric'

        if not all(col in self.df.columns for col in [award_col, price_col]):
            return

        # 1. Box Plot: Prezzo per Criterio
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(
            x=award_col, y=price_col, data=self.df,
            palette='Set2', ax=ax, hue=award_col,
            linewidth=1.5, legend=False
        )
        ax.set_title('Distribuzione Prezzo Base per Criterio di Aggiudicazione', fontsize=18, pad=15)
        ax.set_ylabel('Prezzo Base (€) (Scala Logaritmica)', fontsize=14)
        ax.set_xlabel('Criterio di Aggiudicazione', fontsize=14)
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=15, labelsize=12)
        sns.despine()
        fig.tight_layout()
        self._save_plot(fig, '20_price_distribution_by_award_criteria.png')
        plt.show()
        

        # 2. Bar Plot: Conteggio Criteri per Distretto
        if district_col in self.df.columns:
            fig, ax = plt.subplots(figsize=(15, max(10, self.df[district_col].nunique() * 0.4)))
            order = self.df[district_col].value_counts().index
            sns.countplot(
                y=district_col, hue=award_col, data=self.df,
                order=order, palette='viridis', edgecolor='grey',
                linewidth=0.5, ax=ax
            )
            ax.set_title('Contratti per Criterio di Aggiudicazione e Distretto', fontsize=18, pad=15)
            ax.set_xlabel('Numero di Contratti', fontsize=14)
            ax.set_ylabel('Distretto', fontsize=14)
            ax.legend(title='Criterio Aggiudicazione', title_fontsize='13', fontsize='12')
            sns.despine()
            fig.tight_layout()
            self._save_plot(fig, '21_award_criteria_by_district.png')
            plt.show()
            

        # 3. KDE Plot: Prezzo vs. Scadenza per Criterio
        if deadline_col in self.df.columns:
            plot_data_kde = self.df[(self.df[price_col] > 0) & (self.df[deadline_col] > 0)].copy()
            if not plot_data_kde.empty:
                g = sns.displot(
                    data=plot_data_kde,
                    x=price_col, y=deadline_col, hue=award_col,
                    kind='kde', fill=True, height=8, aspect=1.2,
                    palette='viridis', log_scale=(True, True)
                )
                g.fig.suptitle('Densità Prezzo vs. Scadenza per Criterio (Scala Log)', y=1.03, fontsize=18)
                g.set_axis_labels('Prezzo Base (€)', 'Scadenza Esecuzione (giorni)', fontsize=14)
                g._legend.set_title('Criterio Aggiudicazione')
                g.fig.tight_layout(rect=[0, 0.03, 1, 0.98])
                self._save_plot(g.fig, '22_price_vs_deadline_density_by_award_criteria.png')
                plt.show()
                plt.close(g.fig)

    def generate_final_visualizations(self):
        """Genera visualizzazioni riepilogative finali."""
        print("\n--- Generazione Visualizzazioni Finali Riepilogative ---")

        # Grafico Conteggio per Distretto
        district_col = 'District'
        if district_col in self.df.columns:
            fig, ax = plt.subplots(figsize=(12, max(8, self.df[district_col].nunique() * 0.4)))
            order = self.df[district_col].value_counts().index
            sns.countplot(y=district_col, data=self.df, order=order,
                          palette='viridis', ax=ax, hue=district_col, legend=False,
                          edgecolor='grey', linewidth=0.7)
            ax.set_title('Numero di Contratti per Distretto', fontsize=18, pad=15)
            ax.set_xlabel('Numero di Contratti', fontsize=14)
            ax.set_ylabel('Distretto', fontsize=14)
            sns.despine()
            fig.tight_layout()
            self._save_plot(fig, '03_district_distribution.png')
            plt.show()
            

        # Distribuzioni Feature Chiave (subplot)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Distribuzioni di Feature Chiave Discretizzate', fontsize=20, y=1.02)
        plot_specs = [
            {'col': 'Award criteria class', 'ax': axes[0, 0], 'title': 'Criteri di Aggiudicazione'},
            {'col': 'Base Bid Price (€)_category', 'ax': axes[0, 1], 'title': 'Categoria Prezzo Base'},
            {'col': 'Execution deadline (days)', 'ax': axes[1, 0], 'title': 'Durata Scadenza Esecuzione'},
            {'col': 'Diference between close and signing dates', 'ax': axes[1, 1], 'title': 'Durata Differenza Date'}
        ]
        for spec in plot_specs:
            col_name = spec['col']
            if col_name in self.df.columns:
                order = self.df[col_name].value_counts().index
                sns.countplot(x=col_name, data=self.df, palette='magma', ax=spec['ax'],
                              order=order, hue=col_name, legend=False,
                              edgecolor='black', linewidth=0.8)
                spec['ax'].set_title(spec['title'], fontsize=16)
                spec['ax'].set_xlabel('', fontsize=14)
                spec['ax'].set_ylabel('Conteggio', fontsize=14)
                spec['ax'].tick_params(axis='x', rotation=30, labelsize=12)
                sns.despine(ax=spec['ax'])
            else:
                spec['ax'].set_title(f"{spec['title']}\n(Colonna mancante)", fontsize=16)
                spec['ax'].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        self._save_plot(fig, '04_key_features_distribution.png')
        plt.show()
        

    def generate_geospatial_visualizations(self, geojson_path: str):
        """Genera mappa coropletica del prezzo medio per distretto."""
        print("\n--- Generazione Mappa Geospaziale (Prezzo Medio) ---")
        district_col = 'District'
        price_col = 'Base Bid Price (€)'

        if px is None or not Path(geojson_path).exists() or not all(c in self.df.columns for c in [district_col, price_col]):
            print("Prerequisiti per la mappa non soddisfatti (Plotly, GeoJSON, Colonne). Salto.")
            return

        try:
            with open(geojson_path, 'r', encoding='utf-8') as f:
                districts_geojson = json.load(f)
        except Exception as e:
            print(f"Errore caricamento GeoJSON: {e}")
            return

        geo_metrics = self.df.groupby(district_col).agg(
            contracts=(district_col, 'size'),
            base_bid_mean=(price_col, 'mean')
        ).reset_index()

        if geo_metrics.empty:
            print("Nessun dato aggregato per la mappa.")
            return

        try:
            fig_map = px.choropleth_mapbox(
                geo_metrics,
                geojson=districts_geojson,
                locations=district_col,
                featureidkey='properties.dis_name',
                color='base_bid_mean',
                color_continuous_scale='Viridis',
                mapbox_style='carto-positron',
                zoom=5.5,
                center={'lat': 39.5, 'lon': -8.0},
                opacity=0.7,
                hover_name=district_col,
                hover_data={'contracts': True, 'base_bid_mean': ':.0f'}
            )
            fig_map.update_layout(
                title_text='Valore Medio del Prezzo Base (€) per Distretto',
                title_font_size=20,
                coloraxis_colorbar=dict(title="Prezzo Medio (€)")
            )
            map_path = os.path.join(PLOTS_DIR, '05_map_base_bid_by_district.html')
            fig_map.write_html(map_path)
            print(f"Mappa prezzo medio salvata in: {map_path}")
            fig_map.show()
        except Exception as e:
            print(f"Errore creazione mappa: {e}")

    def generate_advanced_visualizations(self):
        """Genera visualizzazioni avanzate: donut, violin, heatmap."""
        print("\n--- Generazione Visualizzazioni Avanzate ---")
        award_col = 'Award criteria class'
        price_col = 'Base Bid Price (€)'
        year_col = 'Signing Year'
        deadline_num_col = 'Execution deadline (days)_numeric'
        diff_dates_num_col = 'Diference between close and signing dates_numeric'
        diff_price_num_col = 'Difference between the effective and initial price (€)_numeric'
        
        # 1. Donut Chart Moderno per Criteri
        if award_col in self.df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            award_counts = self.df[award_col].value_counts()
            colors = sns.color_palette('plasma', len(award_counts))
            explode = [0.05] * len(award_counts) # Leggera esplosione per tutte

            wedges, texts, autotexts = ax.pie(
                award_counts, labels=None, autopct='%1.1f%%',
                startangle=90, colors=colors,
                pctdistance=0.85, explode=explode,
                textprops={'weight': 'bold', 'size': 13, 'color': 'white'}
            )
            # Styling autotexts
            for at in autotexts:
                at.set_color('white')

            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax.add_artist(centre_circle)

            ax.legend(wedges, award_counts.index, title="Criteri di Aggiudicazione",
                      loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=11, title_fontsize=13)
            
            ax.set_title('Distribuzione Criteri di Aggiudicazione', 
                         fontsize=20, weight='bold', pad=20)
            
            total = len(self.df)
            ax.text(0, 0, f'{total:,}\nContratti', ha='center', va='center',
                    fontsize=18, weight='bold', color='#2C3E50')

            self._save_plot(fig, '09_award_criteria_donut.png')
            plt.show()
            

        # 2. Violin Plot (Sostituito con Boxenplot per leggibilità su larga scala)
        if award_col in self.df.columns and price_col in self.df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.boxenplot(
                x=award_col, y=price_col, data=self.df,
                palette='viridis', ax=ax, hue=award_col, legend=False,
                linewidth=1.5
            )
            ax.set_title('Distribuzione Prezzi per Criterio (Boxen Plot)', 
                         fontsize=18, weight='bold', pad=20)
            ax.set_ylabel('Prezzo Base (€) (Scala Logaritmica)', fontsize=14, weight='bold')
            ax.set_xlabel('Criterio di Aggiudicazione', fontsize=14, weight='bold')
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=15, labelsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            self._format_currency_axis(ax, axis='y', scale='M') # Formatta asse Y in Milioni
            sns.despine()
            plt.tight_layout()
            self._save_plot(fig, '10_price_distribution_by_award_criteria_boxen.png')
            plt.show()
            
            
        # 3. Andamento Temporale (Stilizzato)
        if year_col in self.df.columns and price_col in self.df.columns:
            self.df[year_col] = pd.to_numeric(self.df[year_col], errors='coerce')
            temporal_df = self.df.dropna(subset=[year_col]).groupby(year_col).agg(
                num_contracts=(year_col, 'size'),
                avg_price=(price_col, 'mean')
            ).reset_index()

            if not temporal_df.empty:
                fig, ax1 = plt.subplots(figsize=(14, 7))
                ax1.set_title('Andamento Temporale: Numero Contratti e Prezzo Medio', fontsize=18, pad=15)
                ax1.set_xlabel('Anno di Firma', fontsize=14)

                color1 = 'tab:blue'
                ax1.set_ylabel('Numero di Contratti', color=color1, fontsize=14)
                ax1.plot(temporal_df[year_col], temporal_df['num_contracts'], color=color1, marker='o', lw=2.5, label='Numero Contratti')
                ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)

                ax2 = ax1.twinx()
                color2 = 'tab:red'
                ax2.set_ylabel('Prezzo Medio Base (€)', color=color2, fontsize=14)
                ax2.plot(temporal_df[year_col], temporal_df['avg_price'], color=color2, marker='x', linestyle='--', lw=2, label='Prezzo Medio')
                ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
                self._format_currency_axis(ax2, axis='y', scale='auto') # Formatta asse Y
                
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12)
                
                sns.despine(right=False)
                fig.tight_layout()
                self._save_plot(fig, '11_temporal_trends.png')
                plt.show()
                

        # 4. Heatmap Correlazione Migliorata
        valid_corr_cols = [col for col in [price_col, deadline_num_col, diff_dates_num_col, diff_price_num_col, year_col] 
                           if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])]

        if len(valid_corr_cols) > 1:
            corr_matrix = self.df[valid_corr_cols].corr()
            col_abbrev = {
                price_col: 'Prezzo Base', deadline_num_col: 'Scadenza',
                diff_dates_num_col: 'Diff. Date', diff_price_num_col: 'Var. Prezzo',
                year_col: 'Anno'
            }
            corr_matrix.rename(columns=col_abbrev, index=col_abbrev, inplace=True)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                        center=0, linewidths=2, linecolor='white',
                        cbar_kws={'label': 'Correlazione di Pearson', 'shrink': 0.8},
                        ax=ax, mask=mask, vmin=-1, vmax=1,
                        annot_kws={'size': 12, 'weight': 'bold'})
            
            ax.set_title('Matrice di Correlazione - Feature Numeriche', 
                         fontsize=20, weight='bold', pad=20)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
            
            plt.tight_layout()
            self._save_plot(fig, '12_correlation_heatmap_enhanced.png')
            plt.show()
            

    def generate_price_intensity_plot(self):
        """Genera grafico 2D della relazione prezzo/intensità con contour plot."""
        print("\n--- Generazione Grafico Intensità Prezzo ---")
        price_col = 'Base Bid Price (€)'
        price_day_col = 'Price per Day'
        
        if not all(c in self.df.columns for c in [price_col, price_day_col]):
            return

        plot_data = self.df[(self.df[price_col] > 1) & (self.df[price_day_col] > 1)][[price_col, price_day_col]].dropna().copy()
        
        if len(plot_data) < 100: # Non abbastanza dati per un contour plot
            print("Dati insufficienti per grafico intensità (fallback a scatter).")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(plot_data[price_col], plot_data[price_day_col], alpha=0.5)
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            try:
                # Jointplot con KDE (più pulito di hexbin)
                g = sns.jointplot(
                    data=plot_data,
                    x=price_col,
                    y=price_day_col,
                    kind='kde', # Stima densità
                    fill=True,
                    cmap='viridis',
                    height=10,
                    log_scale=(True, True), # Scala logaritmica
                    marginal_kws=dict(fill=True, color='blue', alpha=0.6)
                )
                g.fig.suptitle('Intensità Prezzo: Prezzo Base vs. Prezzo/Giorno (Scala Log)', y=1.03, fontsize=18)
                g.set_axis_labels('Prezzo Base (€)', 'Prezzo al Giorno (€)', fontsize=14)
                
                # Formattazione assi log
                g.ax_joint.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'€{x/1e6:.1f}M' if x >= 1e6 else (f'€{x/1e3:.0f}K' if x >= 1e3 else f'€{x:.0f}')))
                g.ax_joint.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'€{y/1e3:.0f}K' if y >= 1e3 else f'€{y:.0f}'))

                g.fig.tight_layout(rect=[0, 0.03, 1, 0.98])
                self._save_plot(g.fig, '18_price_intensity_kde.png')
                plt.show()
                plt.close(g.fig)
            except Exception as e:
                print(f"Errore durante la generazione del grafico KDE: {e}")

    def generate_role_based_visualizations(self):
        """Genera visualizzazioni per analisti finanziari e project manager."""
        print("\n--- Generazione Visualizzazioni per Ruoli Specifici ---")
        year_col = 'Signing Year'
        award_col = 'Award criteria class'
        price_col = 'Base Bid Price (€)'
        eu_pub_col = 'Published in the EU journal'
        district_col = 'District'
        deadline_num_col = 'Execution deadline (days)_numeric'

        # 1. Financial: Valore Totale per Anno e Criterio (Stacked Bar)
        if all(c in self.df.columns for c in [year_col, award_col, price_col]):
            financial_df = self.df.groupby([year_col, award_col])[price_col].sum().unstack().fillna(0)
            if not financial_df.empty:
                fig, ax = plt.subplots(figsize=(16, 8))
                financial_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', edgecolor='grey', linewidth=0.5)
                ax.set_title('Valore Totale Contratti (€) per Anno e Criterio', fontsize=18, pad=15)
                ax.set_ylabel('Valore Totale (€)', fontsize=14)
                ax.set_xlabel('Anno di Firma', fontsize=14)
                ax.tick_params(axis='x', rotation=45, labelsize=12)
                self._format_currency_axis(ax, axis='y', scale='M') # Formatta asse Y
                ax.legend(title='Criterio Aggiudicazione', bbox_to_anchor=(1.05, 1), loc='upper left')
                sns.despine()
                fig.tight_layout()
                self._save_plot(fig, '14_financial_stacked_bar_value_by_year.png')
                plt.show()
                

        # 2. Financial: Confronto Prezzo per Pubblicazione EU
        if all(c in self.df.columns for c in [eu_pub_col, price_col]):
            if self.df[eu_pub_col].nunique() <= 5:
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.boxplot(x=eu_pub_col, y=price_col, data=self.df, ax=ax, palette='coolwarm', linewidth=1.5)
                ax.set_title('Confronto Prezzo Base per Pubblicazione in Gazzetta EU', fontsize=18, pad=15)
                ax.set_ylabel('Prezzo Base (€) (Scala Logaritmica)', fontsize=14)
                ax.set_xticklabels(['Non Pubblicato (0)', 'Pubblicato (1)'])
                ax.set_xlabel('Pubblicato in Gazzetta EU', fontsize=14)
                ax.set_yscale('log')
                self._format_currency_axis(ax, axis='y', scale='auto')
                sns.despine()
                fig.tight_layout()
                self._save_plot(fig, '15_financial_price_vs_eu_publication.png')
                plt.show()
                

        # 3. Manager: Scadenza Media per Distretto
        if all(c in self.df.columns for c in [district_col, deadline_num_col]):
            manager_df = self.df.groupby(district_col)[deadline_num_col].mean().sort_values(ascending=True)
            if not manager_df.empty:
                fig, ax = plt.subplots(figsize=(12, max(8, len(manager_df) * 0.4)))
                colors = sns.color_palette('coolwarm_r', len(manager_df)) # Invertito
                bars = manager_df.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.7)
                ax.set_title('Scadenza Media di Esecuzione per Distretto', fontsize=18, pad=15)
                ax.set_xlabel('Giorni Medi di Esecuzione', fontsize=14)
                ax.set_ylabel('Distretto', fontsize=14)
                for i, v in enumerate(manager_df):
                    ax.text(v + manager_df.max()*0.01, i, f'{v:.0f}', va='center', fontsize=10)
                sns.despine()
                fig.tight_layout()
                self._save_plot(fig, '16_manager_avg_deadline_by_district.png')
                plt.show()
                

    def generate_additional_geospatial_plot(self, geojson_path: str):
        """Genera mappa coropletica del numero di contratti per distretto."""
        print("\n--- Generazione Mappa Geospaziale (Numero Contratti) ---")
        district_col = 'District'
        if px is None or not Path(geojson_path).exists() or district_col not in self.df.columns:
            print("Prerequisiti per la mappa contratti non soddisfatti. Salto.")
            return

        try:
            with open(geojson_path, 'r', encoding='utf-8') as f:
                districts_geojson = json.load(f)
        except Exception as e:
            print(f"Errore caricamento GeoJSON: {e}")
            return

        geo_metrics = self.df.groupby(district_col).size().reset_index(name='contracts')
        if geo_metrics.empty: return

        try:
            fig_map = px.choropleth_mapbox(
                geo_metrics, geojson=districts_geojson, locations=district_col,
                featureidkey='properties.dis_name', color='contracts',
                color_continuous_scale='Plasma', mapbox_style='carto-positron',
                zoom=5.5, center={'lat': 39.5, 'lon': -8.0}, opacity=0.7,
                hover_data={'contracts': True}
            )
            fig_map.update_layout(
                title_text='Numero di Contratti per Distretto',
                title_font_size=20,
                coloraxis_colorbar=dict(title="Numero Contratti")
            )
            map_path = os.path.join(PLOTS_DIR, '17_map_contracts_by_district.html')
            fig_map.write_html(map_path)
            print(f"Mappa numero contratti salvata in: {map_path}")
            fig_map.show()
        except Exception as e:
            print(f"Errore creazione mappa contratti: {e}")

    def generate_word_cloud(self, text_col='cpvs_clean_text'):
        """Genera una word cloud dalle keyword testuali."""
        print(f"\n--- Generazione Word Cloud per '{text_col}' ---")
        if WordCloud is None or text_col not in self.df.columns:
            print("Prerequisiti per Word Cloud non soddisfatti. Salto.")
            return

        text = ' '.join(self.df[text_col].fillna('').astype(str))
        if not text.strip():
            print("Nessun testo valido per generare la word cloud.")
            return

        try:
            cloud = WordCloud(width=1200, height=600, background_color='white',
                              colormap='viridis', max_words=150, contour_width=1,
                              contour_color='steelblue', random_state=42).generate(text)
            plt.figure(figsize=(15, 7))
            plt.imshow(cloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud delle Keyword ({text_col})', fontsize=18, pad=15)
            self._save_plot(plt.gcf(), '06_wordcloud_cpvs.png')
            plt.show()
            
        except Exception as e:
            print(f"Errore durante la generazione della word cloud: {e}")

    def generate_temporal_heatmap(self):
        """Genera heatmap temporale: contratti per mese e anno."""
        print("\n--- Generazione Heatmap Temporale ---")
        year_col = 'Signing Year'
        month_col = 'Signing Month'
        
        if not all(c in self.df.columns for c in [year_col, month_col]):
            print(f"Colonne '{year_col}' o '{month_col}' mancanti. Salto.")
            return

        temporal_data = self.df.groupby([year_col, month_col]).size().reset_index(name='contracts')
        pivot_table = temporal_data.pivot(index=month_col, columns=year_col, values='contracts').fillna(0)
        
        # Assicura che l'indice sia 1-12
        pivot_table = pivot_table.reindex(range(1, 13)).fillna(0) 
        month_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 
                       'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
        pivot_table.index = month_names
        
        fig, ax = plt.subplots(figsize=(max(10, pivot_table.shape[1] * 0.8), 8))
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlGnBu', # Palette diversa
                    linewidths=1, linecolor='white', cbar_kws={'label': 'Numero Contratti'},
                    ax=ax, vmin=0) # Vmin a 0
        ax.set_title('Intensità Temporale: Numero di Contratti per Mese e Anno', 
                     fontsize=20, pad=20, weight='bold')
        ax.set_xlabel('Anno di Firma', fontsize=14, weight='bold')
        ax.set_ylabel('Mese', fontsize=14, weight='bold')
        ax.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        self._save_plot(fig, '07_temporal_heatmap.png')
        plt.show()
        

    def generate_stacked_area_chart(self):
        """Genera stacked area chart dell'evoluzione del valore totale dei contratti."""
        print("\n--- Generazione Stacked Area Chart ---")
        year_col = 'Signing Year'
        price_col = 'Base Bid Price (€)'
        award_col = 'Award criteria class'
        
        if not all(c in self.df.columns for c in [year_col, price_col, award_col]):
            print("Colonne necessarie mancanti. Salto.")
            return

        temporal_value = self.df.groupby([year_col, award_col])[price_col].sum().reset_index()
        pivot_value = temporal_value.pivot(index=year_col, columns=award_col, values=price_col).fillna(0)
        
        if pivot_value.empty:
            print("Dati insufficienti per lo stacked area chart.")
            return

        fig, ax = plt.subplots(figsize=(16, 9))
        colors = sns.color_palette('Set2', len(pivot_value.columns))
        pivot_value.plot(kind='area', stacked=True, ax=ax, alpha=0.8, 
                         color=colors, linewidth=1)
        
        self._format_currency_axis(ax, axis='y', scale='M') # Formatta Y in Milioni
        
        ax.set_title('Evoluzione del Valore Totale dei Contratti nel Tempo', 
                     fontsize=22, pad=20, weight='bold')
        ax.set_xlabel('Anno di Firma', fontsize=14, weight='bold')
        ax.set_ylabel('Valore Totale Contratti', fontsize=14, weight='bold')
        ax.legend(title='Criterio Aggiudicazione', title_fontsize=12, fontsize=11,
                  loc='upper left', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        sns.despine()
        plt.tight_layout()
        self._save_plot(fig, '08_stacked_area_value_evolution.png')
        plt.show()
        

    def generate_treemap_budget_distribution(self):
        """Genera treemap della distribuzione del budget per distretto e criterio."""
        print("\n--- Generazione Treemap Distribuzione Budget ---")
        if px is None: return
            
        district_col = 'District'
        award_col = 'Award criteria class'
        price_col = 'Base Bid Price (€)'
        
        if not all(c in self.df.columns for c in [district_col, award_col, price_col]):
            return

        treemap_data = self.df.groupby([district_col, award_col])[price_col].sum().reset_index()
        treemap_data.columns = ['District', 'Criterion', 'Total_Value']
        treemap_data = treemap_data[treemap_data['Total_Value'] > 0]
        
        if treemap_data.empty: return

        fig = px.treemap(
            treemap_data,
            path=[px.Constant("Tutti i Distretti"), 'District', 'Criterion'], # Aggiunge root
            values='Total_Value',
            title='Distribuzione Budget: Valore Contratti per Distretto e Criterio',
            color='Total_Value',
            color_continuous_scale='Viridis',
            hover_name='Criterion',
            hover_data={'District': False, 'Criterion': False, 'Total_Value': ':,.0f'}
        )
        
        fig.update_layout(
            title_font_size=22,
            height=800,
            coloraxis_colorbar_title_text='Valore Totale (€)'
        )
        fig.update_traces(
            textinfo='label+percent root', # Mostra % del totale
            marker=dict(line=dict(width=1, color='white'))
        )
        
        treemap_path = os.path.join(PLOTS_DIR, '09_treemap_budget_distribution.html')
        fig.write_html(treemap_path)
        print(f"Treemap interattivo salvato in: {treemap_path}")
        fig.show()

    def generate_kpi_dashboard(self):
        """Genera dashboard con KPI principali e grafici di supporto."""
        print("\n--- Generazione Dashboard KPI ---")
        price_col = 'Base Bid Price (€)'
        deadline_col = 'Execution deadline (days)_numeric'
        district_col = 'District'
        year_col = 'Signing Year'
        
        if not all(c in self.df.columns for c in [price_col, deadline_col, district_col, year_col]):
            print("Colonne necessarie mancanti. Salto dashboard KPI.")
            return

        fig = plt.figure(figsize=(18, 12)) # Leggermente più alto
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        fig.patch.set_facecolor('#F8F9FA') # Sfondo leggero

        # Funzione helper per KPI Card
        def create_kpi_card(ax, value, title, subtitle, color):
            ax.axis('off')
            ax.text(0.5, 0.65, value, ha='center', va='center', 
                   fontsize=44, weight='bold', color=color) # Fontsize ridotto
            ax.text(0.5, 0.35, title, ha='center', va='center', 
                   fontsize=16, weight='bold', color='#2C3E50')
            ax.text(0.5, 0.15, subtitle, ha='center', va='center', 
                   fontsize=11, color='#7F8C8D')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            rect = plt.Rectangle((0, 0), 1, 1, 
                                facecolor='white', edgecolor=color, 
                                linewidth=3, transform=ax.transAxes,
                                alpha=0.1) # Sfondo più leggero
            ax.add_patch(rect)
        
        # KPI 1
        ax1 = fig.add_subplot(gs[0, 0])
        total_contracts = len(self.df)
        create_kpi_card(ax1, f'{total_contracts:,}', 'Contratti Totali', 
                       'Dataset analizzato', '#3498DB')
        
        # KPI 2
        ax2 = fig.add_subplot(gs[0, 1])
        avg_value = self.df[price_col].mean()
        create_kpi_card(ax2, f'€{avg_value/1e6:.2f}M', 'Valore Medio', 
                       'Per contratto', '#2ECC71')
        
        # KPI 3
        ax3 = fig.add_subplot(gs[0, 2])
        avg_deadline = self.df[deadline_col].mean()
        create_kpi_card(ax3, f'{avg_deadline:.0f}', 'Giorni Medi', 
                       'Scadenza esecuzione', '#E74C3C')
        
        # Grafico 4: Top 5 Distretti per Valore
        ax4 = fig.add_subplot(gs[1, :2])
        top_districts = self.df.groupby(district_col)[price_col].sum().nlargest(5).sort_values(ascending=True)
        colors_bar = sns.color_palette('viridis', len(top_districts))
        bars = ax4.barh(range(len(top_districts)), top_districts.values, color=colors_bar, edgecolor='black', linewidth=0.5)
        ax4.set_yticks(range(len(top_districts)))
        ax4.set_yticklabels(top_districts.index, fontsize=12, weight='bold')
        ax4.set_xlabel('Valore Totale (€)', fontsize=14, weight='bold')
        ax4.set_title('Top 5 Distretti per Valore Contratti', fontsize=16, weight='bold', pad=10)
        self._format_currency_axis(ax4, axis='x', scale='M') # Formatta asse X in Milioni
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        sns.despine(ax=ax4, left=True)
        
        # Grafico 5: Distribuzione Prezzi (Boxenplot)
        ax5 = fig.add_subplot(gs[1, 2])
        if 'Award criteria class' in self.df.columns:
            sns.boxenplot(x='Award criteria class', y=price_col, data=self.df,
                          palette='Set2', ax=ax5, linewidth=1.5,
                          showfliers=False) # Nasconde outlier per pulizia
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=15, ha='right')
            ax5.set_xlabel('')
            ax5.set_ylabel('Prezzo (€, Log)', fontsize=12, weight='bold')
            ax5.set_title('Distribuzione Prezzi per Criterio', fontsize=16, weight='bold', pad=10)
            ax5.set_yscale('log')
            ax5.grid(axis='y', alpha=0.3, linestyle='--')
            sns.despine(ax=ax5)
        
        # Grafico 6: Timeline
        ax6 = fig.add_subplot(gs[2, :])
        yearly = self.df.groupby(year_col).agg(
            count=(year_col, 'size'),
            value=(price_col, 'sum')
        )
        
        ax6_twin = ax6.twinx()
        
        # Linea per Conteggio
        ax6.plot(yearly.index, yearly['count'], marker='o', linewidth=3, 
                 markersize=8, color='#3498DB', label='Numero Contratti',
                 alpha=0.8)
        # Barre per Valore
        ax6_twin.bar(yearly.index, yearly['value'], color='#E74C3C', 
                     alpha=0.5, label='Valore Totale (€)', width=0.6)
        
        ax6.set_xlabel('Anno', fontsize=14, weight='bold')
        ax6.set_ylabel('Numero Contratti', fontsize=14, weight='bold', color='#3498DB')
        ax6_twin.set_ylabel('Valore Totale (€)', fontsize=14, weight='bold', color='#E74C3C')
        ax6.set_title('Andamento Temporale: Volume vs. Valore', fontsize=16, weight='bold', pad=10)
        ax6.tick_params(axis='y', labelcolor='#3498DB')
        ax6_twin.tick_params(axis='y', labelcolor='#E74C3C')
        self._format_currency_axis(ax6_twin, axis='y', scale='M')
        
        lines, labels = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12)
        
        sns.despine(ax=ax6, right=False)
        ax6.grid(True, alpha=0.2, linestyle='--')
        
        plt.suptitle('Dashboard Riepilogativa - Appalti Pubblici Portoghesi', 
                     fontsize=24, weight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_plot(fig, '10_kpi_dashboard.png')
        plt.show()
        

    def generate_pdf_report(self, report_title: str = 'PPP Portugal - Report di Analisi Narrativa', output_path: str = None):
        """Genera un report PDF narrativo con ReportLab, includendo riferimenti ai grafici HTML."""
        if SimpleDocTemplate is None:
            print("ReportLab non trovato. Salto generazione PDF.")
            return

        if output_path is None:
            output_path = os.path.join(PLOTS_DIR, 'PPP_narrative_report.pdf')
        print(f"\n--- Generazione Report PDF Narrativo: {output_path} ---")

        plot_info = {
            # --- Grafici di Preprocessing ---
            '01_missing_values_percentage.png': {
                "title": "Analisi Preliminare: Valori Mancanti",
                "description": "Questo grafico mostra la percentuale di dati mancanti per ogni colonna. Colonne con un'alta percentuale di valori nulli (es. 'Conclusion of a framework agreement') sono state rimosse perché non avrebbero fornito insight affidabili. Questa fase è cruciale per garantire la robustezza dell'analisi."
            },
            '02a_distribution_before_outlier_diference_between_close_and_signing_dates.png': {
                "title": "Analisi Preliminare: Distribuzione 'Differenza Date' (Grezza)",
                "description": "Visualizzazione della distribuzione iniziale della differenza di giorni tra chiusura e firma. La presenza di una coda lunga (outlier) è evidente nel box plot, giustificando la successiva fase di pulizia per stabilizzare le analisi."
            },
            '02a_distribution_before_outlier_execution_deadline_(days).png': {
                "title": "Analisi Preliminare: Distribuzione 'Scadenza Esecuzione' (Grezza)",
                "description": "Visualizzazione della distribuzione iniziale dei giorni di scadenza. Come per la differenza date, la distribuzione è fortemente asimmetrica a destra, con outlier estremi che vengono rimossi per non distorcere le medie e le visualizzazioni successive."
            },
            '04_key_features_distribution.png': {
                "title": "Distribuzione delle Feature Chiave Discretizzate",
                "description": "Questi grafici mostrano come si distribuiscono alcune delle variabili più importanti dopo la pulizia e la discretizzazione. Si conferma che la maggior parte dei contratti usa il 'prezzo più basso' e rientra nelle categorie 'Low' per prezzo e 'Short' per durata."
            },
            
            # --- Grafici di Storytelling Principale ---
            '10_kpi_dashboard.png': {
                "title": "Dashboard Riepilogativa (KPI)",
                "description": "La dashboard principale fornisce una visione d'insieme. Si evidenziano i KPI (Key Performance Indicators) come il numero totale di contratti, il valore medio e la durata media. I grafici di supporto mostrano i distretti principali per valore, la distribuzione dei prezzi e l'andamento temporale aggregato."
            },
            '07_temporal_heatmap.png': {
                "title": "Analisi di Stagionalità (Heatmap)",
                "description": "Questa heatmap analizza la distribuzione dei contratti nel corso dei mesi e degli anni. Si possono identificare pattern stagionali: ad esempio, una maggiore attività di appalto in certi periodi dell'anno (es. fine o inizio anno fiscale) rispetto ad altri (es. agosto)."
            },
            '08_stacked_area_value_evolution.png': {
                "title": "Evoluzione del Valore Totale per Criterio",
                "description": "Questo grafico mostra come il valore *totale* dei contratti (in milioni di €) si è evoluto nel tempo, suddiviso per criterio di aggiudicazione. Si osserva visivamente se il volume finanziario dei contratti 'multifattoriali' (qualità) sta crescendo rispetto a quelli basati solo sul 'prezzo più basso'."
            },
            '03_district_distribution.png': {
                "title": "Distribuzione Geografica dei Contratti (Volume)",
                "description": "Il grafico a barre orizzontali illustra il numero totale di contratti per distretto, ordinati per volume. Emerge una chiara concentrazione nelle aree metropolitane di Lisbona e Porto, che sono i principali centri economici del paese."
            },
            
            # --- GRAFICI HTML ---
            '17_map_contracts_by_district.html': { 
                "title": "Mappa Geografica (Volume Contratti) - [Interattivo]",
                "description": "Questa mappa coropletica (visualizzazione interattiva) conferma geograficamente i dati del grafico a barre. Le aree più scure indicano un maggior numero di contratti, rinforzando l'evidenza della concentrazione su Lisbona e Porto. (Grafico HTML disponibile nella cartella 'plots/')"
            },
             '05_map_base_bid_by_district.html': { 
                "title": "Mappa Geografica (Valore Medio) - [Interattivo]",
                "description": "Contrariamente alla mappa del volume, questa visualizzazione (interattiva) mostra il *valore medio* dei contratti. Emerge un pattern interessante: distretti con *meno* contratti (es. Bragança) possono avere un valore medio *più alto*, suggerendo la presenza di pochi ma grandi progetti infrastrutturali. (Grafico HTML disponibile nella cartella 'plots/')"
            },
            '09_treemap_budget_distribution.html': {
                "title": "Treemap Distribuzione Budget (Distretto/Criterio) - [Interattivo]",
                "description": "Questo Treemap interattivo mostra come il budget totale è allocato. I rettangoli più grandi (es. Lisbona) rappresentano la quota maggiore del budget. All'interno di ogni distretto, la suddivisione per criterio mostra come quel budget è distribuito. (Grafico HTML disponibile nella cartella 'plots/')"
            },
            '13_cpvs_sem_semantic_clusters.html': {
                "title": "Cluster Semantici (Analisi Testuale) - [Interattivo]",
                "description": "Questa visualizzazione interattiva (basata su PCA e K-Means) raggruppa i contratti in base al significato semantico della loro descrizione. I cluster separati (es. cluster 0 vs cluster 1) rappresentano gruppi tematici distinti, come 'grandi lavori di costruzione' vs 'servizi di consulenza e progettazione'. (Grafico HTML disponibile nella cartella 'plots/')"
            },
            # --- Fine Grafici HTML ---

            '21_award_criteria_by_district.png': {
                "title": "Analisi Criteri di Aggiudicazione per Distretto",
                "description": "Questo grafico combina la dimensione geografica con quella decisionale. Mostra come la maggior parte dei distretti si affidi prevalentemente al 'prezzo più basso'. Tuttavia, in centri urbani come Lisbona, la proporzione di contratti 'multifattoriali' è visibilmente più alta."
            },
            '09_award_criteria_donut.png': {
                "title": "Distribuzione dei Criteri di Aggiudicazione",
                "description": "Questa visualizzazione (stile 'donut') mostra la proporzione tra i diversi criteri di aggiudicazione. Il 'prezzo più basso' è chiaramente dominante, ma una fetta significativa di contratti utilizza criteri 'multifattoriali', suggerendo che la qualità è comunque un fattore rilevante."
            },
            '10_price_distribution_by_award_criteria_boxen.png': {
                "title": "Distribuzione Prezzi per Criterio (Boxen Plot)",
                "description": "Il Boxen Plot confronta la distribuzione dei prezzi. Entrambi i criteri mostrano una grande dispersione. I contratti 'multifattoriali' hanno una mediana leggermente più alta, ma la variabilità è tale che il tipo di progetto è probabilmente un driver di costo più forte del solo criterio."
            },
            '18_price_intensity_kde.png': {
                "title": "Analisi Intensità Progetto (Prezzo vs. Costo/Giorno)",
                "description": "Questo grafico di densità (KDE) su scala logaritmica mostra dove si concentrano i contratti. Le aree più scure indicano una maggiore densità. La maggior parte si concentra su prezzi bassi e costi giornalieri bassi (probabilmente manutenzione)."
            },
            '06_wordcloud_cpvs.png': {
                "title": "Temi Principali (Word Cloud CPVS)",
                "description": "La nuvola di parole evidenzia i termini più frequenti nelle descrizioni CPVS (le categorie dei lavori). Termini come 'lavori', 'costruzione', 'manutenzione' dominano, confermando il focus del dataset sul settore edile e infrastrutturale."
            },
            '11_temporal_trends.png': {
                "title": "Andamento Temporale (Contratti vs Prezzo Medio)",
                "description": "Questo grafico a doppia scala mostra l'evoluzione del numero di contratti (blu) e del prezzo medio (rosso). Si possono notare picchi nel volume dei contratti in determinati periodi (es. 2009-2010), non sempre accompagnati da un aumento del prezzo medio."
            },
            '12_correlation_heatmap_enhanced.png': {
                "title": "Matrice di Correlazione (Feature Numeriche)",
                "description": "La heatmap visualizza le correlazioni lineari (coefficiente di Pearson). Si nota una debole correlazione positiva tra prezzo e scadenza. L'assenza di correlazioni forti suggerisce che le relazioni tra le variabili sono complesse."
            },
            '14_financial_stacked_bar_value_by_year.png': {
                "title": "Analisi Finanziaria: Valore Contratti per Anno e Criterio",
                "description": "Questo grafico a barre impilate, utile per un'analisi finanziaria, mostra il valore totale (€) dei contratti per anno, suddiviso per criterio. Permette di osservare se c'è stato uno spostamento nel tempo del valore aggregato."
            },
             '15_financial_price_vs_eu_publication.png': {
                 "title": "Analisi Finanziaria: Confronto Prezzi per Pubblicazione EU",
                 "description": "Questo box plot confronta la distribuzione dei prezzi (in scala log) tra i contratti pubblicati nella Gazzetta UE (1) e quelli non pubblicati (0). I contratti pubblicati mostrano una mediana e una dispersione significativamente più elevate."
             },
            '16_manager_avg_deadline_by_district.png': {
                "title": "Analisi Gestionale: Scadenza Media per Distretto",
                "description": "Questo grafico orizzontale, rilevante per la gestione progettuale, evidenzia come la durata media dei progetti (scadenza) vari in modo significativo da un distretto all'altro. Distretti con scadenze medie più lunghe potrebbero presentare maggiori complessità."
            }
        }

        # Ottieni TUTTI i file definiti in plot_info (sia .png che .html)
        plot_files_ordered = [f for f in plot_info.keys() if os.path.exists(os.path.join(PLOTS_DIR, f))]

        if not plot_files_ordered:
            print("Nessun file PNG o HTML corrispondente trovato in PLOTS_DIR per generare il report.")
            return

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

        for fname in plot_files_ordered:
            info = plot_info[fname]
            img_path = os.path.join(PLOTS_DIR, fname)

            story.append(Paragraph(info['title'], styles['h2']))
            story.append(Spacer(1, 0.1*inch))
            desc_paragraph = Paragraph(textwrap.fill(info['description'], width=100), styles['Normal'])
            story.append(desc_paragraph)
            story.append(Spacer(1, 0.2*inch))

            if fname.endswith('.png'):
                try:
                    with PILImage.open(img_path) as img_pil:
                        width_px, height_px = img_pil.size
                    
                    max_width = 9.5 * inch
                    max_height = 5.8 * inch
                    
                    ratio = min(max_width / width_px, max_height / height_px)
                    img_width = width_px * ratio
                    img_height = height_px * ratio

                    img_reportlab = Image(img_path, width=img_width, height=img_height)
                    img_reportlab.hAlign = 'CENTER' # Centra l'immagine
                    story.append(img_reportlab)
                except Exception as e:
                    print(f"Errore durante l'aggiunta dell'immagine {fname} al PDF: {e}")
                    story.append(Paragraph(f"(Errore nel caricamento immagine {fname})", styles['Italic']))
            
            elif fname.endswith('.html'):
                story.append(Spacer(1, 0.5*inch))
                placeholder_text = f"<i>Nota: Questo è un grafico interattivo (HTML). Aprire il file <b>'{fname}'</b> nella cartella 'plots/' per l'esplorazione.</i>"
                story.append(Paragraph(placeholder_text, styles['Italic']))
                story.append(Spacer(1, 0.5*inch))

            story.append(PageBreak())

        try:
            doc.build(story)
            print(f"Report PDF narrativo generato con successo: {output_path}")
        except Exception as e:
            print(f"Errore durante la costruzione del PDF: {e}")

# %% [markdown]
# ## Esecuzione della Pipeline di Storytelling
# Vengono ora istanziate le classi e invocati i metodi in sequenza per eseguire l'intera pipeline, dalla pulizia alla generazione del report finale.

# %% [markdown]
# ### Fase 1: Preprocessing
# Vengono eseguite tutte le fasi di caricamento, pulizia, feature engineering e salvataggio dei dati.

# %% [markdown]
# #### Fase 1.1: Caricamento e Ispezione Iniziale
# **Obiettivo**: Caricare il dataset e ottenere una prima comprensione della sua struttura, dei tipi di dati e della presenza di valori nulli.
# 
# **Operazioni**:
# 1.  Si istanzia `DataPreprocessor`, che carica il file Excel.
# 2.  Si esegue `inspect_dataframe` per una prima analisi della struttura (tipi di dati, uso memoria), dei valori mancanti e delle prime righe.

# %% [code]
# --- Esecuzione Pipeline ---

# Fase 1: Preprocessing
print("--- INIZIO FASE 1: PREPROCESSING ---")

# 1.1 Caricamento e Ispezione
preprocessor = DataPreprocessor('Datasets/PPPData_EN_1.0.xlsx')
preprocessor.inspect_dataframe("Stato Iniziale del DataFrame")

# %% [markdown]
# #### Fase 1.2: Analisi Valori Mancanti e Pulizia Colonne
# **Obiettivo**: Identificare le colonne problematiche tramite visualizzazione e rimuovere quelle irrilevanti o troppo vuote.
# 
# **Operazioni**:
# 1.  Si istanzia un `DataVisualizer` temporaneo per generare il grafico dei valori mancanti.
# 2.  `analyze_missing_values` visualizza la percentuale di dati mancanti per ogni colonna, guidando la fase successiva.
# 3.  Si definisce una lista (`columns_to_drop`) di colonne da rimuovere (es. 'ID', 'Country', o colonne quasi vuote come 'Conclusion of a framework agreement').
# 4.  `prune_columns` rimuove le colonne specificate.

# %% [code]
# 1.2 Analisi Mancanti e Pulizia Colonne
visualizer_for_prep = DataVisualizer(preprocessor.df) # Istanza temporanea per i grafici di preprocessing
preprocessor.analyze_missing_values(visualizer_for_prep)

columns_to_drop = [
    'Count', 'ID', 'Short Description1', 'Country', 'Award criteria',
    'Involves joint procurement (with several entities) (T/F)',
    'Awarded by a central purchasing body (T/F)',
    'Conclusion of a framework agreement (T/F)', 'Electronic auction (T/F)',
    'Negotiation phase (T/F)', 'Contracting by lots (T/F)', 'Collateral',
    'Contract end type', 'Justification for price change', 'Justification for deadline change'
]
preprocessor.prune_columns(columns_to_drop)

# %% [markdown]
# #### Fase 1.3: Correzione Tipi e Feature Engineering (Date e Testo)
# **Obiettivo**: Standardizzare i dati (es. booleani, stringhe) e arricchire il dataset con feature derivate dalle date e dal testo.
# 
# **Operazioni**:
# 1.  `clean_and_correct`: Si applicano correzioni specifiche (es. mappatura 0/1 per booleani, rimozione spazi extra) e si rimuovono righe con dati palesemente incoerenti o con valori nulli in colonne chiave (es. 'Base Bid Price (€)').
# 2.  `engineer_date_features`: Si convertono le colonne data in formato datetime, si estraggono 'Anno' e 'Mese' e si calcola la differenza in giorni tra la chiusura e la firma (`Diference between close and signing dates`).
# 3.  `engineer_text_features`: Si processa la colonna 'Cpvs Designation'. Si utilizza `TfidfVectorizer` per identificare le 10 keyword (bigrammi/trigrammi) più rilevanti, creando nuove colonne binarie per la loro presenza.

# %% [code]
# 1.3 Correzione Tipi e Feature Engineering (Date e Testo)
preprocessor.clean_and_correct()
preprocessor.engineer_date_features()

cpvs_vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.8, ngram_range=(2, 3), max_features=10)
preprocessor.engineer_text_features('Cpvs Designation', 'cpvs', cpvs_vectorizer)

# %% [markdown]
# #### Fase 1.4: Gestione Outlier e Feature Finanziarie
# **Obiettivo**: Identificare e gestire valori numerici anomali (outlier) e creare nuove metriche finanziarie.
# 
# **Operazioni**:
# 1.  `process_numerical_features`: Si visualizza la distribuzione delle scadenze (prima degli outlier). Si rimuovono gli outlier estremi (tramite IQR) per stabilizzare l'analisi. Le colonne numeriche (es. 'Execution deadline') vengono discretizzate in categorie ('Short', 'Medium', 'Long').
# 2.  `engineer_financial_features`: Si crea la metrica 'Price per Day' (Prezzo/Giorno), normalizzando il costo del contratto sulla sua durata.

# %% [code]
# 1.4 Gestione Outlier e Feature Finanziarie
preprocessor.process_numerical_features(visualizer_for_prep)
del visualizer_for_prep # Rilascia l'istanza temporanea

preprocessor.engineer_financial_features()

# %% [markdown]
# #### Fase 1.5: Imputazione Finale e Salvataggio
# **Obiettivo**: Gestire gli ultimi valori mancanti e salvare il dataset pulito per l'analisi.
# 
# **Operazioni**:
# 1.  `impute_and_finalize`: Si imputano (riempiono) i pochi valori nulli rimasti usando la mediana per le colonne numeriche e la moda per quelle categoriche. Infine, si rimuove qualsiasi riga che dovesse ancora contenere NaN.
# 2.  `save_data`: Si salva il DataFrame pulito e pronto all'uso in un file CSV.

# %% [code]
# 1.5 Imputazione Finale e Salvataggio
preprocessor.impute_and_finalize()
preprocessor.save_data(CLEANED_DATA_PATH)

# %% [markdown]
# #### Fase 1.6: Riepilogo e Analisi Semantica
# **Obiettivo**: Eseguire un controllo finale sul dataset pulito e preparare l'analisi semantica avanzata.
# 
# **Operazioni**:
# 1.  `inspect_dataframe` e `summarize_data`: Si esegue un controllo finale per verificare tipi di dati, assenza di nulli e statistiche descrittive.
# 2.  `compute_semantic_embeddings`: Si trasformano le descrizioni testuali (CPVS) in vettori numerici (embeddings) che catturano il significato.
# 3.  `perform_text_clustering`: Si utilizza K-Means sugli embeddings per raggruppare i contratti in 5 cluster tematici (es. "lavori stradali", "costruzione edifici") e si stampa un'analisi finanziaria preliminare per cluster.

# %% [code]
# 1.6 Riepilogo e Analisi Semantica
preprocessor.inspect_dataframe("Stato Finale del DataFrame (Pronto per Visualizzazione)")
preprocessor.summarize_data()

# Fase 1.b: Analisi Semantica (richiede il DF pulito)
if preprocessor.compute_semantic_embeddings('cpvs_clean_text', prefix='cpvs_sem'):
    preprocessor.perform_text_clustering(n_clusters=5, prefix='cpvs_sem')

print("--- FINE FASE 1: PREPROCESSING ---")


# %% [markdown]
# ### Fase 2: Visualizzazione e Storytelling
# Utilizzando il DataFrame pulito, vengono generate tutte le visualizzazioni in un ordine logico per costruire una narrazione, partendo da una panoramica generale (KPI) fino ad analisi specifiche e testuali.

# %% [code]
# --- Inizio Fase Visualizzazioni ---
print("\n--- INIZIO FASE 2: VISUALIZZAZIONE E REPORTING ---")
# Crea l'istanza principale del Visualizer con il DF finale
visualizer = DataVisualizer(preprocessor.df)

# %% [markdown]
# #### 2.1 Dashboard KPI (Panoramica)
# **Obiettivo**: Fornire una visione d'insieme immediata (Overview) dei principali indicatori del dataset.
# 
# **Insight**: La dashboard mostra i KPI fondamentali (Contratti Totali, Valore Medio, Durata Media), i principali distretti per valore e l'andamento aggregato di volume e valore nel tempo.

# %% [code]
# 1. Panoramica KPI
visualizer.generate_kpi_dashboard()

# %% [markdown]
# #### 2.2 Analisi Temporale (Stagionalità e Valore)
# **Obiettivo**: Esplorare i pattern temporali (Zoom & Filter su Anno/Mese).
# 
# **Insight**: La Heatmap rivela possibili pattern stagionali (es. più contratti a fine anno). Lo Stacked Area Chart mostra l'evoluzione del *valore* totale dei contratti nel tempo, evidenziando la crescita dei contratti "multifattoriali" rispetto a quelli basati solo sul prezzo.

# %% [code]
# 2. Analisi stagionale
visualizer.generate_temporal_heatmap()

# %% [code]
# 3. Evoluzione del valore
visualizer.generate_stacked_area_chart()

# %% [markdown]
# #### 2.3 Analisi Distributiva e Geografica
# **Obiettivo**: Capire *dove* e *come* si distribuiscono i contratti.
# 
# **Insight**: 
# - I grafici distributivi (Distretti, Categorie) confermano la concentrazione su Lisbona/Porto e sul criterio del "prezzo più basso".
# - Le mappe (Details-on-Demand) mostrano un pattern cruciale: il *volume* (N° contratti) è alto a Lisbona, ma il *valore medio* è spesso più alto in distretti rurali, suggerendo grandi progetti infrastrutturali.

# %% [code]
# 4. Distribuzioni (Distretti, Feature chiave)
visualizer.generate_final_visualizations()

# %% [code]
# 5. Mappa (Valore Medio)
visualizer.generate_geospatial_visualizations(geojson_path=GEOJSON_PATH)

# %% [code]
# 6. Mappa (Volume)
visualizer.generate_additional_geospatial_plot(geojson_path=GEOJSON_PATH)

# %% [markdown]
# #### 2.4 Analisi per Ruolo e Budget
# **Obiettivo**: Fornire visualizzazioni mirate per specifici stakeholder (Analista Finanziario, Project Manager).
# 
# **Insight**: 
# - (Finanziario) Il valore dei contratti pubblicati in UE è significativamente più alto.
# - (Manageriale) La durata media dei progetti varia molto per distretto, impattando la pianificazione.
# - Il Treemap mostra come il budget è allocato, confermando che Lisbona (per volume) e i contratti "multifattoriali" (per valore) dominano la spesa.

# %% [code]
# 7. Analisi per Ruolo (Finanziario, Manageriale)
visualizer.generate_role_based_visualizations()

# %% [code]
# 8. Treemap (Budget per Distretto/Criterio)
visualizer.generate_treemap_budget_distribution()

# %% [markdown]
# #### 2.5 Analisi Approfondita: Criteri e Intensità
# **Obiettivo**: Approfondire la relazione tra criteri di aggiudicazione, prezzo e durata.
# 
# **Insight**: 
# - (Boxenplot/KDE) I contratti "multifattoriali" hanno mediane di prezzo più alte e distribuzioni più ampie.
# - (Grafico Intensità) La maggior parte dei contratti si concentra su prezzi e costi giornalieri bassi (probabilmente manutenzione), ma esiste una "coda lunga" di progetti ad alto valore e alta intensità di costo.

# %% [code]
# 9. Analisi Criteri (Boxplot, Conteggio, Densità)
visualizer.analyze_award_criteria_and_price()

# %% [code]
# 10. Visualizzazioni Avanzate (Donut, Heatmap)
visualizer.generate_advanced_visualizations()

# %% [code]
# 11. Analisi Intensità
visualizer.generate_price_intensity_plot()

# %% [markdown]
# #### 2.6 Analisi Testuale (WordCloud e Cluster)
# **Obiettivo**: Visualizzare i temi estratti dalle descrizioni testuali.
# 
# **Insight**: 
# - La WordCloud conferma il focus su "lavori", "costruzione", "manutenzione".
# - Lo scatter plot dei cluster (interattivo) mostra visivamente i raggruppamenti tematici; passando il mouse su un cluster si possono ispezionare i testi, confermando (ad esempio) che un cluster riguarda la "costruzione" e un altro i "servizi".

# %% [code]
# 12. Analisi Testuale (WordCloud)
visualizer.generate_word_cloud(text_col='cpvs_clean_text')

# %% [code]
# 13. Analisi Testuale (Cluster)
visualizer.visualize_text_clusters(prefix='cpvs_sem')

# %% [markdown]
# #### 2.7 Generazione Report PDF Finale
# **Obiettivo**: Consolidare tutte le visualizzazioni statiche (PNG) e la loro interpretazione narrativa in un unico documento PDF condivisibile.

# %% [code]
# 14: Genera Report PDF Finale
visualizer.generate_pdf_report(
    report_title='Report di Analisi Narrativa: Appalti Pubblici Portoghesi (PPP)',
    output_path=os.path.join(PLOTS_DIR, 'Report_Analisi_Appalti_Portogallo.pdf')
)

print("\n--- Pipeline di Visualizzazione e Reporting Completata ---")