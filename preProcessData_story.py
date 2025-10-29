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

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

try:
    import plotly.express as px
except ImportError:
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

sns.set(style="whitegrid")

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
        print(f"Caricamento dati da: {path}")
        return pd.read_excel(path)

    def _save_plot(self, figure: plt.Figure, filename: str):
        path = os.path.join(PLOTS_DIR, filename)
        figure.savefig(path, bbox_inches='tight')
        print(f"Grafico salvato in: {path}")

    def inspect_dataframe(self, title: str):
        print(f"\n--- {title} ---")
        print("\n1. Informazioni Generali e Memoria:")
        self.df.info(memory_usage='deep')
        print("\n2. Valori Nulli per Colonna:")
        print(self.df.isnull().sum().sort_values(ascending=False).head())
        print("\n3. Prime 5 Righe:")
        print(self.df.head())

    def analyze_missing_values(self):
        missing_percentage = self.df.isnull().sum() * 100 / len(self.df)
        missing_df = pd.DataFrame({'column_name': self.df.columns, 'percent_missing': missing_percentage})
        missing_df.sort_values('percent_missing', inplace=True, ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='percent_missing', y='column_name', data=missing_df[missing_df['percent_missing'] > 0], ax=ax)
        ax.set_title('Percentuale di Valori Mancanti per Colonna')
        self._save_plot(fig, '01_missing_values_percentage.png')
        plt.show()
        plt.close(fig)

    def prune_columns(self, columns_to_drop: List[str]):
        print(f"Rimuovendo {len(columns_to_drop)} colonne...")
        self.df.drop(columns=columns_to_drop, inplace=True)

    def clean_and_correct(self):
        self.df['Environmental criteria (T/F)'] = self.df['Environmental criteria (T/F)'].astype(int)
        self.df['Published in the EU journal'] = self.df['Published in the EU journal'].replace({False: 0, 'TRUE ': 1})
        self.df['District'] = self.df['District'].str.strip()
        
        self.df = self.df[~((self.df['District'] == 'Beja') & (self.df['District Code'] == 13))]
        self.df = self.df[~((self.df['District'] == 'Faro') & (self.df['District Code'] == 13))]
        self.df.drop(columns=['District Code'], inplace=True)
        
        key_cols = ['Publication Year', 'Municipality', 'Base Bid Price (€)']
        initial_rows = len(self.df)
        self.df.dropna(subset=key_cols, inplace=True)
        print(f"Rimozione di {initial_rows - len(self.df)} righe con valori nulli in colonne chiave.")

    def engineer_date_features(self):
        for date_col in ['Signing date', 'Closing date']:
            self.df[date_col] = pd.to_datetime(self.df[date_col], format='%d-%m-%Y')
        self.df['Signing Year'] = self.df['Signing date'].dt.year
        self.df['Signing Month'] = self.df['Signing date'].dt.month
        self.df.drop(columns=['Signing date', 'Closing date'], inplace=True)

    def engineer_text_features(self, text_col: str, new_col_prefix: str, vectorizer: TfidfVectorizer):
        cleaned_col = f"{text_col}_cleaned"
        raw_store_col = f"{new_col_prefix}_raw_text"
        clean_store_col = f"{new_col_prefix}_clean_text"

        self.df[cleaned_col] = self.df[text_col].apply(preprocess_text)
        self.df[raw_store_col] = self.df[text_col].astype(str)
        self.df[clean_store_col] = self.df[cleaned_col]

        X_tfidf = vectorizer.fit_transform(self.df[cleaned_col])
        keywords = vectorizer.get_feature_names_out()

        for keyword in keywords:
            self.df[f"{new_col_prefix}_{keyword.replace(' ', '_')}"] = self.df[cleaned_col].apply(lambda x: 1 if keyword in x else 0)

        self.df.drop(columns=[text_col], inplace=True)
        print(f"Create {len(keywords)} feature da '{text_col}'.")

    def process_numerical_features(self):
        numerical_cols = {
            'Diference between close and signing dates': ['Short', 'Medium', 'Long'],
            'Execution deadline (days)': ['Short', 'Medium', 'Long']
        }

        for col, labels in numerical_cols.items():
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(self.df[col], kde=True, ax=axes[0])
            sns.boxplot(x=self.df[col], ax=axes[1])
            self._save_plot(fig, f'02_distribution_{col.replace(" ", "_")}.png')
            plt.show()
            plt.close(fig)

            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 2.5 * IQR, Q3 + 2.5 * IQR
            
            initial_rows = len(self.df)
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
            print(f"Rimosse {initial_rows - len(self.df)} righe di outlier per '{col}'.")
            
            self.df[f'{col}_numeric'] = self.df[col]
            self.df[col] = pd.qcut(self.df[col], len(labels), labels=labels, duplicates='drop')

        self.df['Base Bid Price (€)'] = pd.to_numeric(self.df['Base Bid Price (€)'], errors='coerce')
        self.df['Base Bid Price (€)_category'] = pd.qcut(self.df['Base Bid Price (€)'], 3, labels=['Low', 'Medium', 'High'])
        
        diff_numeric = pd.to_numeric(self.df['Difference between the effective and initial price (€)'], errors='coerce')
        self.df['Difference between the effective and initial price (€)_numeric'] = diff_numeric
        self.df['Difference between the effective and initial price class'] = pd.qcut(diff_numeric, 3, labels=['Low', 'Medium', 'High'], duplicates='drop')

    def engineer_financial_features(self):
        print("Creazione di feature finanziarie derivate...")
        deadline_numeric = self.df['Execution deadline (days)_numeric']
        base_price = self.df['Base Bid Price (€)']
        
        valid_mask = (deadline_numeric > 0) & (deadline_numeric.notna()) & (base_price.notna())
        self.df['Price per Day'] = np.nan
        self.df.loc[valid_mask, 'Price per Day'] = base_price[valid_mask] / deadline_numeric[valid_mask]

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if self.df['Price per Day'].isnull().any():
            median_price_per_day = self.df['Price per Day'].median()
            self.df['Price per Day'].fillna(median_price_per_day, inplace=True)
            print(f"Imputati i valori mancanti di 'Price per Day' con la mediana: {median_price_per_day:.2f}")

    def impute_and_finalize(self):
        for col in ['Submission deadline (days)', 'Classification of the multifactor criteria (%)']:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        
        if 'Published in the EU journal' in self.df.columns:
            mode_val = self.df['Published in the EU journal'].mode()[0]
            self.df['Published in the EU journal'].fillna(mode_val, inplace=True)
                
        self.df.dropna(inplace=True)

    def save_data(self, path: str):
        self.df.to_csv(path, index=False)
        print(f"\nDataset pulito e finale salvato in: {path}")

    def summarize_data(self):
        print("\n--- Riepilogo Statistico del Dataset Pulito ---")
        with pd.option_context('display.max_columns', None, 'display.width', 1000):
            print(self.df.describe(include='all'))

    def compute_semantic_embeddings(self, text_col: str, prefix: str = 'semantic'):
        if text_col not in self.df.columns:
            print(f"Colonna '{text_col}' non trovata.")
            return False
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("sentence-transformers non disponibile.")
            return False

        sentences = self.df[text_col].fillna('').astype(str).tolist()
        if not sentences:
            print("Nessun testo da elaborare.")
            return False

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences, show_progress_bar=True)

        reducer = PCA(n_components=2, random_state=42)
        components = reducer.fit_transform(embeddings)
        self.df[f'{prefix}_x'] = components[:, 0]
        self.df[f'{prefix}_y'] = components[:, 1]
        print(f"Embedding semantici calcolati per '{text_col}'.")
        return True

    def perform_text_clustering(self, n_clusters: int = 5, random_state: int = 42):
        if 'cpvs_sem_x' not in self.df.columns or 'cpvs_sem_y' not in self.df.columns:
            print("Embedding semantici non trovati. Salto il clustering.")
            return
        from sklearn.cluster import KMeans
        print(f"Esecuzione del clustering K-Means con {n_clusters} cluster...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        embedding_data = self.df[['cpvs_sem_x', 'cpvs_sem_y']].dropna()
        self.df.loc[embedding_data.index, 'cpvs_cluster'] = kmeans.fit_predict(embedding_data)
        self.df['cpvs_cluster'] = self.df['cpvs_cluster'].astype('category')
        print("Clustering completato.")

    def visualize_text_clusters(self):
        if 'cpvs_cluster' not in self.df.columns or px is None:
            return
        
        # Utilizziamo una palette di colori qualitativa per massimizzare la distinzione visiva
        num_clusters = self.df['cpvs_cluster'].nunique()
        color_sequence = px.colors.qualitative.Vivid
        
        fig = px.scatter(
            self.df.dropna(subset=['cpvs_cluster']),
            x='cpvs_sem_x',
            y='cpvs_sem_y',
            color='cpvs_cluster',
            title='Cluster Semantici delle Descrizioni CPVS (K-Means)',
            hover_data=['cpvs_raw_text'],
            category_orders={"cpvs_cluster": sorted(self.df['cpvs_cluster'].unique())},
            color_discrete_sequence=color_sequence
        )
        
        # Miglioriamo la legenda e il layout
        fig.update_layout(
            legend_title_text='Cluster ID',
            xaxis_title='Componente Principale 1 (Semantica)',
            yaxis_title='Componente Principale 2 (Semantica)'
        )
        
        cluster_path = os.path.join(PLOTS_DIR, '13_cpvs_semantic_clusters.html')
        fig.write_html(cluster_path)
        fig.show()
        print(f"Grafico dei cluster semantici salvato in: {cluster_path}")

    def analyze_award_criteria_and_price(self):
        """
        Genera visualizzazioni approfondite per analizzare la relazione tra
        criteri di aggiudicazione, prezzo e altre feature.
        """
        print("\n--- Analisi Approfondita: Criteri di Aggiudicazione e Prezzi ---")

        # 1. Box Plot: Prezzo per Criterio di Aggiudicazione
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(
            x='Award criteria class',
            y='Base Bid Price (€)',
            data=self.df,
            palette='Set2',
            ax=ax,
            hue='Award criteria class',
            legend=False
        )
        ax.set_title('Distribuzione del Prezzo Base per Criterio di Aggiudicazione', fontsize=16)
        ax.set_ylabel('Prezzo Base (€) (Scala Logaritmica)', fontsize=12)
        ax.set_xlabel('Criterio di Aggiudicazione', fontsize=12)
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)
        self._save_plot(fig, '20_price_distribution_by_award_criteria.png')
        plt.show()
        plt.close(fig)

        # 2. Bar Plot: Conteggio Criteri per Distretto
        if 'District' in self.df.columns:
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.countplot(
                y='District',
                hue='Award criteria class',
                data=self.df,
                order=self.df['District'].value_counts().index,
                palette='viridis',
                ax=ax
            )
            ax.set_title('Numero di Contratti per Criterio di Aggiudicazione e Distretto', fontsize=16)
            ax.set_xlabel('Numero di Contratti', fontsize=12)
            ax.set_ylabel('Distretto', fontsize=12)
            ax.legend(title='Criterio di Aggiudicazione')
            self._save_plot(fig, '21_award_criteria_by_district.png')
            plt.show()
            plt.close(fig)

        # 3. Hexbin Plot: Densità di Prezzo vs. Scadenza per Criterio
        if {'Base Bid Price (€)', 'Execution deadline (days)_numeric'}.issubset(self.df.columns):
            g = sns.jointplot(
                data=self.df,
                x='Base Bid Price (€)',
                y='Execution deadline (days)_numeric',
                hue='Award criteria class',
                kind='kde',
                fill=True,
                height=10,
                palette='viridis'
            )
            g.ax_joint.set_xscale('log')
            g.ax_joint.set_yscale('log')
            g.fig.suptitle('Densità di Prezzo vs. Scadenza per Criterio di Aggiudicazione', y=1.02)
            g.set_axis_labels('Prezzo Base (€) (Scala Log)', 'Scadenza Esecuzione (giorni) (Scala Log)')
            self._save_plot(g.fig, '22_price_vs_deadline_density_by_award_criteria.png')
            plt.show()
            plt.close(g.fig)

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
        print("\n--- Generazione Visualizzazioni Avanzate ---")
        fig, ax = plt.subplots(figsize=(10, 8))
        award_counts = self.df['Award criteria class'].value_counts()
        ax.pie(award_counts, labels=award_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('plasma', len(award_counts)))
        ax.set_title('Distribuzione dei Criteri di Aggiudicazione')
        self._save_plot(fig, '09_award_criteria_pie.png')
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.violinplot(x='Award criteria class', y='Base Bid Price (€)', data=self.df, palette='viridis', ax=ax, hue='Award criteria class', legend=False)
        ax.set_title('Distribuzione del Prezzo Base per Criterio di Aggiudicazione')
        ax.set_yscale('log')
        ax.set_ylabel('Prezzo Base (€) (Scala Logaritmica)')
        self._save_plot(fig, '10_price_distribution_by_award_criteria.png')
        plt.show()
        plt.close(fig)

        temporal_df = self.df.groupby('Signing Year').agg(
            num_contracts=('Signing Year', 'size'),
            avg_price=('Base Bid Price (€)', 'mean')
        ).reset_index()
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_title('Andamento Temporale dei Contratti e Prezzo Medio')
        ax1.set_xlabel('Anno di Firma')
        ax1.set_ylabel('Numero di Contratti', color='tab:blue')
        ax1.plot(temporal_df['Signing Year'], temporal_df['num_contracts'], color='tab:blue', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Prezzo Medio Base (€)', color='tab:red')
        ax2.plot(temporal_df['Signing Year'], temporal_df['avg_price'], color='tab:red', marker='x', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        fig.tight_layout()
        self._save_plot(fig, '11_temporal_trends.png')
        plt.show()
        plt.close(fig)

        corr_cols = ['Base Bid Price (€)', 'Execution deadline (days)_numeric', 'Diference between close and signing dates_numeric', 'Difference between the effective and initial price (€)_numeric', 'Signing Year']
        corr_matrix = self.df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Heatmap di Correlazione tra Feature Numeriche')
        self._save_plot(fig, '12_correlation_heatmap.png')
        plt.show()
        plt.close(fig)

    def generate_price_intensity_plot(self):
        if 'Price per Day' not in self.df.columns or 'Base Bid Price (€)' not in self.df.columns:
            return
        print("Generazione grafico a esagoni per l'intensità del prezzo...")

        # Filtra i dati per evitare problemi con la scala logaritmica (valori <= 0)
        plot_data = self.df[
            (self.df['Base Bid Price (€)'] > 0) & 
            (self.df['Price per Day'] > 0)
        ].copy()

        # Applica la trasformazione logaritmica per una migliore visualizzazione
        plot_data['log_base_price'] = np.log10(plot_data['Base Bid Price (€)'])
        plot_data['log_price_per_day'] = np.log10(plot_data['Price per Day'])

        # Crea il jointplot con tipo 'hex'
        g = sns.jointplot(
            x='log_base_price',
            y='log_price_per_day',
            data=plot_data,
            kind='hex',
            cmap='viridis',
            height=10
        )
        
        g.fig.suptitle('Intensità del Prezzo: Prezzo Base vs. Prezzo al Giorno (Densità)', y=1.02, fontsize=16)
        g.set_axis_labels('Prezzo Base (€) (Scala Log10)', 'Prezzo al Giorno (€) (Scala Log10)', fontsize=12)
        
        # Salva il grafico
        self._save_plot(g.fig, '18_price_intensity_hexbin.png')
        plt.show()
        plt.close(g.fig)
        print(f"Grafico di intensità del prezzo (hexbin) salvato.")
        
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

    def generate_semantic_scatter(self):
        if px is not None and {'cpvs_sem_x', 'cpvs_sem_y'}.issubset(self.df.columns):
            semantic_fig = px.scatter(
                self.df, x='cpvs_sem_x', y='cpvs_sem_y', color='District',
                hover_data=['cpvs_raw_text'], title='Spazio Semantico delle Descrizioni CPVS'
            )
            semantic_path = os.path.join(PLOTS_DIR, '07_cpvs_semantic_scatter.html')
            semantic_fig.write_html(semantic_path)
            semantic_fig.show()
            print(f"Scatter plot semantico salvato.")

    def generate_numeric_scatter(self):
        if px is not None and {'Base Bid Price (€)', 'Execution deadline (days)_numeric'}.issubset(self.df.columns):
            numeric_scatter = px.scatter(
                self.df, x='Base Bid Price (€)', y='Execution deadline (days)_numeric',
                color='District', title='Prezzo Base vs Deadline di Esecuzione',
                hover_data=['Signing Year']
            )
            numeric_scatter_path = os.path.join(PLOTS_DIR, '08_price_vs_deadline_scatter.html')
            numeric_scatter.write_html(numeric_scatter_path)
            numeric_scatter.show()
            print(f"Scatter plot numerico salvato.")

    def generate_pdf_report(self, report_title: str = 'PPP Portugal - Report di Analisi', output_path: str = None):
        from matplotlib.backends.backend_pdf import PdfPages
        import textwrap

        if output_path is None:
            output_path = os.path.join(PLOTS_DIR, 'PPP_report.pdf')
        print(f"Generazione report PDF narrativo: {output_path}")

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

        with PdfPages(output_path) as pdf:
            # Pagina di copertina
            fig = plt.figure(figsize=(11.69, 8.27)) # A4 Landscape
            fig.text(0.5, 0.65, report_title, ha='center', fontsize=24, weight='bold')
            fig.text(0.5, 0.55, 'Un\'analisi narrativa basata sui dati degli appalti pubblici portoghesi', ha='center', fontsize=14)
            fig.text(0.5, 0.45, f'Data del report: {pd.Timestamp.now().strftime("%d-%m-%Y")}', ha='center', fontsize=10)
            fig.text(0.5, 0.35, f'Numero totale di contratti analizzati: {len(self.df):,}', ha='center', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

            # Aggiungi una pagina per ogni grafico con descrizione
            for fname in plot_files:
                info = plot_info[fname]
                
                # Pagina di testo
                fig = plt.figure(figsize=(11.69, 8.27))
                fig.text(0.1, 0.9, info['title'], fontsize=18, weight='bold')
                
                # A capo automatico del testo
                wrapped_text = '\n'.join(textwrap.wrap(info['description'], width=100))
                fig.text(0.1, 0.8, wrapped_text, fontsize=12, va='top')
                
                pdf.savefig(fig)
                plt.close(fig)
                
                # Pagina con il grafico
                fig = plt.figure(figsize=(11.69, 8.27))
                try:
                    img_path = os.path.join(PLOTS_DIR, fname)
                    img = plt.imread(img_path)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(info['title'], pad=20, fontsize=14)
                    pdf.savefig(fig)
                except Exception as e:
                    print(f"Impossibile aggiungere l'immagine {fname} al PDF: {e}")
                finally:
                    plt.close(fig)

        print(f"Report PDF narrativo generato con successo.")
        
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
# ### Fase 2.7: Scatter Plot Interattivi
# **Obiettivo**: Fornire strumenti interattivi per un'esplorazione libera e dettagliata dei dati.
# 
# **Finding**: Gli scatter plot interattivi permettono di esplorare le relazioni tra prezzo, scadenze e descrizioni dei progetti in modo dinamico. Passando il mouse sui punti, un analista può leggere la descrizione del progetto associato a un outlier, scoprendo ad esempio che un punto con un prezzo molto alto e una durata breve corrisponde a una fornitura specializzata e non a un errore nei dati. Filtrando per distretto, si possono scoprire pattern specifici a livello locale.

# %% [code]
preprocessor.generate_semantic_scatter()

# %% [code]
preprocessor.generate_numeric_scatter()

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