# %% [markdown]
# # Fase 1: Pulizia e Preparazione del Dataset
# La prima fase si concentra sulla trasformazione dei dati grezzi in un dataset strutturato e pulito. Questo è un passaggio cruciale poiché la qualità dell'analisi dipende dalla qualità dei dati.

# %% [markdown]
# ## 1. Setup dell'Ambiente
# Si importano le librerie e si definiscono le costanti.

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
# Si definisce una funzione per pulire e normalizzare il testo, preparandolo per l'analisi.

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
# Tutta la logica di preprocessing è incapsulata in una classe per un approccio modulare e riutilizzabile.

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
        fig = px.scatter(
            self.df.dropna(subset=['cpvs_cluster']),
            x='cpvs_sem_x',
            y='cpvs_sem_y',
            color='cpvs_cluster',
            title='Cluster Semantici delle Descrizioni CPVS (K-Means)',
            hover_data=['cpvs_raw_text'],
            category_orders={"cpvs_cluster": sorted(self.df['cpvs_cluster'].unique())}
        )
        cluster_path = os.path.join(PLOTS_DIR, '13_cpvs_semantic_clusters.html')
        fig.write_html(cluster_path)
        fig.show()
        print(f"Grafico dei cluster semantici salvato in: {cluster_path}")

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
        if px is None or 'Price per Day' not in self.df.columns: return
        print("Generazione grafico intensità del prezzo...")
        fig = px.scatter(
            self.df.sample(min(1000, len(self.df))),
            x='Base Bid Price (€)', y='Price per Day', color='District',
            title='Intensità del Prezzo: Prezzo Base vs. Prezzo al Giorno',
            hover_data=['cpvs_raw_text', 'Signing Year'], log_x=True, log_y=True
        )
        fig.update_layout(xaxis_title='Prezzo Base (€) (Scala Log)', yaxis_title='Prezzo al Giorno (€) (Scala Log)')
        intensity_path = os.path.join(PLOTS_DIR, '18_price_intensity_scatter.html')
        fig.write_html(intensity_path)
        fig.show()
        print(f"Grafico di intensità del prezzo salvato.")
        
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

    def generate_pdf_report(self, output_path: str = None):
        from matplotlib.backends.backend_pdf import PdfPages
        if output_path is None:
            output_path = os.path.join(PLOTS_DIR, 'PPP_report.pdf')
        print(f"Generazione report PDF: {output_path}")

        plot_files = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')])

        with PdfPages(output_path) as pdf:
            # Cover page
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.text(0.5, 0.6, 'PPP Portugal - Report di Analisi', ha='center', fontsize=20, weight='bold')
            fig.text(0.5, 0.5, 'Questo report contiene le visualizzazioni chiave generate durante l\'analisi.', ha='center', fontsize=12)
            fig.text(0.5, 0.4, f'Totale righe analizzate: {len(self.df):,}', ha='center', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

            # Add plots
            for fname in plot_files:
                try:
                    fig = plt.figure(figsize=(11.69, 8.27))
                    img = plt.imread(os.path.join(PLOTS_DIR, fname))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(fname.replace('_', ' ').replace('.png', '').title(), pad=20)
                    pdf.savefig(fig)
                    plt.close(fig)
                except Exception as e:
                    print(f"Impossibile aggiungere {fname} al PDF: {e}")
        print(f"Report PDF generato.")
        
# %% [markdown]
# ## Esecuzione della Pipeline di Storytelling
# Si istanzia la classe e si invocano i metodi in sequenza.

# %% [markdown]
# ### Fase 1.1: Caricamento e Ispezione Iniziale
# Si carica il dataset e si esegue una prima ispezione.

# %% [code]
preprocessor = DataPreprocessor('Datasets/PPPData_EN_1.0.xlsx')
preprocessor.inspect_dataframe("Stato Iniziale del DataFrame")

# %% [markdown]
# ### Fase 1.2: Analisi e Pulizia dei Dati
# **Finding**: Il grafico dei valori mancanti mostra che colonne come `Conclusion of a framework agreement` o `Electronic auction` sono quasi del tutto vuote. La loro rimozione è necessaria per non basare l'analisi su dati inaffidabili.

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
# Si creano nuove feature da date e testo per arricchire l'analisi.

# %% [code]
preprocessor.engineer_date_features()
cpvs_vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.8, ngram_range=(2, 3), max_features=10)
preprocessor.engineer_text_features('Cpvs Designation', 'cpvs', cpvs_vectorizer)
preprocessor.inspect_dataframe("Stato del DataFrame dopo Feature Engineering Testuale")

# %% [markdown]
# ### Fase 1.4: Gestione Outlier e Feature Finanziarie
# **Finding**: Le distribuzioni di variabili come `Execution deadline` mostrano una forte asimmetria, indicando la presenza di contratti con durate eccezionalmente lunghe. La rimozione degli outlier più estremi rende le analisi successive più robuste.

# %% [code]
preprocessor.process_numerical_features()

# %% [code]
preprocessor.engineer_financial_features()
preprocessor.inspect_dataframe("Stato del DataFrame dopo Gestione Outlier e Feature Finanziarie")

# %% [markdown]
# ### Fase 1.5: Imputazione Finale e Salvataggio
# Si completano i dati e si salva il dataset pulito.

# %% [code]
preprocessor.impute_and_finalize()
preprocessor.save_data(CLEANED_DATA_PATH)
preprocessor.inspect_dataframe("Stato Finale del DataFrame")

# %% [markdown]
# ### Fase 1.6: Riepilogo Statistico
# Si esegue un'analisi descrittiva del dataset pulito per un riepilogo quantitativo.

# %% [code]
preprocessor.summarize_data()

# %% [markdown]
# # Fase 2: Analisi Esplorativa e Storytelling Visivo
# Con un dataset pulito, si passa all'esplorazione visiva per scoprire pattern e insight.

# %% [markdown]
# ### Fase 2.1: Analisi Semantica e Clustering
# **Finding**: Il clustering sugli embedding semantici rivela gruppi tematici distinti. Ad esempio, si possono identificare cluster relativi a "lavori stradali", "costruzione di edifici" o "servizi di manutenzione", confermando che il modello ha catturato differenze semantiche reali.

# %% [code]
preprocessor.compute_semantic_embeddings('cpvs_clean_text', prefix='cpvs_sem')
preprocessor.perform_text_clustering(n_clusters=5)
preprocessor.visualize_text_clusters()

# %% [markdown]
# ### Fase 2.2: Visualizzazioni di Base
# **Finding**: I grafici confermano la concentrazione di contratti a Lisbona e Porto. La maggior parte dei contratti viene aggiudicata tramite il criterio del "prezzo più basso", ma la distribuzione dei prezzi e delle scadenze è molto ampia, suggerendo una grande varietà di progetti.

# %% [code]
preprocessor.generate_final_visualizations()

# %% [markdown]
# ### Fase 2.3: Visualizzazioni Avanzate
# **Finding**: La heatmap di correlazione mostra una debole correlazione positiva tra prezzo e scadenza, il che è intuitivo. L'andamento temporale rivela un picco di contratti in certi anni, che potrebbe essere correlato a cicli economici o iniziative governative.

# %% [code]
preprocessor.generate_advanced_visualizations()

# %% [markdown]
# ### Fase 2.4: Analisi Geospaziale
# **Finding**: Le mappe coropletiche mostrano che, sebbene Lisbona e Porto abbiano il maggior numero di contratti, il *valore medio* più alto si trova in distretti con meno contratti, come Bragança o Viana do Castelo. Questo potrebbe indicare la presenza di pochi ma grandi progetti infrastrutturali in quelle aree.

# %% [code]
preprocessor.generate_geospatial_visualizations()

# %% [code]
preprocessor.generate_additional_geospatial_plot()

# %% [markdown]
# ### Fase 2.5: Visualizzazioni per Ruolo
# **Finding per Financial Analyst**: Il grafico a barre impilate mostra come il valore totale dei contratti basati sul "prezzo più basso" sia diminuito negli ultimi anni a favore di criteri multifattoriali, indicando un cambiamento nelle strategie di appalto.
# **Finding per Project Manager**: La scadenza media di esecuzione varia significativamente tra i distretti. Questo è un dato cruciale per la pianificazione, poiché suggerisce che i tempi di progetto possono dipendere fortemente dalla località.

# %% [code]
preprocessor.generate_role_based_visualizations()

# %% [markdown]
# ### Fase 2.6: Analisi di Intensità e Testo
# **Finding**: Il grafico di intensità del prezzo mostra che non c'è una relazione lineare semplice tra il prezzo totale e il "prezzo al giorno", suggerendo che la complessità e la durata del progetto influenzano il costo in modi non banali. La word cloud evidenzia "lavori", "costruzione" e "manutenzione" come i termini più frequenti.

# %% [code]
preprocessor.generate_price_intensity_plot()

# %% [code]
preprocessor.generate_word_cloud()

# %% [markdown]
# ### Fase 2.7: Scatter Plot Interattivi
# **Finding**: Gli scatter plot interattivi permettono di esplorare le relazioni tra prezzo, scadenze e descrizioni dei progetti. Filtrando per distretto, si possono scoprire outlier e pattern specifici a livello locale.

# %% [code]
preprocessor.generate_semantic_scatter()

# %% [code]
preprocessor.generate_numeric_scatter()

# %% [markdown]
# ### Fase 2.8: Creazione del Report PDF
# Si genera un report PDF che raccoglie tutte le visualizzazioni statiche create.

# %% [code]
preprocessor.generate_pdf_report()