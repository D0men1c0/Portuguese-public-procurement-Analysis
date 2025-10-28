# %% [markdown]
# # Storytelling e Preprocessing del Dataset sugli Appalti Pubblici
# Questo script Python, strutturato come un notebook, racconta la storia della pulizia e della preparazione di un dataset sugli appalti pubblici portoghesi.
# L'obiettivo è trasformare dati grezzi in un formato pulito e significativo, pronto per essere esplorato con una dashboard interattiva (es. Streamlit).
# Ogni passaggio è documentato seguendo i principi di **Visual Analytics** e **Information Seeking Mantra** (Overview first, Zoom and Filter, then Details-on-Demand).

# %% [markdown]
# ## 1. Setup dell'Ambiente
# Importiamo le librerie necessarie e definiamo le costanti.

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import spacy
import os
import sys
import json
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
# ## 1.5. Funzione di Preprocessing Testuale
# Per estrarre insight dai dati testuali (es. CPVS), serve una funzione che pulisca e normalizzi il testo, riducendo rumore e ridondanza.
# Questo è fondamentale per creare variabili utili a visualizzazioni come Word Cloud, filtri nominali e analisi di correlazione.

# %% [code]
def preprocess_text(text: str) -> str:
    """
    Pulisce una stringa di testo per l'analisi NLP:
    - Rimuove punteggiatura e numeri, converte in minuscolo.
    - Rimuove le stopword.
    - Esegue la lemmatizzazione.
    Restituisce il testo pulito, pronto per estrazione keyword e feature engineering.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(f'[{re.escape(string.punctuation)}0-9]', '', text.lower())
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.strip()]
    return " ".join(lemmas)

# %% [markdown]
# ## 2. La Classe `DataPreprocessor`: Un Approccio Modulare
# Per evitare "spaghetti code" e rendere il processo robusto e riutilizzabile, incapsuliamo tutta la logica in una classe. Ogni metodo rappresenta un passo logico della pipeline di preprocessing.

# %% [code]
class DataPreprocessor:
    """Una classe per orchestrare l\'intero processo di pulizia e preparazione dei dati."""
    
    def __init__(self, file_path: str):
        """Inizializza il preprocessor caricando i dati."""
        self.df = self._load_data(file_path)

    def _load_data(self, path: str) -> pd.DataFrame:
        """Metodo privato per caricare il dataset da un file Excel."""
        print(f"Caricamento dati da: {path}")
        return pd.read_excel(path)

    def _save_plot(self, figure: plt.Figure, filename: str):
        """Salva una figura nella cartella dei plot."""
        path = os.path.join(PLOTS_DIR, filename)
        figure.savefig(path, bbox_inches='tight')
        print(f"Grafico salvato in: {path}")

    def inspect_dataframe(self, title: str):
        """Mostra informazioni chiave sullo stato attuale del DataFrame."""
        print(f"\n--- {title} ---")
        print("\n1. Informazioni Generali e Memoria:")
        self.df.info()
        print("\n2. Valori Nulli per Colonna:")
        print(self.df.isnull().sum().sort_values(ascending=False).head())
        print("\n3. Prime 5 Righe:")
        print(self.df.head())

        def launch_streamlit_app(self, geojson_path: str = 'Datasets/portugal_districts.geojson'):
            """Avvia un'app Streamlit con visualizzazioni interattive basate sul dataframe preprocessato."""
            try:
                import streamlit as st  # type: ignore[import-not-found]
            except ImportError:
                print("Streamlit non disponibile: esegui 'pip install streamlit plotly' per l'app interattiva.")
                return

            if px is None:
                st.warning("Plotly non installato: installa 'plotly' per le visualizzazioni interattive.")
                return

            st.set_page_config(page_title="PPP Portugal Analytics", layout="wide")
            st.title("Dashboard Interattiva sugli Appalti Pubblici Portoghesi")

            df_local = self.df.copy()

            with st.sidebar:
                st.header("Filtri")
                districts = sorted(df_local['District'].dropna().unique())
                selected_districts = st.multiselect("Seleziona i distretti", districts)

                year_min, year_max = int(df_local['Signing Year'].min()), int(df_local['Signing Year'].max())
                year_range = st.slider("Intervallo anni", min_value=year_min, max_value=year_max, value=(year_min, year_max), step=1)

                if 'Base Bid Price (€)_numeric' in df_local.columns:
                    price_min = float(df_local['Base Bid Price (€)_numeric'].min())
                    price_max = float(df_local['Base Bid Price (€)_numeric'].max())
                    if price_min == price_max:
                        st.info("Prezzo base uniforme nel dataset: filtro disabilitato.")
                        price_range = None
                    else:
                        price_range = st.slider("Prezzo base (€)", min_value=price_min, max_value=price_max, value=(price_min, price_max))
                else:
                    price_range = None

                keyword = st.text_input("Filtra per keyword CPVS", "")

            filtered = df_local[
                (df_local['Signing Year'] >= year_range[0]) &
                (df_local['Signing Year'] <= year_range[1])
            ]

            if selected_districts:
                filtered = filtered[filtered['District'].isin(selected_districts)]

            if price_range and 'Base Bid Price (€)_numeric' in filtered.columns:
                filtered = filtered[
                    (filtered['Base Bid Price (€)_numeric'] >= price_range[0]) &
                    (filtered['Base Bid Price (€)_numeric'] <= price_range[1])
                ]

            if keyword and 'cpvs_clean_text' in filtered.columns:
                filtered = filtered[filtered['cpvs_clean_text'].str.contains(keyword, case=False, na=False)]

            st.subheader("Contratti per Distretto")
            district_counts = filtered.groupby('District').size().reset_index(name='Contracts')
            bar_fig = px.bar(district_counts, x='District', y='Contracts', color='District', title='Numero Contratti per Distretto')
            st.plotly_chart(bar_fig, use_container_width=True)

            st.subheader("Distribuzione del Prezzo Base")
            if 'Base Bid Price (€)_numeric' in filtered.columns:
                hist_fig = px.histogram(filtered, x='Base Bid Price (€)_numeric', nbins=30, color='District', title='Distribuzione Prezzo Base (€)')
                st.plotly_chart(hist_fig, use_container_width=True)
            else:
                st.info("Prezzo base numerico non disponibile per l'istogramma.")

            if 'cpvs_sem_x' in filtered.columns:
                st.subheader("Mappa Semantica delle Keyword CPVS")
                sem_fig = px.scatter(
                    filtered,
                    x='cpvs_sem_x',
                    y='cpvs_sem_y',
                    color='District',
                    hover_data=['cpvs_raw_text'],
                    title='Embedding Semantici (Sentence-BERT + PCA)'
                )
                st.plotly_chart(sem_fig, use_container_width=True)

            geojson_file = Path(geojson_path)
            if geojson_file.exists() and 'Base Bid Price (€)_numeric' in filtered.columns:
                with geojson_file.open('r', encoding='utf-8') as f:
                    geojson_data = json.load(f)
                geo_metrics = filtered.groupby('District').agg(
                    contracts=('District', 'count'),
                    base_bid_mean=('Base Bid Price (€)_numeric', 'mean')
                ).reset_index()
                if not geo_metrics.empty:
                    map_fig = px.choropleth_mapbox(
                        geo_metrics,
                        geojson=geojson_data,
                        locations='District',
                        featureidkey='properties.name',
                        color='base_bid_mean',
                        color_continuous_scale='Plasma',
                        mapbox_style='carto-positron',
                        zoom=5.5,
                        center={'lat': 39.5, 'lon': -8.0},
                        opacity=0.7,
                        hover_data={'contracts': True, 'base_bid_mean': ':.0f'}
                    )
                    map_fig.update_layout(title='Prezzo Medio Base per Distretto (Filtrato)')
                    st.subheader("Mappa Coropletica (Interattiva)")
                    st.plotly_chart(map_fig, use_container_width=True)
            else:
                st.info("Aggiungi un file GeoJSON dei distretti per visualizzare la mappa coropletica.")

            if WordCloud and not filtered.empty and 'cpvs_clean_text' in filtered.columns:
                st.subheader("Word Cloud Dinamica")
                cloud_text = ' '.join(filtered['cpvs_clean_text'])
                if cloud_text.strip():
                    cloud = WordCloud(width=800, height=400, background_color='white').generate(cloud_text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(cloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Nessuna keyword disponibile con i filtri correnti.")
            elif WordCloud is None:
                st.info("Installa 'wordcloud' per visualizzare la Word Cloud dinamica.")

            csv_export = filtered.to_csv(index=False).encode('utf-8')
            st.download_button("Scarica il dataset filtrato", data=csv_export, file_name='PPP_filtered.csv', mime='text/csv')

    def analyze_missing_values(self):
        """Analizza e visualizza i valori mancanti come primo passo della quality assessment."""
        missing_percentage = self.df.isnull().sum() * 100 / len(self.df)
        missing_df = pd.DataFrame({'column_name': self.df.columns, 'percent_missing': missing_percentage})
        missing_df.sort_values('percent_missing', inplace=True, ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='percent_missing', y='column_name', data=missing_df[missing_df['percent_missing'] > 0], ax=ax)
        ax.set_title('Percentuale di Valori Mancanti per Colonna')
        self._save_plot(fig, '01_missing_values_percentage.png')
        plt.show()

    def prune_columns(self, columns_to_drop: List[str]):
        """Rimuove le colonne con troppi NaN, ridondanti o non informative."""
        print(f"Rimuovendo {len(columns_to_drop)} colonne...")
        self.df.drop(columns=columns_to_drop, inplace=True)

    def clean_and_correct(self):
        """Esegue la pulizia di base, la correzione di tipi e la gestione di anomalie."""
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
        """Crea feature da colonne data per analisi di serie temporali (Line Chart)."""
        for date_col in ['Signing date', 'Closing date']:
            self.df[date_col] = pd.to_datetime(self.df[date_col], format='%d-%m-%Y')
        self.df['Signing Year'] = self.df['Signing date'].dt.year
        self.df['Signing Month'] = self.df['Signing date'].dt.month
        self.df.drop(columns=['Signing date', 'Closing date'], inplace=True)

    def engineer_text_features(self, text_col: str, new_col_prefix: str, vectorizer: TfidfVectorizer):
        """Funzione riutilizzabile per estrarre keyword da una colonna di testo usando TF-IDF.

        Args:
            text_col: La colonna di testo da processare.
            new_col_prefix: Prefisso per le nuove colonne create.
            vectorizer: L'istanza di TfidfVectorizer da usare.
        """
        cleaned_col = f"{text_col}_cleaned"
        raw_store_col = f"{new_col_prefix}_raw_text"
        clean_store_col = f"{new_col_prefix}_clean_text"

        self.df[cleaned_col] = self.df[text_col].apply(preprocess_text)
        self.df[raw_store_col] = self.df[text_col].astype(str)
        self.df[clean_store_col] = self.df[cleaned_col]

        preview_df = self.df[[text_col]].copy()
        preview_df[clean_store_col] = self.df[cleaned_col]
        print(f"Esempio di pulizia per '{text_col}':")
        print(preview_df.head(2))

        X_tfidf = vectorizer.fit_transform(self.df[cleaned_col])
        keywords = vectorizer.get_feature_names_out()

        for keyword in keywords:
            self.df[f"{new_col_prefix}_{keyword.replace(' ', '_')}"] = self.df[cleaned_col].apply(lambda x: 1 if keyword in x else 0)

        self.df.drop(columns=[text_col], inplace=True)
        print(f"Create {len(keywords)} feature da '{text_col}'.")

    def process_numerical_features(self):
        """Gestisce outlier e discretizza le colonne numeriche."""
        numerical_cols = {
            'Diference between close and signing dates': ['Short', 'Medium', 'Long'],
            'Execution deadline (days)': ['Short', 'Medium', 'Long']
        }

        for col, labels in numerical_cols.items():
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f'Analisi Distribuzione: {col}', fontsize=14)
            sns.histplot(self.df[col], kde=True, ax=axes[0])
            axes[0].set_title('Istogramma')
            sns.boxplot(x=self.df[col], ax=axes[1])
            axes[1].set_title('Box Plot')
            self._save_plot(fig, f'02_distribution_{col.replace(" ", "_")}.png')
            plt.show()

            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 2.5 * IQR, Q3 + 2.5 * IQR
            
            initial_rows = len(self.df)
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
            print(f"Rimosse {initial_rows - len(self.df)} righe di outlier per '{col}'.")
            
            self.df[f'{col}_numeric'] = self.df[col]
            self.df[col] = pd.qcut(self.df[col], len(labels), labels=labels, duplicates='drop')

        self.df['Base Bid Price (€)_numeric'] = pd.to_numeric(self.df['Base Bid Price (€)'], errors='coerce')
        self.df['Base Bid Price (€)'] = pd.qcut(self.df['Base Bid Price (€)_numeric'], 3, labels=['Low', 'Medium', 'High'])
        diff_numeric = pd.to_numeric(self.df['Difference between the effective and initial price (€)'], errors='coerce')
        self.df['Difference between the effective and initial price (€)_numeric'] = diff_numeric
        self.df['Difference between the effective and initial price class'] = pd.qcut(diff_numeric, 3, labels=['Low', 'Medium', 'High'], duplicates='drop')

    def impute_and_finalize(self):
        """Esegue l\'imputazione finale e le ultime pulizie."""
        for col in ['Submission deadline (days)', 'Classification of the multifactor criteria (%)']:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
        
        if 'Published in the EU journal' in self.df.columns:
            mode_val = self.df['Published in the EU journal'].mode()[0]
            self.df['Published in the EU journal'].fillna(mode_val, inplace=True)
                
        self.df.dropna(inplace=True)

    def save_data(self, path: str):
        """Salva il DataFrame pulito in un file CSV."""
        self.df.to_csv(path, index=False)
        print(f"\nDataset pulito e finale salvato in: {path}")

    def generate_final_visualizations(self):
        """Genera e salva i grafici di riepilogo del dataset pulito."""
        print("\n--- Generazione Visualizzazioni Finali ---")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.countplot(y='District', data=self.df, order=self.df['District'].value_counts().index, palette='viridis', ax=ax)
        ax.set_title('Numero di Contratti per Distretto (Dato Nominale)')
        self._save_plot(fig, '03_district_distribution.png')
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribuzioni di Feature Chiave', fontsize=16)
        
        plot_specs = [
            {'col': 'Award criteria class', 'ax': axes[0, 0], 'title': 'Classe Criteri di Aggiudicazione'},
            {'col': 'Base Bid Price (€)', 'ax': axes[0, 1], 'title': 'Prezzo Base (Discretizzato)'},
            {'col': 'Execution deadline (days)', 'ax': axes[1, 0], 'title': 'Scadenza Esecuzione (Discretizzato)'},
            {'col': 'Diference between close and signing dates', 'ax': axes[1, 1], 'title': 'Differenza Date (Discretizzata)'}
        ]
        
        for spec in plot_specs:
            sns.countplot(x=spec['col'], data=self.df, palette='magma', ax=spec['ax'])
            spec['ax'].set_title(spec['title'])
            spec['ax'].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_plot(fig, '04_key_features_distribution.png')
        plt.show()

    def generate_geospatial_visualizations(self, geojson_path: str = 'Datasets/portugal_districts.geojson'):
        """Crea una mappa coropletica dei distretti se è disponibile un file GeoJSON compatibile."""
        if px is None:
            print("Plotly non disponibile: installa 'plotly' per le visualizzazioni geografiche.")
            return

        geojson_file = Path(geojson_path)
        if not geojson_file.exists():
            print(
                "GeoJSON non trovato: scarica un file dei distretti portoghesi in 'Datasets/portugal_districts.geojson'.\n"
                "Ad esempio: https://raw.githubusercontent.com/ppinheiroalmeida/portugal-geojson/master/portugal-districts.geojson"
            )
            return

        with geojson_file.open('r', encoding='utf-8') as f:
            districts_geojson = json.load(f)

        metrics = self.df.groupby('District').agg(
            contracts=('District', 'count'),
            base_bid_mean=('Base Bid Price (€)_numeric', 'mean'),
            execution_deadline_mode=('Execution deadline (days)', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        ).reset_index()

        fig_map = px.choropleth_mapbox(
            metrics,
            geojson=districts_geojson,
            locations='District',
            featureidkey='properties.name',
            color='base_bid_mean',
            color_continuous_scale='Viridis',
            mapbox_style='carto-positron',
            zoom=5.5,
            center={'lat': 39.5, 'lon': -8.0},
            opacity=0.7,
            hover_data={'contracts': True, 'base_bid_mean': ':.0f'}
        )
        fig_map.update_layout(title='Valore Medio del Prezzo Base per Distretto')
        fig_map.write_html(os.path.join(PLOTS_DIR, '05_map_base_bid_by_district.html'))
        print("Mappa coropletica salvata come HTML interattivo (05_map_base_bid_by_district.html).")

    def compute_semantic_embeddings(self, text_col: str, prefix: str = 'semantic'):
        """Calcola embedding Sentence-BERT su una colonna testuale e riduce a 2 dimensioni."""
        if text_col not in self.df.columns:
            print(f"Colonna '{text_col}' non trovata: salta embedding semantici.")
            return False

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("sentence-transformers non disponibile: installa 'sentence-transformers' per embeddings semantici.")
            return False

        sentences = self.df[text_col].fillna('').astype(str).tolist()
        if not sentences:
            print("Nessun testo da elaborare per embeddings.")
            return False

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(sentences, show_progress_bar=True)

        reducer = PCA(n_components=2, random_state=42)
        components = reducer.fit_transform(embeddings)
        self.df[f'{prefix}_x'] = components[:, 0]
        self.df[f'{prefix}_y'] = components[:, 1]
        print(f"Embedding Sentence-BERT calcolati e ridotti a 2 componenti per '{text_col}'.")
        return True

# %% [markdown]
# ## Inizio della Pipeline di Storytelling
# Ora istanziamo la nostra classe e invochiamo i metodi in sequenza, ispezionando il risultato ad ogni passo.

# %% [markdown]
# ### Fase 1: Caricamento e Visione d'Insieme
# Applichiamo il mantra "Overview first" per una prima comprensione del dataset grezzo.

# %% [code]
preprocessor = DataPreprocessor('Datasets/PPPData_EN_1.0.xlsx')
preprocessor.inspect_dataframe("Stato Iniziale del DataFrame")

# %% [markdown]
# ### Fase 2: Analisi Qualità Dati e Pulizia Colonne
# Identifichiamo le colonne problematiche tramite un\'analisi visuale dei valori mancanti.

# %% [code]
preprocessor.analyze_missing_values()

# %% [markdown]
# **Risultato:** Il grafico mostra diverse colonne con oltre il 50% di dati mancanti, rendendole inaffidabili.
# **Azione:** Procediamo a rimuoverle, insieme a identificatori e testo non utile, per massimizzare il **Data-Ink Ratio**.

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
preprocessor.inspect_dataframe("Stato del DataFrame dopo il Pruning")

# %% [markdown]
# ### Fase 3: Correzione Dati e Rimozione Righe
# Eseguiamo pulizie più fini per garantire la consistenza, fondamentale per creare visualizzazioni corrette (es. **Choropleth Map**).

# %% [code]
preprocessor.clean_and_correct()
preprocessor.inspect_dataframe("Stato del DataFrame dopo la Pulizia Fine")

# %% [markdown]
# ### Fase 4: Feature Engineering
# Creiamo nuove variabili per arricchire il dataset e abilitare nuove forme di analisi.

# %% [code]
preprocessor.engineer_date_features()

# %% [code]
# Definiamo i parametri per l\'estrazione di keyword da CPVS
cpvs_vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.8, ngram_range=(2, 3), max_features=10)
preprocessor.engineer_text_features('Cpvs Designation', 'cpvs', cpvs_vectorizer)

# %% [code]
preprocessor.inspect_dataframe("Stato del DataFrame dopo la Feature Engineering")

# %% [code]
preprocessor.compute_semantic_embeddings('cpvs_clean_text', prefix='cpvs_sem')

# %% [markdown]
# ### Fase 5: Gestione Outlier e Discretizzazione
# Analizziamo e trattiamo le variabili numeriche per migliorarne la rappresentazione ed evitare **deceptive visualizations**.

# %% [code]
preprocessor.process_numerical_features()
preprocessor.inspect_dataframe("Stato del DataFrame dopo la Gestione degli Outlier")

# %% [markdown]
# ### Fase 6: Imputazione Finale
# Completiamo il dataset riempiendo gli ultimi valori mancanti per avere una base dati solida.

# %% [code]
preprocessor.impute_and_finalize()
preprocessor.inspect_dataframe("Stato Finale del DataFrame (Nessun Valore Nullo)")

# %% [markdown]
# ### Fase 7: Salvataggio del Dataset Pulito
# Il nostro processo di trasformazione è completo. Salviamo il risultato.

# %% [code]
preprocessor.save_data(CLEANED_DATA_PATH)

# %% [markdown]
# ### Fase 8: Visualizzazioni di Riepilogo
# Concludiamo con una "Overview" del dataset finale, pronta per la fase di "Zoom and Filter" in una dashboard.

# %% [code]
preprocessor.generate_final_visualizations()

# %% [markdown]
# ## Sezione Extra: Visualizzazioni Avanzate e Componenti Interattivi
# Estendiamo lo storytelling con grafici interattivi, mappe geografiche e analisi semantiche.

# %% [markdown]
# ### Mappa Coropletica (Plotly Mapbox)
# Rappresentiamo il prezzo medio per distretto su mappa, seguendo il principio "Overview first".

# %% [code]
preprocessor.generate_geospatial_visualizations()

# %% [markdown]
# ### Word Cloud delle Keyword CPVS
# Visualizziamo la frequenza delle keyword estratte via TF-IDF, utile per scoprire pattern testuali.

# %% [code]
if WordCloud and 'cpvs_clean_text' in preprocessor.df.columns:
    text = ' '.join(preprocessor.df['cpvs_clean_text'])
    if text.strip():
        cloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud delle Keyword CPVS')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '06_wordcloud_cpvs.png'))
        plt.show()
    else:
        print("Word Cloud non generata: testo CPVS vuoto dopo la pulizia.")
elif WordCloud is None:
    print("Libreria 'wordcloud' non disponibile: esegui 'pip install wordcloud'.")

# %% [markdown]
# ### Scatter Plot Semantico (Sentence-BERT + PCA)
# Riduciamo gli embedding in 2D per analizzare cluster e correlazioni tra descrizioni CPVS.

# %% [code]
if px is not None and {'cpvs_sem_x', 'cpvs_sem_y'}.issubset(preprocessor.df.columns):
    semantic_fig = px.scatter(
        preprocessor.df,
        x='cpvs_sem_x',
        y='cpvs_sem_y',
        color='District',
        hover_data=['cpvs_raw_text'],
        title='Spazio Semantico delle Descrizioni CPVS'
    )
    semantic_path = os.path.join(PLOTS_DIR, '07_cpvs_semantic_scatter.html')
    semantic_fig.write_html(semantic_path)
    print(f"Scatter plot semantico salvato in formato HTML interattivo ({os.path.basename(semantic_path)}).")
else:
    print("Impossibile generare lo scatter semantico: assicurati di aver installato 'sentence-transformers' e 'plotly'.")

# %% [markdown]
# ### Scatter Plot Numerico (Plotly)
# Esploriamo la relazione tra prezzo base e scadenza d'esecuzione con un grafico interattivo.

# %% [code]
if px is not None and {'Base Bid Price (€)_numeric', 'Execution deadline (days)_numeric'}.issubset(preprocessor.df.columns):
    numeric_scatter = px.scatter(
        preprocessor.df,
        x='Base Bid Price (€)_numeric',
        y='Execution deadline (days)_numeric',
        color='District',
        title='Prezzo Base vs Deadline di Esecuzione',
        hover_data=['Signing Year']
    )
    numeric_scatter_path = os.path.join(PLOTS_DIR, '08_price_vs_deadline_scatter.html')
    numeric_scatter.write_html(numeric_scatter_path)
    print(f"Scatter plot numerico salvato in formato HTML interattivo ({os.path.basename(numeric_scatter_path)}).")
elif px is None:
    print("Plotly non disponibile: installa 'plotly' per gli scatter interattivi.")

# %% [markdown]
# ### Avvio della Dashboard Streamlit (Opzionale)
# Esegui `streamlit run preProcessData_story.py -- --streamlit` per attivare la modalità interattiva con filtri dinamici.

# %% [code]
if '--streamlit' in sys.argv:
    preprocessor.launch_streamlit_app()
