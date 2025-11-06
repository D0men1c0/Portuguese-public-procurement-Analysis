import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
import json
from typing import List, Dict, Tuple, Optional
from PIL import Image
from matplotlib.ticker import FuncFormatter
from wordcloud import WordCloud

# --- Configurazione Pagina e Stile ---

st.set_page_config(
    page_title="Portuguese Procurement Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True

top_col = st.container()
with top_col:
    if st.button("☰", help="Mostra/Nascondi Filtri", use_container_width=False):
        st.session_state.sidebar_visible = not st.session_state.sidebar_visible

# CSS dinamico per nascondere o mostrare la sidebar
st.markdown(f"""
    <style>
        section[data-testid="stSidebar"] {{
            display: {"block" if st.session_state.sidebar_visible else "none"};
            transition: all 0.15s ease-in-out;
        }}
        div.stButton > button:first-child {{
            background-color: #1E3A8A !important;
            color: white !important;
            border-radius: 8px;
            padding: 0.4rem 0.8rem;
            font-weight: 600;
            border: none;
        }}
        div.stButton > button:first-child:hover {{
            background-color: #3B82F6 !important;
        }}
    </style>
""", unsafe_allow_html=True)

# --- Costanti ---

DATA_FILE_PATH: str = 'Datasets/PPPData_EN_cleaned_2.csv'
GEOJSON_PATH_1: str = 'Datasets/portugal_districts.geojson'
GEOJSON_PATH_2: str = 'Datasets/georef-portugal-distrito-millesime.json'
PLOTS_DIR: str = 'plots'
WORDCLOUD_IMG: str = '06_wordcloud_cpvs.png'

# --- Caricamento Dati in Cache ---

@st.cache_data
def load_data(file_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(file_path):
        st.error(f"File non trovato: {file_path}. Eseguire lo script di preprocessing.")
        st.stop()
    
    try:
        df = pd.read_csv(file_path)
        numeric_cols: List[str] = [
            'Base Bid Price (€)', 'Execution deadline (days)_numeric', 
            'Diference between close and signing dates_numeric',
            'Difference between the effective and initial price (€)_numeric',
            'Price per Day', 'Publication Year', 'Publication Month',
            'cpvs_sem_x', 'cpvs_sem_y'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        cat_cols: List[str] = [
            'District', 'Award criteria class', 'Base Bid Price (€)_category',
            'Execution deadline (days)', 'Diference between close and signing dates',
            'cpvs_cluster'
        ]
        for col in cat_cols:
            if col in df.columns:
                if 'cluster' in col and pd.api.types.is_numeric_dtype(df[col]):
                     df[col] = df[col].dropna().astype(float).astype(int).astype(str)
                else:
                    df[col] = df[col].astype('category')
        return df
    except Exception as e:
        st.error(f"Errore durante il caricamento o la conversione dei dati: {e}")
        st.stop()

@st.cache_data
def load_geojson() -> Tuple[Optional[Dict], Optional[str]]:
    path_to_use, key_to_use = None, None
    if os.path.exists(GEOJSON_PATH_1):
        path_to_use, key_to_use = GEOJSON_PATH_1, 'properties.NAME_1'
    elif os.path.exists(GEOJSON_PATH_2):
        path_to_use, key_to_use = GEOJSON_PATH_2, 'properties.dis_name'
    else:
        st.warning("File GeoJSON non trovato in `Datasets/`. Le mappe saranno disabilitate.")
        return None, None
    
    try:
        with open(path_to_use, 'r', encoding='utf-8') as f:
            return json.load(f), key_to_use
    except Exception as e:
        st.error(f"Errore durante il caricamento del file GeoJSON {path_to_use}: {e}")
        return None, None

@st.cache_data
def load_static_image(image_name: str) -> Optional[Image.Image]:
    path = os.path.join(PLOTS_DIR, image_name)
    if os.path.exists(path):
        try:
            return Image.open(path)
        except Exception as e:
            st.error(f"Errore nel caricamento dell'immagine {image_name}: {e}")
    return None

# --- Classi Utility ---

class Utils:
    @staticmethod
    def format_currency(value: float, decimals: int = 0) -> str:
        if pd.isna(value): return "N/A"
        if abs(value) >= 1e9: return f"€{value/1e9:.{decimals}f}B"
        if abs(value) >= 1e6: return f"€{value/1e6:.{decimals}f}M"
        if abs(value) >= 1e3: return f"€{value/1e3:.{decimals}f}K"
        return f"€{value:.{decimals}f}"

# --- Classi della Dashboard ---

class FilterManager:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        st.sidebar.markdown("## Filtri Dashboard")

    def _apply_year_filter(self) -> pd.DataFrame:
        if 'Publication Year' in self.df.columns:
            years = sorted(self.df['Publication Year'].dropna().astype(int).unique())
            if years:
                default_years = years if len(years) <= 3 else years[-3:]
                selected_years = st.sidebar.multiselect("Anno Pubblicazione", options=years, default=default_years)
                if selected_years:
                    return self.df[self.df['Publication Year'].isin(selected_years)]
        return self.df

    def _apply_district_filter(self) -> pd.DataFrame:
        if 'District' in self.df.columns:
            districts = sorted(self.df['District'].dropna().unique())
            selected_districts = st.sidebar.multiselect("Distretto", options=districts, default=[])
            if selected_districts:
                return self.df[self.df['District'].isin(selected_districts)]
        return self.df

    def _apply_criteria_filter(self) -> pd.DataFrame:
        criteria_col = 'Award criteria class'
        if criteria_col in self.df.columns:
            criteria = sorted(self.df[criteria_col].dropna().unique())
            selected_criteria = st.sidebar.multiselect("Criterio Aggiudicazione", options=criteria, default=[])
            if selected_criteria:
                return self.df[self.df[criteria_col].isin(selected_criteria)]
        return self.df

    def _apply_price_filter(self) -> pd.DataFrame:
        price_col = 'Base Bid Price (€)'
        if price_col in self.df.columns:
            min_price, max_price = float(self.df[price_col].min()), float(self.df[price_col].max())
            use_log = (max_price / (min_price + 1e-6)) > 1000 and min_price >= 0

            if use_log:
                min_log, max_log = np.log10(min_price + 1), np.log10(max_price + 1)
                log_range = st.sidebar.slider("Fascia di Prezzo (€) (Log)", min_log, max_log, (min_log, max_log), format="€10^%.1f")
                price_range = (10**log_range[0], 10**log_range[1])
            else:
                price_range = st.sidebar.slider("Fascia di Prezzo (€)", min_price, max_price, (min_price, max_price), format="€%.0f")
            
            return self.df[(self.df[price_col] >= price_range[0]) & (self.df[price_col] <= price_range[1])]
        return self.df

    def apply_filters(self) -> pd.DataFrame:
        self.df = self._apply_year_filter()
        self.df = self._apply_district_filter()
        self.df = self._apply_criteria_filter()
        self.df = self._apply_price_filter()
        return self.df


class ExecutiveDashboard:
    @staticmethod
    def _render_kpis(df: pd.DataFrame) -> None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Contratti Totali", f"{len(df):,}")
        
        if 'Base Bid Price (€)' in df.columns:
            total_value = df['Base Bid Price (€)'].sum()
            col2.metric("Valore Totale", Utils.format_currency(total_value, 2))
            avg_value = df['Base Bid Price (€)'].mean()
            col3.metric("Valore Medio", Utils.format_currency(avg_value, 0))
            
        if 'Execution deadline (days)_numeric' in df.columns:
            avg_deadline = df['Execution deadline (days)_numeric'].mean()
            col4.metric("Scadenza Media (Giorni)", f"{avg_deadline:.0f}")

    @staticmethod
    def _render_criteria_pie(df: pd.DataFrame) -> None:
        criteria_col = 'Award criteria class'
        if criteria_col in df.columns:
            st.markdown("### Contratti per Criterio")
            award_dist = df[criteria_col].value_counts().reset_index()
            award_dist.columns = ['Criterio', 'Conteggio']
            
            fig = px.pie(award_dist, values='Conteggio', names='Criterio',
                         title="Distribuzione Criteri di Aggiudicazione", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition='inside', textinfo='percent+label',
                              marker=dict(line=dict(color='#000000', width=1)))
            fig.update_layout(height=400, margin=dict(t=50, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_district_bars(df: pd.DataFrame) -> None:
        if 'District' in df.columns:
            st.markdown("### Top 10 Distretti per Volume Contratti")
            district_counts = df['District'].value_counts().nlargest(10).reset_index()
            district_counts.columns = ['Distretto', 'Conteggio']
            
            fig = px.bar(district_counts.sort_values('Conteggio'), 
                         x='Conteggio', y='Distretto', orientation='h',
                         color='Conteggio', color_continuous_scale='Viridis',
                         text='Conteggio')
            fig.update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0),
                              coloraxis_showscale=False,
                              yaxis_title=None, xaxis_title="Numero Contratti")
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render(df: pd.DataFrame) -> None:
        st.markdown("## Panoramica Esecutiva")
        ExecutiveDashboard._render_kpis(df)
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col2:
            ExecutiveDashboard._render_criteria_pie(df)
        with col1:
            ExecutiveDashboard._render_district_bars(df)


class FinancialDashboard:
    @staticmethod
    def _render_price_distribution(df: pd.DataFrame, price_col: str) -> None:
        st.markdown("### Distribuzione Prezzo Base (Log)")
        fig = px.histogram(df, x=price_col, nbins=100, marginal='box',
                         log_y=True, log_x=True, title="Distribuzione Prezzo Base (Scala Log)")
        fig.update_traces(marker=dict(color='#3498db', line=dict(color='black', width=0.5)))
        fig.update_layout(height=400, yaxis_title="Conteggio (Log)")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_price_by_criteria(df: pd.DataFrame, price_col: str, criteria_col: str) -> None:
        st.markdown("### Prezzo per Criterio (Log)")
        fig = px.box(df, x=criteria_col, y=price_col, color=criteria_col,
                     notched=True, title="Distribuzione Prezzo per Criterio (Scala Log)",
                     log_y=True)
        fig.update_layout(height=400, showlegend=False,
                          xaxis_title="Criterio Aggiudicazione",
                          yaxis_title="Prezzo Base (€, Log)")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_price_intensity(df: pd.DataFrame, price_col: str, deadline_col: str, price_day_col: str, criteria_col: str) -> None:
        st.markdown("### Analisi Intensità Progetto (Prezzo vs. Costo/Giorno)")
        sample_size = min(5000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        df_filt = df_sample[(df_sample[price_col] > 1) & (df_sample[deadline_col] > 1) & (df_sample[price_day_col] > 1)]
        
        fig = px.scatter(
            df_filt, x=price_col, y=price_day_col,
            color=criteria_col if criteria_col in df.columns else None,
            opacity=0.6, marginal_x='histogram', marginal_y='histogram',
            log_x=True, log_y=True,
            title="Intensità Prezzo (Log) vs Prezzo Base (Log)",
            hover_data=['District'] if 'District' in df.columns else None
        )
        fig.update_layout(height=500, xaxis_title="Prezzo Base (€, Log)", yaxis_title="Prezzo al Giorno (€, Log)")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_financial_metrics_table(df: pd.DataFrame, price_col: str) -> None:
        st.markdown("### Metriche Finanziarie per Distretto (Top 15 per Valore)")
        district_stats = df.groupby('District')[price_col].agg(
            Total_Value=('sum'), Average_Value=('mean'), Contracts=('count')
        ).reset_index()
        
        stats_display = district_stats.sort_values('Total_Value', ascending=False).head(15)
        stats_display['Total_Value'] = stats_display['Total_Value'].apply(lambda x: Utils.format_currency(x, 1))
        stats_display['Average_Value'] = stats_display['Average_Value'].apply(lambda x: Utils.format_currency(x, 0))
        
        st.dataframe(stats_display.set_index('District'), width='stretch')

    @staticmethod
    def _render_budget_treemap(df: pd.DataFrame, district_col: str, criteria_col: str, price_col: str) -> None:
        st.markdown("### Allocazione Budget Gerarchica (Distretto → Criterio)")
        
        # Preparazione dati
        tree_data = df.groupby([district_col, criteria_col])[price_col].sum().reset_index()
        tree_data.columns = ['Distretto', 'Criterio', 'Valore']
        tree_data = tree_data[tree_data['Valore'] > 0] # Filtro valori positivi

        if tree_data.empty:
            st.warning("Dati insufficienti per generare la Treemap con i filtri correnti.")
            return

        try:
            fig = px.treemap(tree_data, path=['Distretto', 'Criterio'], values='Valore',
                             title="Allocazione Budget: Distretto → Criterio Aggiudicazione",
                             color='Valore', color_continuous_scale='Blues',
                             hover_data={'Valore': ':,.0f'})
            fig.update_traces(textinfo='label+value+percent parent',
                              marker=dict(line=dict(width=2, color='white')))
            fig.update_layout(height=600, margin=dict(t=50, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Impossibile generare la Treemap per questa specifica combinazione di filtri. (Errore: {e})")

    @staticmethod
    def render(df: pd.DataFrame) -> None:
        st.markdown("## Analisi Finanziaria")
        price_col = 'Base Bid Price (€)'
        if price_col not in df.columns:
            st.warning("Colonna 'Base Bid Price (€)' non trovata."); return
        
        criteria_col, deadline_col, price_day_col = 'Award criteria class', 'Execution deadline (days)_numeric', 'Price per Day'
        
        col1, col2 = st.columns(2)
        with col1:
            FinancialDashboard._render_price_distribution(df, price_col)
        with col2:
            if criteria_col in df.columns:
                FinancialDashboard._render_price_by_criteria(df, price_col, criteria_col)
        
        if price_day_col in df.columns and deadline_col in df.columns:
            FinancialDashboard._render_price_intensity(df, price_col, deadline_col, price_day_col, criteria_col)
        
        if 'District' in df.columns and criteria_col in df.columns:
            FinancialDashboard._render_budget_treemap(df, 'District', criteria_col, price_col)
        
        if 'District' in df.columns:
            FinancialDashboard._render_financial_metrics_table(df, price_col)


class TemporalDashboard:
    @staticmethod
    def _render_volume_trend(df: pd.DataFrame, year_col: str) -> None:
        st.markdown("### Andamento Volume Contratti")
        yearly_counts = df[year_col].value_counts().sort_index().reset_index()
        yearly_counts.columns = ['Anno', 'Contratti']
        fig = px.line(yearly_counts, x='Anno', y='Contratti', markers=True, line_shape='spline', title="Contratti per Anno")
        fig.update_traces(line_color='#e74c3c', marker=dict(size=8))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_value_trend(df: pd.DataFrame, year_col: str, price_col: str) -> None:
        st.markdown("### Andamento Valore Totale Contratti")
        yearly_value = df.groupby(year_col)[price_col].sum().reset_index()
        yearly_value.columns = ['Anno', 'Valore Totale']
        fig = px.area(yearly_value, x='Anno', y='Valore Totale', line_shape='spline', title="Valore Totale per Anno")
        fig.update_traces(fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.3)', line_color='#3498db')
        fig.update_layout(height=400, yaxis_title="Valore Totale (€)")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_seasonal_heatmap(df: pd.DataFrame, year_col: str, month_col: str) -> None:
        st.markdown("### Heatmap Stagionale (Mese x Anno)")
        df[month_col] = df[month_col].astype(int)
        heatmap_data = df.groupby([year_col, month_col]).size().reset_index(name='Conteggio')
        heatmap_pivot = heatmap_data.pivot(index=month_col, columns=year_col, values='Conteggio').fillna(0)
        heatmap_pivot = heatmap_pivot.reindex(index=range(1, 13), columns=sorted(heatmap_pivot.columns)).fillna(0)
        
        fig = px.imshow(heatmap_pivot,
                        labels=dict(x="Anno", y="Mese", color="Contratti"),
                        x=heatmap_pivot.columns.astype(str),
                        y=['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic'],
                        color_continuous_scale='YlOrRd', aspect='auto',
                        title="Volume Contratti per Mese e Anno")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_criteria_evolution(df: pd.DataFrame, year_col: str, criteria_col: str, price_col: str) -> None:
        st.markdown("### Evoluzione Criteri di Aggiudicazione nel Tempo")
        evolution = df.groupby([year_col, criteria_col])[price_col].sum().reset_index()
        evolution.columns = ['Anno', 'Criterio', 'Valore Totale']
        
        fig = px.area(evolution, x='Anno', y='Valore Totale', color='Criterio',
                      title="Evoluzione Composizione Mercato per Criterio",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=500, hovermode='x unified',
                          yaxis_title="Valore Totale (€)",
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render(df: pd.DataFrame) -> None:
        st.markdown("## Analisi Temporale")
        year_col, month_col, price_col = 'Publication Year', 'Publication Month', 'Base Bid Price (€)'
        criteria_col = 'Award criteria class'
        
        if year_col not in df.columns:
            st.warning("Colonna 'Publication Year' non trovata."); return
            
        df[year_col] = df[year_col].astype(int)
        col1, col2 = st.columns(2)
        with col1:
            TemporalDashboard._render_volume_trend(df, year_col)
        with col2:
            if price_col in df.columns:
                TemporalDashboard._render_value_trend(df, year_col, price_col)
        
        if month_col in df.columns:
            TemporalDashboard._render_seasonal_heatmap(df, year_col, month_col)
        
        if criteria_col in df.columns and price_col in df.columns:
            TemporalDashboard._render_criteria_evolution(df, year_col, criteria_col, price_col)

class GeographicDashboard:
    @staticmethod
    def _render_maps(df: pd.DataFrame, geojson: Dict, key: str, price_col: str, district_col: str) -> None:
        st.markdown("### Mappe Interattive dei Distretti")
        col1, col2 = st.columns(2)
        metrics = df.groupby(district_col).agg(
            Contracts=(district_col, 'size'),
            Average_Value=(price_col, 'mean'),
            Total_Value=(price_col, 'sum')
        ).reset_index()
        
        hover_data = {'Contracts': True, 'Average_Value': ':.0f', 'Total_Value': ':,.0f'}
        base_map_args = dict(geojson=geojson, locations=district_col, featureidkey=key,
                             mapbox_style='carto-positron', zoom=5.5, 
                             center={'lat': 39.5, 'lon': -8.0}, opacity=0.7,
                             hover_name=district_col, hover_data=hover_data)

        with col1:
            fig_vol = px.choropleth_mapbox(metrics, color='Contracts',
                                         color_continuous_scale='Plasma',
                                         title="Volume Contratti per Distretto", **base_map_args)
            fig_vol.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_vol, use_container_width=True)

        with col2:
            fig_val = px.choropleth_mapbox(metrics, color='Average_Value',
                                         color_continuous_scale='Viridis',
                                         title="Valore Medio (€) per Distretto", **base_map_args)
            fig_val.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_val, use_container_width=True)

    @staticmethod
    def _render_bar_charts(df: pd.DataFrame, price_col: str, district_col: str) -> None:
        col1, col2 = st.columns(2)
        metrics = df.groupby(district_col)[price_col].agg(
            Total_Value='sum', Average_Value='mean'
        ).reset_index()

        with col1:
            st.markdown("### Valore Totale per Distretto (Top 15)")
            data = metrics.sort_values('Total_Value', ascending=False).head(15)
            fig = px.bar(data, x='Total_Value', y=district_col, orientation='h', 
                         color='Total_Value', color_continuous_scale='Reds',
                         title="Top 15 Distretti per Valore Totale")
            fig.update_layout(height=500, showlegend=False, xaxis_title="Valore Totale (€)", 
                              yaxis_title=None, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Valore Medio per Distretto (Top 15)")
            data = metrics.sort_values('Average_Value', ascending=False).head(15)
            fig = px.bar(data, x='Average_Value', y=district_col, orientation='h',
                         color='Average_Value', color_continuous_scale='Blues',
                         title="Top 15 Distretti per Valore Medio")
            fig.update_layout(height=500, showlegend=False, xaxis_title="Valore Medio (€)",
                              yaxis_title=None, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_scatter(df: pd.DataFrame, price_col: str, district_col: str) -> None:
        st.markdown("### Analisi Valore Medio e Numero di Contratti per Distretto")

        if df is None or df.empty:
            st.warning("Nessun dato disponibile per il grafico scatter.")
            return

        # Calcolo metriche
        metrics = df.groupby(district_col).agg(
            Total_Value=(price_col, 'sum'),
            Average_Value=(price_col, 'mean'),
            Contracts=(price_col, 'count')
        ).reset_index()

        # Pulizia dati
        metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=['Average_Value', 'Contracts', 'Total_Value'])

        # Se dopo la pulizia non restano dati validi, evita di disegnare il grafico
        if metrics.empty or metrics['Average_Value'].isna().any() or metrics['Contracts'].isna().any():
            st.warning("Dati insufficienti o non validi per visualizzare il grafico scatter.")
            return

        try:
            fig = px.scatter(
                metrics,
                x='Contracts',
                y='Total_Value',
                size='Average_Value',
                color='District' if 'District' in metrics.columns else district_col,
                hover_name=district_col,
                size_max=50,
                hover_data={'Average_Value': ':.0f', 'Total_Value': ':.0f'}
            )
            fig.update_layout(
                title="Valore Totale vs Numero di Contratti per Distretto",
                xaxis_title="Numero di Contratti",
                yaxis_title="Valore Totale (€)",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        except ValueError:
            st.warning("Impossibile generare il grafico scatter per mancanza di dati validi.")

    @staticmethod
    def render(df: pd.DataFrame, geojson: Dict, feature_key: str) -> None:
        st.markdown("## Analisi Geografica")
        district_col, price_col = 'District', 'Base Bid Price (€)'
        
        if district_col not in df.columns or price_col not in df.columns:
            st.warning("Colonne Geografiche o di Prezzo non trovate."); return
        
        if geojson is None or feature_key is None:
            st.error("Dati GeoJSON non caricati."); return

        GeographicDashboard._render_maps(df, geojson, feature_key, price_col, district_col)
        st.markdown("---")
        GeographicDashboard._render_bar_charts(df, price_col, district_col)
        GeographicDashboard._render_scatter(df, price_col, district_col)

class TextAnalysisDashboard:
    @staticmethod
    def _get_keyword_frequencies(df_full: pd.DataFrame) -> Optional[pd.Series]:
        keyword_cols: List[str] = [col for col in df_full.columns if col.startswith('cpvs_keyword_')]
        if not keyword_cols:
            return None
        keyword_counts = df_full[keyword_cols].sum().sort_values(ascending=False)
        keyword_counts.index = keyword_counts.index.str.replace('cpvs_keyword_', '', regex=False).str.replace('_', ' ')
        keyword_counts = keyword_counts.fillna(0)

        # Filtra solo valori numerici validi e > 0
        keyword_counts = keyword_counts[keyword_counts > 0]
        if keyword_counts.empty:
            return None
        return keyword_counts

    @staticmethod
    def _render_keyword_bars(frequencies: pd.Series) -> None:
        st.markdown("### Frequenza Keyword (Grafico a Barre)")
        if frequencies is None or frequencies.empty:
            st.warning("Nessuna keyword disponibile per il grafico a barre.")
            return

        df_keywords = frequencies.reset_index()
        df_keywords.columns = ['Keyword', 'Numero di Contratti']

        fig = px.bar(
            df_keywords.sort_values('Numero di Contratti', ascending=True),
            x='Numero di Contratti',
            y='Keyword',
            orientation='h',
            title="Frequenza delle Top Keyword nei Contratti",
            text='Numero di Contratti',
            color='Numero di Contratti',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=500,
            showlegend=False,
            yaxis_title=None,
            xaxis_title="Numero di Contratti (Frequenza Assoluta)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_pie_chart(frequencies: pd.Series) -> None:
        st.markdown("### Distribuzione Percentuale delle Keyword (Pie Chart)")
        if frequencies is None or frequencies.empty:
            st.warning("Nessuna keyword disponibile per il grafico a torta.")
            return

        df_keywords = frequencies.reset_index()
        df_keywords.columns = ['Keyword', 'Numero di Contratti']
        
        # Clean data: remove NaN and ensure numeric values
        df_keywords['Numero di Contratti'] = pd.to_numeric(df_keywords['Numero di Contratti'], errors='coerce').fillna(0)
        df_keywords = df_keywords[df_keywords['Numero di Contratti'] > 0]
        
        total_contracts = df_keywords['Numero di Contratti'].sum()
        if total_contracts == 0 or pd.isna(total_contracts):
            st.warning("Nessun dato valido per creare il grafico a torta.")
            return

        df_keywords['Percentuale'] = (df_keywords['Numero di Contratti'] / total_contracts) * 100

        fig = px.pie(
            df_keywords,
            values='Numero di Contratti',
            names='Keyword',
            title="Distribuzione Percentuale delle Keyword nei Contratti",
            hole=0.3,
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_wordcloud(frequencies: pd.Series) -> None:
        st.markdown("### Temi Principali (Word Cloud Dinamica)")
        if frequencies is None or frequencies.empty:
            st.warning("Nessuna keyword da visualizzare nella WordCloud.")
            return

        # Clean data: remove NaN, inf, and non-positive values
        freq_clean = frequencies.replace([np.inf, -np.inf], np.nan).dropna()
        freq_clean = freq_clean[freq_clean > 0]
        
        if freq_clean.empty:
            st.warning("Nessuna keyword valida per generare la WordCloud.")
            return
        
        # Convert to dict with proper numeric values
        freq_dict = {str(k): float(v) for k, v in freq_clean.items() if pd.notna(v) and v > 0}
        
        if not freq_dict:
            st.warning("Nessuna keyword valida per generare la WordCloud.")
            return

        wc = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            colormap='viridis',
            random_state=42
        )
        wc.generate_from_frequencies(freq_dict)
        st.image(wc.to_array(), use_container_width=True)

    @staticmethod
    def render(df_full: pd.DataFrame) -> None:
        st.markdown("## Analisi Testuale (CPVS)")
        st.info("Questa sezione analizza le keyword più rilevanti (identificate tramite TF-IDF) e mostra la loro frequenza totale nel dataset.")
        frequencies = TextAnalysisDashboard._get_keyword_frequencies(df_full)

        if frequencies is not None and not frequencies.empty:
            col1, col2 = st.columns(2)
            with col1: TextAnalysisDashboard._render_keyword_bars(frequencies)
            with col2: TextAnalysisDashboard._render_pie_chart(frequencies)
            TextAnalysisDashboard._render_wordcloud(frequencies)
        else:
            st.warning("Nessuna keyword trovata nel dataset.")

class DataExplorer:
    @staticmethod
    def _render_stats(df: pd.DataFrame) -> None:
        st.markdown(f"### Statistiche Descrittive ({len(df):,} Righe Selezionate)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats_df = df[numeric_cols].describe().T
            stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2)
            for col in stats_df.columns:
                 if col not in ['count']:
                    stats_df[col] = stats_df[col].apply(lambda x: f"{x:,.2f}")
            st.dataframe(stats_df, width='stretch')
        
    @staticmethod
    def _render_info(df: pd.DataFrame) -> None:
        st.markdown(f"### Informazioni Colonne ({len(df.columns)} Colonne)")
        
        # Prevent division by zero
        total_rows = len(df) if len(df) > 0 else 1
        percentuale_nulli = ((df.isnull().sum() / total_rows) * 100).round(2).values
        
        col_info = pd.DataFrame({
            'Colonna': df.columns,
            'Tipo Dati': df.dtypes.values.astype(str), # Correzione PyArrow
            'Valori Non-Nulli': df.count().values,
            'Valori Unici': df.nunique().values,
            'Percentuale Nulli': percentuale_nulli
        })
        st.dataframe(col_info.set_index('Colonna'), width='stretch')

    @staticmethod
    def render(df: pd.DataFrame) -> None:
        st.markdown("## Esplorazione dei Dati Filtrati")
        DataExplorer._render_stats(df)
        DataExplorer._render_info(df)
        st.markdown(f"### Dati Grezzi (Primi 100 Record Filtrati)")
        st.dataframe(df.head(100), width='stretch')
        
class App:
    def __init__(self, data_path: str):
        self.df_cleaned = load_data(data_path)
        self.geojson, self.geo_key = load_geojson()
        self.df_filtered = self.df_cleaned.copy()

    def _apply_sidebar_filters(self) -> None:
        filter_manager = FilterManager(self.df_cleaned)
        self.df_filtered = filter_manager.apply_filters()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Record Selezionati: {len(self.df_filtered):,}**")
        st.sidebar.markdown(f"**Record Totali: {len(self.df_cleaned):,}**")

    def _render_header_footer(self) -> None:
        st.markdown("<h1 class='main-header'>Dashboard Appalti Pubblici Portoghesi 🏗️</h1>",
                    unsafe_allow_html=True)
        st.markdown("#### Analisi interattiva dei contratti di costruzione e servizi (PPP)")

    def _render_tabs(self) -> None:
        tab_list = [
            "Panoramica Esecutiva 📊",
            "Analisi Geografica 🗺️",
            "Analisi Temporale 📅",
            "Analisi Finanziaria 💰",
            "Analisi Testuale ☁️",
            "Esplora Dati 🔬"
        ]
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)
        
        with tab1:
            ExecutiveDashboard.render(self.df_filtered)
        with tab2:
            GeographicDashboard.render(self.df_filtered, self.geojson, self.geo_key)
        with tab3:
            TemporalDashboard.render(self.df_filtered)
        with tab4:
            FinancialDashboard.render(self.df_filtered)
        with tab5:
            TextAnalysisDashboard.render(self.df_cleaned) 
        with tab6:
            DataExplorer.render(self.df_filtered)

    def run(self) -> None:
        self._apply_sidebar_filters()
        self._render_header_footer()
        self._render_tabs()

# --- Esecuzione Applicazione ---
if __name__ == "__main__":
    app = App(DATA_FILE_PATH)
    app.run()