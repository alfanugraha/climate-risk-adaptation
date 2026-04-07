"""
OCBC Climate Risk Dashboard — Streamlit App
Indonesia Flood Risk Assessment & Portfolio Impact Analysis

Cara menjalankan:
  Di Colab  : gunakan pyngrok (lihat notebook)
  Di lokal  : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
import json
import os
from datetime import datetime

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="OCBC Climate Risk Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS custom — warna mengikuti brand OCBC
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #D85A30;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .risk-badge-tinggi { 
        background: #A32D2D; color: white; 
        padding: 2px 10px; border-radius: 12px; font-size: 0.85em;
    }
    .risk-badge-sedang { 
        background: #854F0B; color: white; 
        padding: 2px 10px; border-radius: 12px; font-size: 0.85em;
    }
    .risk-badge-rendah { 
        background: #0F6E56; color: white; 
        padding: 2px 10px; border-radius: 12px; font-size: 0.85em;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        padding: 8px 20px; border-radius: 8px 8px 0 0;
    }
    h1 { color: #1a1a2e; }
    .sidebar-info {
        background: #e8f4fd; padding: 0.8rem;
        border-radius: 8px; font-size: 0.85em; margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING — dengan caching untuk performa
# ============================================================
@st.cache_data(ttl=3600, show_spinner="Memuat data banjir Indonesia...")
def load_flood_data(filepath=None):
    """
    Load data banjir dari parquet atau generate data sintetis.
    Data sintetis dibuat berdasarkan distribusi statistik BNPB.
    """
    if filepath and os.path.exists(filepath):
        df = pd.read_parquet(filepath)
    else:
        # Data sintetis berbasis distribusi nyata
        np.random.seed(42)
        PROVINCE_WEIGHTS = {
            'Jawa Barat': 0.18, 'Jawa Tengah': 0.15, 'Jawa Timur': 0.13,
            'Sumatera Utara': 0.09, 'Kalimantan Selatan': 0.08,
            'Sulawesi Selatan': 0.07, 'Banten': 0.06, 'DKI Jakarta': 0.05,
            'Riau': 0.05, 'Kalimantan Tengah': 0.04,
            'Papua': 0.04, 'Nusa Tenggara Timur': 0.03, 'Aceh': 0.03
        }
        n = 2500
        provinces = list(PROVINCE_WEIGHTS.keys())
        weights = list(PROVINCE_WEIGHTS.values())
        df = pd.DataFrame({
            'event_id': [f'FLD-{i:05d}' for i in range(n)],
            'date': pd.to_datetime(np.random.choice(
                pd.date_range('2020-01-01', '2024-12-31', freq='D'), n
            )),
            'province': np.random.choice(provinces, n, p=weights),
            'flood_area_km2': np.random.lognormal(2.5, 1.2, n).round(2),
            'duration_days': np.random.geometric(0.25, n),
            'lat': np.random.uniform(-8.5, 5.5, n).round(4),
            'lon': np.random.uniform(95.0, 141.0, n).round(4),
            'source_type': np.random.choice(
                ['news_article', 'community_report', 'government_report', 'social_media'],
                n, p=[0.45, 0.30, 0.20, 0.05]
            )
        })

    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter
    return df


@st.cache_data(show_spinner=False)
def calculate_risk_scores(df, group_col='province', w_freq=0.40, w_mag=0.35, w_trend=0.25):
    """Hitung Climate Flood Risk Score per wilayah."""
    n_years = max(df['year'].nunique(), 1)
    freq = (df.groupby(group_col)['event_id'].count() / n_years).rename('freq_per_year')
    magnitude = df.groupby(group_col)['flood_area_km2'].mean().rename('avg_area_km2')

    max_year = df['year'].max()
    recent = df[df['year'] >= max_year - 2]
    older = df[df['year'] < max_year - 2]
    trend_raw = (
        recent.groupby(group_col)['event_id'].count() -
        older.groupby(group_col)['event_id'].count()
    ).fillna(0)

    result = pd.DataFrame({
        'freq_per_year': freq, 'avg_area_km2': magnitude, 'trend_raw': trend_raw
    }).fillna(0)

    def minmax(s):
        return (s - s.min()) / (s.max() - s.min()) * 100 if s.max() != s.min() else pd.Series(50, index=s.index)

    result['comp_frekuensi'] = minmax(result['freq_per_year']).round(1)
    result['comp_magnitude'] = minmax(result['avg_area_km2']).round(1)
    result['comp_trend'] = minmax(result['trend_raw'].clip(lower=0)).round(1)
    result['risk_score'] = (
        w_freq * result['comp_frekuensi'] +
        w_mag  * result['comp_magnitude'] +
        w_trend * result['comp_trend']
    ).round(1)

    def kategori(s):
        if s <= 25: return 'Rendah'
        elif s <= 50: return 'Sedang'
        elif s <= 75: return 'Tinggi'
        else: return 'Sangat Tinggi'

    result['kategori_risiko'] = result['risk_score'].apply(kategori)
    result['ranking'] = result['risk_score'].rank(ascending=False).astype(int)
    return result.sort_values('risk_score', ascending=False).reset_index()


# ============================================================
# SIR MODEL
# ============================================================
def run_sir_scenarios(N=10000, I0=210, beta=0.30, gamma=0.15):
    """Jalankan 3 skenario SIR credit contagion."""

    def sir_ode(t, y, b):
        S, I, R = y
        return [-b*S*I/N, b*S*I/N - gamma*I, gamma*I]

    def solve_piecewise(beta_segments):
        """Solve SIR dengan beta berbeda per segmen waktu."""
        t_all, S_all, I_all, R_all = [], [], [], []
        y = [N - I0, I0, 0]
        for (t0, t1, b) in beta_segments:
            if t0 >= t1:
                continue
            t_eval = np.linspace(t0, t1, (t1 - t0) * 10)
            sol = solve_ivp(sir_ode, (t0, t1), y, args=(b,), t_eval=t_eval)
            t_all.extend(sol.t)
            S_all.extend(sol.y[0])
            I_all.extend(sol.y[1])
            R_all.extend(sol.y[2])
            y = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1]]
        return np.array(t_all), np.array(I_all)

    scenarios = {
        'Baseline': [(0, 12, beta)],
        'Moderat (β ×1.5)': [(0, 3, beta), (3, 5, beta*1.5), (5, 12, beta)],
        'Ekstrem (β ×2.5)': [(0, 3, beta), (3, 7, beta*2.5), (7, 12, beta)],
    }
    colors = {'Baseline': '#1D9E75', 'Moderat (β ×1.5)': '#EF9F27', 'Ekstrem (β ×2.5)': '#A32D2D'}
    results = {}
    for name, segs in scenarios.items():
        t, I = solve_piecewise(segs)
        results[name] = {'t': t, 'npl_pct': I / N * 100, 'color': colors[name]}
    return results


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/OCBC_Bank_Logo.svg/320px-OCBC_Bank_Logo.svg.png",
             width=140)
    st.title("Climate Risk Dashboard")
    st.caption("Indonesia Flood Risk Assessment")

    st.divider()
    st.subheader("Filter Data")

    # Upload data nyata
    uploaded = st.file_uploader(
        "Upload data banjir (.parquet)",
        type=['parquet'],
        help="Upload file parquet dari flood-explorer repo"
    )
    filepath = None
    if uploaded:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as f:
            f.write(uploaded.read())
            filepath = f.name

    df_raw = load_flood_data(filepath)

    # Filter tahun
    years_available = sorted(df_raw['year'].unique())
    selected_years = st.multiselect(
        "Tahun", years_available, default=years_available,
        help="Filter data berdasarkan tahun kejadian"
    )

    # Filter provinsi
    provinces = sorted(df_raw['province'].unique())
    selected_provinces = st.multiselect(
        "Provinsi", provinces, default=provinces[:6],
        help="Pilih provinsi yang ingin dianalisis"
    )

    # Bobot risk score
    st.subheader("Bobot Risk Score")
    w_freq = st.slider("Frekuensi (%)", 10, 80, 40, 5) / 100
    w_mag = st.slider("Magnitude (%)", 10, 80, 35, 5) / 100
    w_trend = round(1.0 - w_freq - w_mag, 2)
    st.caption(f"Trend (auto): {w_trend*100:.0f}%")
    if w_trend < 0:
        st.error("Total bobot melebihi 100%. Kurangi Frekuensi atau Magnitude.")

    st.markdown("""<div class='sidebar-info'>
    <b>Data source:</b> Groundsource (Google Research) via flood-explorer<br>
    <b>Periode:</b> 2020–2024<br>
    <b>Pelatihan OCBC — Climate Risk AI</b>
    </div>""", unsafe_allow_html=True)


# ============================================================
# FILTER DATA
# ============================================================
df = df_raw[
    df_raw['year'].isin(selected_years) &
    df_raw['province'].isin(selected_provinces)
].copy()

risk_scores = calculate_risk_scores(df, w_freq=w_freq, w_mag=w_mag, w_trend=w_trend)

# ============================================================
# HEADER METRICS
# ============================================================
st.title("🌊 Indonesia Flood Risk — Climate Risk Dashboard")
st.caption(f"Data: {len(df):,} kejadian banjir | Provinsi terpilih: {len(selected_provinces)} | Tahun: {min(selected_years) if selected_years else '-'}–{max(selected_years) if selected_years else '-'}")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Kejadian", f"{len(df):,}", help="Jumlah kejadian banjir dalam filter")
with col2:
    st.metric("Total Luas Terdampak", f"{df['flood_area_km2'].sum():,.0f} km²")
with col3:
    avg_dur = df['duration_days'].mean()
    st.metric("Rata-rata Durasi", f"{avg_dur:.1f} hari")
with col4:
    high_risk = (risk_scores['kategori_risiko'].isin(['Tinggi', 'Sangat Tinggi'])).sum()
    st.metric("Provinsi Risiko Tinggi", f"{high_risk}", delta=f"dari {len(risk_scores)} total")
with col5:
    if not risk_scores.empty:
        top_prov = risk_scores.iloc[0]['province']
        top_score = risk_scores.iloc[0]['risk_score']
        st.metric("Risiko Tertinggi", top_prov, delta=f"Score: {top_score:.1f}")

st.divider()


# ============================================================
# TABS UTAMA
# ============================================================
tab_peta, tab_analitik, tab_risk, tab_sir = st.tabs([
    "🗺️ Peta Risiko", "📊 Analitik Banjir", "🏦 Risk Assessment Portofolio", "📈 Model SIR"
])


# --------------------------------------------------------
# TAB 1: PETA RISIKO
# --------------------------------------------------------
with tab_peta:
    st.subheader("Distribusi Spasial Kejadian Banjir")
    st.info("Peta menampilkan lokasi kejadian banjir berdasarkan koordinat. "
            "Untuk peta choropleth per wilayah administratif, upload GeoJSON dari fasilitator.")

    col_map, col_score = st.columns([3, 2])

    with col_map:
        metric_choice = st.radio(
            "Tampilkan:", ["Sebaran titik kejadian", "Heatmap intensitas"],
            horizontal=True
        )
        if metric_choice == "Sebaran titik kejadian":
            fig_map = px.scatter_mapbox(
                df.sample(min(500, len(df))),
                lat='lat', lon='lon',
                size='flood_area_km2',
                color='province',
                hover_name='province',
                hover_data={'flood_area_km2': ':.1f', 'duration_days': True, 'date': True},
                mapbox_style='carto-positron',
                zoom=4, center={'lat': -2.5, 'lon': 118},
                title='Titik Kejadian Banjir (sampel 500 kejadian)',
                height=500
            )
        else:
            fig_map = px.density_mapbox(
                df, lat='lat', lon='lon',
                z='flood_area_km2', radius=15,
                mapbox_style='carto-positron',
                zoom=4, center={'lat': -2.5, 'lon': 118},
                color_continuous_scale='OrRd',
                title='Heatmap Intensitas Banjir (luas terdampak)',
                height=500
            )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_score:
        st.markdown("**Climate Risk Score — Ranking Provinsi**")
        color_map_cat = {
            'Sangat Tinggi': '#A32D2D', 'Tinggi': '#D85A30',
            'Sedang': '#854F0B', 'Rendah': '#0F6E56'
        }
        fig_score = px.bar(
            risk_scores,
            x='risk_score', y='province', orientation='h',
            color='kategori_risiko', color_discrete_map=color_map_cat,
            text='risk_score',
            labels={'risk_score': 'Risk Score', 'province': ''},
            height=500
        )
        fig_score.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_score.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=True,
            margin=dict(l=0, r=60, t=20, b=20)
        )
        st.plotly_chart(fig_score, use_container_width=True)

    # Upload GeoJSON
    st.subheader("Upload GeoJSON untuk Peta Choropleth")
    col_geo1, col_geo2 = st.columns(2)
    with col_geo1:
        geo_level = st.selectbox("Level admin:", ["Provinsi", "Kabupaten/Kota", "Kecamatan", "Desa"])
        geo_file = st.file_uploader(f"Upload GeoJSON {geo_level}", type=['geojson', 'json'])

    with col_geo2:
        if geo_file:
            try:
                geo_data = json.load(geo_file)
                st.success(f"GeoJSON dimuat: {len(geo_data['features'])} fitur")

                # Ambil sample properti
                sample_props = list(geo_data['features'][0]['properties'].keys())
                id_col = st.selectbox("Kolom ID wilayah di GeoJSON:", sample_props)

                if st.button("Render Choropleth Map"):
                    fig_choro = px.choropleth_mapbox(
                        risk_scores,
                        geojson=geo_data,
                        locations='province',
                        featureidkey=f'properties.{id_col}',
                        color='risk_score',
                        color_continuous_scale='OrRd',
                        range_color=[0, 100],
                        mapbox_style='carto-positron',
                        zoom=4, center={'lat': -2.5, 'lon': 118},
                        opacity=0.7,
                        labels={'risk_score': 'Risk Score'},
                        title='Climate Risk Score — Choropleth Map'
                    )
                    fig_choro.update_layout(height=500)
                    st.plotly_chart(fig_choro, use_container_width=True)
            except Exception as e:
                st.error(f"Error membaca GeoJSON: {e}")
        else:
            st.info("Upload GeoJSON dari fasilitator untuk melihat choropleth map per wilayah administratif")


# --------------------------------------------------------
# TAB 2: ANALITIK BANJIR
# --------------------------------------------------------
with tab_analitik:
    st.subheader("Analisis Temporal dan Distribusi Banjir Indonesia")

    col_ts, col_hm = st.columns(2)

    with col_ts:
        # Time series
        monthly_ts = df.groupby(['year', 'month']).agg(
            kejadian=('event_id', 'count'),
            luas_total=('flood_area_km2', 'sum')
        ).reset_index()
        monthly_ts['period'] = pd.to_datetime(
            monthly_ts['year'].astype(str) + '-' + monthly_ts['month'].astype(str).str.zfill(2)
        )

        metric_ts = st.radio("Metrik:", ["Jumlah Kejadian", "Total Luas (km²)"], horizontal=True)
        y_col = 'kejadian' if metric_ts == "Jumlah Kejadian" else 'luas_total'
        y_label = 'Jumlah Kejadian' if metric_ts == "Jumlah Kejadian" else 'Total Luas (km²)'

        fig_ts = px.line(
            monthly_ts, x='period', y=y_col,
            labels={'period': 'Bulan', y_col: y_label},
            title=f'Tren Bulanan: {y_label}',
            color_discrete_sequence=['#D85A30']
        )
        fig_ts.update_traces(mode='lines+markers', marker_size=4)
        fig_ts.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_hm:
        # Heatmap musiman
        top8 = df['province'].value_counts().head(8).index
        hm_data = df[df['province'].isin(top8)].groupby(['province', 'month']).size().unstack(fill_value=0)
        month_labels = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']
        hm_cols = [month_labels[i-1] for i in hm_data.columns]

        fig_hm = px.imshow(
            hm_data, x=hm_cols,
            labels=dict(x='Bulan', y='Provinsi', color='Kejadian'),
            color_continuous_scale='OrRd',
            title='Pola Musiman — Top 8 Provinsi'
        )
        fig_hm.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig_hm, use_container_width=True)

    col_bar, col_scat = st.columns(2)

    with col_bar:
        # Bar chart per provinsi
        prov_agg = df.groupby('province').agg(
            kejadian=('event_id', 'count'),
            luas_total=('flood_area_km2', 'sum'),
            durasi_avg=('duration_days', 'mean')
        ).sort_values('kejadian', ascending=False).head(10).reset_index()

        fig_bar = px.bar(
            prov_agg, x='kejadian', y='province', orientation='h',
            color='luas_total', color_continuous_scale='OrRd',
            labels={'kejadian': 'Jumlah Kejadian', 'province': '',
                    'luas_total': 'Total Luas (km²)'},
            title='Top 10 Provinsi — Frekuensi Banjir',
            text='kejadian'
        )
        fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
        fig_bar.update_layout(height=350, yaxis={'categoryorder': 'total ascending'},
                              margin=dict(t=40, b=20, r=40))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_scat:
        # Scatter: luas vs durasi
        fig_scat = px.scatter(
            df.sample(min(300, len(df))),
            x='duration_days', y='flood_area_km2',
            color='province', size='flood_area_km2',
            log_y=True, opacity=0.7,
            labels={'duration_days': 'Durasi (hari)',
                    'flood_area_km2': 'Luas Terdampak km² (log)',
                    'province': 'Provinsi'},
            title='Hubungan Durasi vs Luas Terdampak',
            hover_data=['province', 'date'] if 'date' in df.columns else ['province']
        )
        fig_scat.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig_scat, use_container_width=True)


# --------------------------------------------------------
# TAB 3: RISK ASSESSMENT PORTOFOLIO
# --------------------------------------------------------
with tab_risk:
    st.subheader("Estimasi Dampak Risiko Banjir ke Portofolio OCBC")
    st.info("Masukkan data portofolio Anda untuk menghitung estimasi potensi kerugian per wilayah. "
            "Angka default adalah data fiktif untuk keperluan pelatihan.")

    # Input portofolio
    st.markdown("**Input Portofolio per Provinsi (Rp Miliar)**")

    default_portfolio = {
        'Jawa Barat': (8500, 1200, 3200),
        'Jawa Tengah': (5200, 2800, 2100),
        'Jawa Timur': (4800, 2100, 2400),
        'DKI Jakarta': (12000, 50, 4500),
        'Banten': (3200, 400, 1200),
        'Sumatera Utara': (2100, 1800, 900),
    }

    portfolio_rows = []
    cols_input = st.columns([3, 2, 2, 2])
    cols_input[0].markdown("**Provinsi**")
    cols_input[1].markdown("**KPR (Rp M)**")
    cols_input[2].markdown("**Agrikultur (Rp M)**")
    cols_input[3].markdown("**UMKM (Rp M)**")

    for prov in selected_provinces[:8]:
        defaults = default_portfolio.get(prov, (500, 200, 300))
        c0, c1, c2, c3 = st.columns([3, 2, 2, 2])
        c0.markdown(f"*{prov}*")
        kpr = c1.number_input('', value=defaults[0], key=f'kpr_{prov}',
                               label_visibility='collapsed', min_value=0, step=100)
        agri = c2.number_input('', value=defaults[1], key=f'agri_{prov}',
                                label_visibility='collapsed', min_value=0, step=100)
        umkm = c3.number_input('', value=defaults[2], key=f'umkm_{prov}',
                                label_visibility='collapsed', min_value=0, step=100)
        portfolio_rows.append({
            'province': prov, 'kpr': kpr, 'agrikultur': agri, 'umkm': umkm
        })

    df_portfolio = pd.DataFrame(portfolio_rows)
    df_portfolio['total'] = df_portfolio['kpr'] + df_portfolio['agrikultur'] + df_portfolio['umkm']
    df_portfolio = df_portfolio.merge(
        risk_scores[['province', 'risk_score', 'kategori_risiko']],
        on='province', how='left'
    ).fillna({'risk_score': 0, 'kategori_risiko': 'Rendah'})

    # Asumsi kerugian
    st.markdown("**Asumsi Potensi Kerugian**")
    col_a1, col_a2, col_a3 = st.columns(3)
    kpr_loss_pct = col_a1.slider("KPR: penurunan nilai agunan (%)", 5, 50, 25) / 100
    agri_loss_pct = col_a2.slider("Agrikultur: gagal panen (%)", 10, 60, 30) / 100
    umkm_loss_pct = col_a3.slider("UMKM: gangguan operasional (%)", 5, 40, 20) / 100

    # Hitung potensi kerugian — proporsional terhadap risk score
    df_portfolio['exp_loss_kpr'] = (
        df_portfolio['kpr'] * kpr_loss_pct * df_portfolio['risk_score'] / 100
    ).round(1)
    df_portfolio['exp_loss_agri'] = (
        df_portfolio['agrikultur'] * agri_loss_pct * df_portfolio['risk_score'] / 100
    ).round(1)
    df_portfolio['exp_loss_umkm'] = (
        df_portfolio['umkm'] * umkm_loss_pct * df_portfolio['risk_score'] / 100
    ).round(1)
    df_portfolio['total_exp_loss'] = (
        df_portfolio['exp_loss_kpr'] +
        df_portfolio['exp_loss_agri'] +
        df_portfolio['exp_loss_umkm']
    ).round(1)
    df_portfolio['loss_pct'] = (
        df_portfolio['total_exp_loss'] / df_portfolio['total'] * 100
    ).round(2)

    # Tampilkan hasil
    st.divider()
    st.subheader("Hasil Estimasi")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Portofolio", f"Rp {df_portfolio['total'].sum():,.0f} M")
    col_m2.metric("Estimasi Kerugian",
                  f"Rp {df_portfolio['total_exp_loss'].sum():,.1f} M",
                  delta=f"{df_portfolio['total_exp_loss'].sum()/df_portfolio['total'].sum()*100:.2f}% dari portofolio",
                  delta_color="inverse")
    col_m3.metric("Provinsi Risiko Sangat Tinggi",
                  f"{(df_portfolio['kategori_risiko']=='Sangat Tinggi').sum()}")

    # Tabel hasil
    display_cols = ['province', 'risk_score', 'kategori_risiko', 'total',
                    'total_exp_loss', 'loss_pct']
    col_labels = {
        'province': 'Provinsi', 'risk_score': 'Risk Score',
        'kategori_risiko': 'Kategori', 'total': 'Total Portfolio (Rp M)',
        'total_exp_loss': 'Est. Kerugian (Rp M)', 'loss_pct': 'Loss (%)'
    }
    st.dataframe(
        df_portfolio[display_cols].rename(columns=col_labels)
        .set_index('Provinsi')
        .style.background_gradient(subset=['Risk Score', 'Loss (%)'], cmap='OrRd'),
        use_container_width=True
    )

    # Waterfall chart
    fig_wf = go.Figure(go.Waterfall(
        orientation='v',
        measure=['relative'] * len(df_portfolio) + ['total'],
        x=list(df_portfolio['province']) + ['TOTAL'],
        y=list(df_portfolio['total_exp_loss']) + [df_portfolio['total_exp_loss'].sum()],
        connector={'line': {'color': 'rgb(63, 63, 63)'}},
        decreasing={'marker': {'color': '#D85A30'}},
        totals={'marker': {'color': '#A32D2D'}},
        text=[f"Rp {v:.0f}M" for v in list(df_portfolio['total_exp_loss']) +
              [df_portfolio['total_exp_loss'].sum()]],
        textposition='outside'
    ))
    fig_wf.update_layout(
        title='Dekomposisi Estimasi Kerugian per Provinsi',
        yaxis_title='Estimasi Kerugian (Rp Miliar)',
        height=380, showlegend=False
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    st.caption("DISCLAIMER: Semua angka dalam tab ini adalah estimasi untuk keperluan pelatihan. "
               "Analisis risiko aktual memerlukan data portofolio resmi dan model yang divalidasi oleh tim risiko OCBC.")


# --------------------------------------------------------
# TAB 4: MODEL SIR
# --------------------------------------------------------
with tab_sir:
    st.subheader("SIR Model — Credit Risk Contagion saat Banjir")
    st.markdown("""
    Model SIR diadaptasi dari epidemiologi untuk mensimulasikan bagaimana tekanan kredit
    "menyebar" antar debitur saat terjadi bencana banjir:
    - **S (Sehat)** → debitur dengan kredit performing
    - **I (Tertekan)** → debitur dengan kredit bermasalah (special mention–loss)
    - **R (Resolved)** → debitur yang direstrukturisasi atau dihapusbukukan
    """)

    col_param, col_result = st.columns([1, 2])

    with col_param:
        st.markdown("**Parameter Model**")
        N = st.number_input("Total debitur (N)", 1000, 100000, 10000, 1000)
        I0 = st.number_input("NPL awal (debitur)", 1, 5000, 210, 10)
        beta = st.slider("β — laju transmisi tekanan", 0.05, 1.0, 0.30, 0.05,
                         help="Seberapa cepat tekanan kredit menyebar (lebih tinggi = lebih cepat)")
        gamma = st.slider("γ — laju resolusi", 0.05, 0.5, 0.15, 0.05,
                          help="Seberapa cepat kredit bermasalah terselesaikan")
        st.metric("R₀ (basic reproduction number)", f"{beta/gamma:.2f}",
                  help="R₀ > 1 = tekanan kredit akan menyebar. R₀ < 1 = akan mereda sendiri")

    with col_result:
        if st.button("Jalankan Simulasi", type='primary'):
            with st.spinner('Menghitung simulasi...'):
                sir_results = run_sir_scenarios(N=N, I0=I0, beta=beta, gamma=gamma)

            fig_sir = go.Figure()
            for name, data in sir_results.items():
                dash = 'solid' if 'Baseline' in name else ('dash' if 'Moderat' in name else 'dot')
                fig_sir.add_trace(go.Scatter(
                    x=data['t'], y=data['npl_pct'],
                    name=name, line=dict(color=data['color'], width=2.5, dash=dash),
                    hovertemplate=f'{name}<br>Q%{{x:.1f}}: NPL %{{y:.2f}}%<extra></extra>'
                ))

            fig_sir.add_vrect(x0=3, x1=5, fillcolor='#EF9F27', opacity=0.08,
                              annotation_text='Shock moderat', annotation_position='top left')
            fig_sir.add_vrect(x0=3, x1=7, fillcolor='#A32D2D', opacity=0.05,
                              annotation_text='Shock ekstrem', annotation_position='top right')
            fig_sir.add_hline(y=I0/N*100, line_dash='dot', line_color='gray',
                              annotation_text=f'NPL baseline {I0/N*100:.1f}%')

            fig_sir.update_layout(
                title='Simulasi Credit Risk Contagion — 3 Skenario Banjir',
                xaxis_title='Kuartal ke-', yaxis_title='NPL Efektif (%)',
                hovermode='x unified', height=420,
                legend=dict(yanchor='top', y=0.98, xanchor='left', x=0.01)
            )
            st.plotly_chart(fig_sir, use_container_width=True)

            # Summary tabel
            st.markdown("**Ringkasan Hasil Simulasi**")
            summary_rows = []
            for name, data in sir_results.items():
                peak_npl = data['npl_pct'].max()
                peak_q = data['t'][data['npl_pct'].argmax()]
                summary_rows.append({
                    'Skenario': name,
                    'Peak NPL (%)': f"{peak_npl:.2f}%",
                    'Kuartal Peak': f"Q{peak_q:.1f}",
                    'Kenaikan dari Baseline': f"+{peak_npl - I0/N*100:.2f}%"
                })
            st.dataframe(pd.DataFrame(summary_rows).set_index('Skenario'),
                         use_container_width=True)
        else:
            st.info("Atur parameter di kiri dan klik 'Jalankan Simulasi'")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "OCBC Climate Risk AI Training | Dataset: Groundsource-based synthetic data | "
    f"Diperbarui: {datetime.now().strftime('%d %b %Y %H:%M')} | "
    "Untuk keperluan pelatihan internal — bukan analisis risiko resmi"
)
