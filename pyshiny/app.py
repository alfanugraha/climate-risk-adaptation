"""
app.py — Climate Risk & Credit Scoring Dashboard for Indonesia
==============================================================
Multi-page PyShiny dashboard: TerraClimate exploration (2021-2025)
and synthetic climate-informed credit scores.

Run:  shiny run app.py --host 0.0.0.0 --port 8000
"""

import copy
import json

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from data_loader import (
    DF_KAB, DF_PROV_CLIMATE, DF_PROV_SCORES,
    GEOJSON_KAB, GEOJSON_PROV,
    YEARS, PROVINCES,
    PROV_TO_NAMAPROP, NAMAPROP_TO_PROV,
)
from scoring import national_score, score_color, score_label, top_regions

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CENTER_LAT, CENTER_LNG = -2.5, 118.0
ZOOM = 5
YEAR_CHOICES = {str(y): str(y) for y in YEARS}

CLIMATE_CMAPS = {
    "tmean_annual": ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"],
    "ppt_annual":   ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
}
SCORE_CMAP = ["#d73027", "#f46d43", "#fee08b", "#a6d96a", "#1a9641"]

BASEMAPS = {
    "CartoDB Positron": "CartoDB positron",
    "OpenStreetMap": "OpenStreetMap",
    "CartoDB Dark": "CartoDB dark_matter",
    "Esri WorldImagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Esri Topo": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CSS + JS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Light theme (default) ── */
:root {
    --bg: #f4f5f7; --sf: #ffffff; --sf2: #f0f1f4; --bd: #dfe1e6;
    --tx: #1a1d27; --tx2: #5e6278; --ac: #5b50e6;
    --ac-glow: rgba(91,80,230,.12); --ok: #16a34a; --wr: #d97706; --dg: #dc2626;
    --r: 12px; --shadow: 0 1px 6px rgba(0,0,0,.06);
    --plotly-tx: #1a1d27; --plotly-grid: #eee;
}

/* ── Dark theme ── */
[data-theme="dark"] {
    --bg: #0f1117; --sf: #1a1d27; --sf2: #242836; --bd: #2e3345;
    --tx: #e4e6ef; --tx2: #8b8fa3; --ac: #6c63ff;
    --ac-glow: rgba(108,99,255,.18); --ok: #22c55e; --wr: #f59e0b; --dg: #ef4444;
    --shadow: 0 1px 12px rgba(0,0,0,.35);
    --plotly-tx: #e4e6ef; --plotly-grid: #2e3345;
}

* { box-sizing: border-box; }
body {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--tx) !important;
    transition: background .25s, color .25s;
}

/* ── Navbar — override ANY inline style PyShiny injects ── */
.navbar,
.navbar-default,
.navbar-static-top,
nav.navbar {
    position: sticky !important;
    top: 0 !important;
    z-index: 1030 !important;
    background: var(--sf) !important;
    background-color: var(--sf) !important;
    border-bottom: 1px solid var(--bd) !important;
    box-shadow: var(--shadow) !important;
    transition: background .25s, background-color .25s, border-color .25s, box-shadow .25s;
}
.navbar-brand,
.navbar > .container-fluid > .navbar-brand {
    font-weight: 800 !important;
    color: var(--tx) !important;
    letter-spacing: -.3px;
    transition: color .25s;
}
.navbar-nav > li > a {
    color: var(--tx2) !important;
    font-weight: 500;
    transition: color .2s;
}
.navbar-nav > li.active > a,
.navbar-nav > li > a:hover {
    color: var(--ac) !important;
}
/* Also style the Bootstrap 5 .nav-link if used */
.nav-link { color: var(--tx2) !important; transition: color .2s; }
.nav-link.active, .nav-link:hover { color: var(--ac) !important; }

/* ── Cards ── */
.card, .well {
    background: var(--sf) !important; border: 1px solid var(--bd) !important;
    border-radius: var(--r) !important; box-shadow: var(--shadow);
    color: var(--tx) !important; transition: background .25s, border-color .25s;
}
.card-header {
    background: var(--sf2) !important;
    border-bottom: 1px solid var(--bd) !important;
    font-weight: 600; transition: background .25s;
    color: var(--tx) !important;
}

/* ── Metric badge ── */
.metric-badge {
    background: linear-gradient(135deg, var(--ac), #8b5cf6);
    color: #fff; padding: 18px 28px;
    text-align: center; box-shadow: 0 4px 20px var(--ac-glow);
    width: 100%; height: 100%;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
}
.metric-badge .label { font-size: .78rem; opacity: .85; text-transform: uppercase; letter-spacing: .8px; }
.metric-badge .value { font-size: 2.2rem; font-weight: 800; line-height: 1.1; margin-top: 4px; }

/* card body containing metric-badge fills remaining height */
.card-body:has(.metric-badge) {
    display: flex; flex-direction: column;
    padding: 0 !important; flex: 1;
}
.card:has(.metric-badge) {
    display: flex; flex-direction: column;
}

/* ── Inputs ── */
.form-control, .selectize-input, .form-select {
    background: var(--sf2) !important; border: 1px solid var(--bd) !important;
    color: var(--tx) !important; border-radius: 8px !important;
    transition: background .25s, border-color .25s, color .25s;
}
.selectize-dropdown, .selectize-dropdown-content {
    background: var(--sf) !important; color: var(--tx) !important;
    border: 1px solid var(--bd) !important;
}
.selectize-dropdown .option.active { background: var(--sf2) !important; }
label.control-label { color: var(--tx2) !important; font-weight: 500; font-size: .82rem; }

/* ── Radio buttons ── */
.shiny-input-radiogroup label,
.shiny-input-checkboxgroup label {
    color: var(--tx) !important;
}

/* ── Data table ── */
.dtable { width: 100%; border-collapse: collapse; }
.dtable th { background: var(--sf2); color: var(--tx2); font-size: .78rem; text-transform: uppercase; letter-spacing: .5px; padding: 10px 12px; border-bottom: 2px solid var(--bd); }
.dtable td { padding: 8px 12px; border-bottom: 1px solid var(--bd); font-size: .88rem; color: var(--tx); }
.dtable tr:hover td { background: var(--sf2); }

/* ── Plotly theme adaptation ── */
.js-plotly-plot text,
.js-plotly-plot .gtitle,
.js-plotly-plot .xtitle,
.js-plotly-plot .ytitle,
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text,
.js-plotly-plot .legendtext {
    fill: var(--tx) !important;
}
.js-plotly-plot .gridlayer line,
.js-plotly-plot .zerolinelayer line {
    stroke: var(--bd) !important;
}
.js-plotly-plot .legend { opacity: 1; }

/* ── Map container ── */
.map-wrap { border-radius: var(--r); overflow: hidden; border: 1px solid var(--bd); }

/* ── Tabs ── */
.tab-content > .tab-pane { padding: 24px 20px; }

/* ── About ── */
.about-section { max-width: 780px; margin: 0 auto; line-height: 1.75; }
.about-section h3 { color: var(--ac); margin-top: 28px; }
.about-section code { background: var(--sf2); padding: 2px 7px; border-radius: 5px; font-size: .88em; color: var(--tx); }
.about-section p, .about-section li { color: var(--tx); }

/* ━━━ Floating filter panel (Jelajah Data) ━━━ */
.float-panel {
    position: absolute; top: 14px; left: 14px; z-index: 1000;
    background: var(--sf); border: 1px solid var(--bd);
    border-radius: var(--r); padding: 18px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,.10);
    min-width: 250px; max-width: 280px;
    transition: background .25s, border-color .25s, box-shadow .25s;
    backdrop-filter: blur(8px);
    background: rgba(255,255,255,.92);
}
[data-theme="dark"] .float-panel {
    box-shadow: 0 4px 24px rgba(0,0,0,.5);
    background: rgba(26,29,39,.92);
}
.float-panel h6 {
    margin: 0 0 12px 0; font-weight: 700; font-size: .9rem;
    color: var(--tx);
}

/* ━━━ Theme toggle button ━━━ */
.theme-toggle {
    cursor: pointer; border: 1px solid var(--bd); background: var(--sf2);
    color: var(--tx); border-radius: 8px; padding: 6px 14px;
    font-family: inherit; font-size: .82rem; font-weight: 600;
    display: inline-flex; align-items: center; gap: 6px;
    transition: all .25s;
}
.theme-toggle:hover { border-color: var(--ac); color: var(--ac); }

/* ━━━ Full-page map (Jelajah Data) — height set by JS ━━━ */
.explore-page { position: relative; }
.explore-page .fullmap {
    width: 100%;
    height: var(--map-h, calc(100vh - 62px));
    border-radius: 0; border: none;
}

/* kill padding on explore tab */
#main_nav-tabpanel-explore_tab { padding: 0 !important; }
</style>
"""

HEAD_SCRIPTS = """
<!-- Load Plotly ONCE from CDN -->
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>

<script>
// ── Theme toggle ──
document.addEventListener('DOMContentLoaded', function() {
    const saved = localStorage.getItem('dashboard-theme') || 'light';
    if (saved === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
    updateToggleLabel();
    setMapHeight();
    window.addEventListener('resize', setMapHeight);
});

function toggleTheme() {
    const html = document.documentElement;
    const isDark = html.getAttribute('data-theme') === 'dark';
    if (isDark) {
        html.removeAttribute('data-theme');
        localStorage.setItem('dashboard-theme', 'light');
    } else {
        html.setAttribute('data-theme', 'dark');
        localStorage.setItem('dashboard-theme', 'dark');
    }
    updateToggleLabel();
}

function updateToggleLabel() {
    const btn = document.getElementById('theme-btn');
    if (!btn) return;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    btn.innerHTML = isDark ? '☀️ Light' : '🌙 Dark';
}

// ── Dynamic map height based on viewport ──
function setMapHeight() {
    const navbar = document.querySelector('.navbar');
    const navH = navbar ? navbar.offsetHeight : 56;
    const vh = window.innerHeight;
    const mapH = vh - navH;
    document.documentElement.style.setProperty('--map-h', mapH + 'px');
}

// Re-measure after Shiny renders new content
if (window.Shiny) {
    document.addEventListener('shiny:value', function() {
        setTimeout(setMapHeight, 100);
    });
}
</script>
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAP BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_choropleth(
    geojson_src: dict,
    lookup: dict[str, float],
    geojson_key: str,
    caption: str,
    colors: list[str],
    vmin: float,
    vmax: float,
    basemap: str = "CartoDB positron",
    height: str = "100%",
) -> str:
    """Build a Folium choropleth and return HTML string."""
    geojson = copy.deepcopy(geojson_src)
    colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=caption)

    for feat in geojson["features"]:
        name = feat["properties"].get(geojson_key, "")
        val = lookup.get(name)
        feat["properties"]["_value"] = f"{val:,.1f}" if val is not None else "N/A"

    def style_fn(feature):
        name = feature["properties"].get(geojson_key, "")
        val = lookup.get(name)
        return {
            "fillColor": colormap(val) if val is not None else "#ccc",
            "color": "#888", "weight": 0.5, "fillOpacity": 0.72,
        }

    def highlight_fn(feature):
        return {"weight": 2.5, "color": "#333", "fillOpacity": 0.92}

    if basemap.startswith("http"):
        m = folium.Map(location=[CENTER_LAT, CENTER_LNG], zoom_start=ZOOM, tiles=None, control_scale=True)
        folium.TileLayer(tiles=basemap, attr="Esri", name="Esri").add_to(m)
    else:
        m = folium.Map(location=[CENTER_LAT, CENTER_LNG], zoom_start=ZOOM, tiles=basemap, control_scale=True)

    tooltip_style = (
        "background:#fff;color:#1a1d27;"
        "font-family:'Plus Jakarta Sans',sans-serif;"
        "padding:8px 14px;border-radius:8px;border:1px solid #dfe1e6;"
        "font-size:13px;box-shadow:0 2px 8px rgba(0,0,0,.1);"
    )

    folium.GeoJson(
        geojson,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=[geojson_key, "_value"],
            aliases=["Wilayah:", f"{caption}:"],
            style=tooltip_style,
        ),
    ).add_to(m)
    colormap.add_to(m)
    return m._repr_html_()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app_ui = ui.page_navbar(
    ui.head_content(ui.HTML(CUSTOM_CSS), ui.HTML(HEAD_SCRIPTS)),

    # ── PAGE 1: Jelajah Data ──
    ui.nav_panel(
        "🌏 Jelajah Data",
        ui.div(
            ui.div(
                ui.tags.h6("🔎 Filter"),
                ui.input_select("explore_year", "Tahun", choices=YEAR_CHOICES, selected=str(YEARS[-1])),
                ui.input_select(
                    "explore_var", "Variabel",
                    choices={
                        "tmean_annual": "🌡️ Suhu Rata-rata (°C)",
                        "ppt_annual": "🌧️ Curah Hujan (mm)",
                    },
                    selected="tmean_annual",
                ),
                ui.input_select(
                    "explore_level", "Level Peta",
                    choices={"kab": "Kabupaten/Kota", "prov": "Provinsi"},
                    selected="kab",
                ),
                ui.input_select(
                    "explore_basemap", "Basemap",
                    choices=list(BASEMAPS.keys()),
                    selected="CartoDB Positron",
                ),
                class_="float-panel",
            ),
            ui.div(ui.output_ui("explore_map"), class_="fullmap"),
            class_="explore-page",
        ),
        value="explore_tab",
    ),

    # ── PAGE 2: Kredit Skoring ──
    ui.nav_panel(
        "📊 Kredit Skoring",
        ui.layout_columns(
            ui.card(
                ui.card_header("Filter"),
                ui.input_select("score_year", "Tahun", choices=YEAR_CHOICES, selected="2021"),
                ui.input_radio_buttons(
                    "score_view", "Tampilan",
                    choices={"prov10": "Top 10 Provinsi", "kab50": "Top 50 Kab/Kota"},
                    selected="prov10",
                ),
                ui.input_select(
                    "score_basemap", "Basemap",
                    choices=list(BASEMAPS.keys()),
                    selected="CartoDB Positron",
                ),
            ),
            ui.card(ui.card_header("Skor Nasional"), ui.output_ui("national_badge")),
            ui.card(ui.card_header("Ringkasan"), ui.output_ui("score_summary")),
            col_widths=(3, 5, 4),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Peta Skor Kredit per Provinsi"),
                ui.div(ui.output_ui("score_map"), class_="map-wrap"),
            ),
            ui.card(ui.card_header("Tabel Data"), ui.output_ui("score_table")),
            col_widths=(8, 4),
        ),
        ui.layout_columns(
            ui.card(ui.card_header("Tren Skor Kredit Nasional"), ui.output_ui("trend_chart")),
            ui.card(ui.card_header("Indikator Iklim per Provinsi"), ui.output_ui("climate_bar_chart")),
            col_widths=(6, 6),
        ),
    ),

    # ── PAGE 3: Tentang ──
    ui.nav_panel(
        "ℹ️ Tentang",
        ui.HTML("""
        <div class="about-section">
            <h2 style="font-weight:800;">Climate Risk Credit Scoring Dashboard</h2>
            <p style="color:var(--tx2);">Dashboard analisis risiko iklim dan skoring kredit sintetis untuk wilayah Indonesia.</p>
            <h3>📡 Sumber Data</h3>
            <p>Data iklim berasal dari <strong>TerraClimate</strong> (Abatzoglou et al., 2018) — dataset iklim global bulanan beresolusi ~4 km.</p>
            <ul><li><code>tmax</code> — Suhu maksimum bulanan (°C)</li><li><code>tmin</code> — Suhu minimum bulanan (°C)</li><li><code>ppt</code> — Curah hujan akumulasi bulanan (mm)</li></ul>
            <p>Periode: <strong>2021–2025</strong>, di-clip per provinsi Indonesia.</p>
            <h3>🧮 Metodologi</h3>
            <p>Data raster NetCDF diagregasi spasial (zonal statistics) ke kabupaten/kota menggunakan batas administrasi Kemendagri 2024. Agregasi temporal: rata-rata tahunan (suhu) dan total tahunan (curah hujan).</p>
            <h3>💳 Formula Skor Kredit</h3>
            <pre style="background:var(--sf2);padding:16px;border-radius:8px;color:var(--ac);">credit_score = 700 − 120 × norm(tmax) − 80 × norm(ppt_std) + noise(σ=15)
Clipped to [300, 850]</pre>
            <p>Daerah dengan suhu ekstrem tinggi dan variabilitas curah hujan besar → skor kredit lebih rendah.</p>
            <h3>🛠️ Tech Stack</h3>
            <p>Python · PyShiny · xarray · GeoPandas · Folium · Plotly · rasterstats</p>
            <hr style="border-color:var(--bd);margin:32px 0;">
            <p style="color:var(--tx2);font-size:.85rem;text-align:center;">Built with ❤️ for climate risk analytics in Indonesia</p>
        </div>
        """),
    ),

    # ── Theme toggle (navbar right) ──
    ui.nav_spacer(),
    ui.nav_control(
        ui.tags.button("🌙 Dark", id="theme-btn", class_="theme-toggle", onclick="toggleTheme()"),
    ),

    title="🇮🇩 Indonesia Climate Risk Dashboard",
    id="main_nav",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SERVER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def server(input: Inputs, output: Outputs, session: Session):

    # ────────── PAGE 1: Jelajah Data ──────────

    @output
    @render.ui
    def explore_map():
        year = int(input.explore_year())
        var = input.explore_var()
        level = input.explore_level()
        basemap_key = input.explore_basemap()
        basemap = BASEMAPS.get(basemap_key, "CartoDB positron")

        caption = "Suhu Rata-rata (°C)" if var == "tmean_annual" else "Curah Hujan (mm)"
        colors = CLIMATE_CMAPS[var]

        if level == "prov":
            df = DF_PROV_CLIMATE[DF_PROV_CLIMATE["year"] == year]
            lookup = {}
            for _, r in df.iterrows():
                np_name = PROV_TO_NAMAPROP.get(r["province"])
                if np_name:
                    lookup[np_name] = r[var]
            html = build_choropleth(
                GEOJSON_PROV, lookup, "nama_prop", caption, colors,
                vmin=df[var].min(), vmax=df[var].max(), basemap=basemap,
            )
        else:
            df = DF_KAB[DF_KAB["year"] == year]
            lookup = dict(zip(df["kabupaten"], df[var]))
            html = build_choropleth(
                GEOJSON_KAB, lookup, "nama_kab", caption, colors,
                vmin=df[var].min(), vmax=df[var].max(), basemap=basemap,
            )

        return ui.HTML(html)

    # ────────── PAGE 2: Kredit Skoring ──────────

    @reactive.Calc
    def sel_year():
        return int(input.score_year())

    @output
    @render.ui
    def national_badge():
        year = sel_year()
        score = national_score(DF_KAB, year)
        return ui.HTML(f"""
            <div class="metric-badge">
                <div class="label">Skor Kredit Nasional {year}</div>
                <div class="value">{score:.0f}</div>
                <div style="margin-top:6px;font-size:.85rem;opacity:.8;">{score_label(score)}</div>
            </div>
        """)

    @output
    @render.ui
    def score_summary():
        year = sel_year()
        df_yr = DF_KAB[DF_KAB["year"] == year]
        best = df_yr.loc[df_yr["credit_score"].idxmax()]
        worst = df_yr.loc[df_yr["credit_score"].idxmin()]
        return ui.HTML(f"""
            <div style="padding:12px;">
                <div style="margin-bottom:16px;">
                    <span style="color:var(--tx2);font-size:.78rem;text-transform:uppercase;">Tertinggi</span><br>
                    <span style="font-weight:700;color:var(--ok);font-size:1.1rem;">{best['credit_score']}</span>
                    <span style="color:var(--tx2);margin-left:8px;">{best['kabupaten']}</span>
                </div>
                <div style="margin-bottom:16px;">
                    <span style="color:var(--tx2);font-size:.78rem;text-transform:uppercase;">Terendah</span><br>
                    <span style="font-weight:700;color:var(--dg);font-size:1.1rem;">{worst['credit_score']}</span>
                    <span style="color:var(--tx2);margin-left:8px;">{worst['kabupaten']}</span>
                </div>
                <div>
                    <span style="color:var(--tx2);font-size:.78rem;text-transform:uppercase;">Std Deviasi</span><br>
                    <span style="font-weight:700;font-size:1.1rem;">{df_yr['credit_score'].std():.1f}</span>
                </div>
            </div>
        """)

    @output
    @render.ui
    def score_map():
        year = sel_year()
        basemap_key = input.score_basemap()
        basemap = BASEMAPS.get(basemap_key, "CartoDB positron")

        df = DF_PROV_SCORES[DF_PROV_SCORES["year"] == year]
        lookup = {}
        for _, r in df.iterrows():
            np_name = PROV_TO_NAMAPROP.get(r["province"])
            if np_name:
                lookup[np_name] = r["credit_score"]

        html = build_choropleth(
            GEOJSON_PROV, lookup, "nama_prop", "Credit Score", SCORE_CMAP,
            vmin=df["credit_score"].min() - 20, vmax=df["credit_score"].max() + 20,
            basemap=basemap, height="460px",
        )
        return ui.HTML(f'<div style="height:460px;">{html}</div>')

    @output
    @render.ui
    def score_table():
        year = sel_year()
        view = input.score_view()
        df = top_regions(DF_KAB, year,
                         n=10 if view == "prov10" else 50,
                         level="province" if view == "prov10" else "kabupaten")

        if view == "prov10":
            cols, headers = ["province", "credit_score"], ["Provinsi", "Skor"]
        else:
            cols, headers = ["province", "kabupaten", "credit_score"], ["Provinsi", "Kab/Kota", "Skor"]

        rows = ""
        for _, r in df[cols].iterrows():
            cells = ""
            for c in cols:
                v = r[c]
                if c == "credit_score":
                    cells += f'<td><span style="color:{score_color(v)};font-weight:700;">{v}</span></td>'
                else:
                    cells += f"<td>{v}</td>"
            rows += f"<tr>{cells}</tr>"

        hdr = "".join(f"<th>{h}</th>" for h in headers)
        return ui.HTML(f'<div style="max-height:420px;overflow-y:auto;"><table class="dtable"><thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody></table></div>')

    @output
    @render.ui
    def trend_chart():
        trend = DF_KAB.groupby("year")["credit_score"].mean().reset_index()
        trend["credit_score"] = trend["credit_score"].round(1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["year"], y=trend["credit_score"],
            mode="lines+markers",
            line=dict(color="#5b50e6", width=3),
            marker=dict(size=10, color="#5b50e6", line=dict(color="#fff", width=2)),
            fill="tozeroy", fillcolor="rgba(91,80,230,0.06)",
            hovertemplate="Tahun %{x}<br>Skor: %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", size=12),
            margin=dict(l=50, r=20, t=20, b=40), height=500,
            xaxis=dict(dtick=1, title="Tahun", gridcolor="#eee"),
            yaxis=dict(title="Skor Kredit", gridcolor="#eee", range=[400, 800]),
        )
        # include_plotlyjs=False because we load Plotly once in <head>
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False))

    @output
    @render.ui
    def climate_bar_chart():
        year = sel_year()
        df = DF_PROV_SCORES[DF_PROV_SCORES["year"] == year].sort_values("tmax_annual", ascending=False).head(15)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["province"], y=df["tmax_annual"], name="Suhu Maksimum (°C)",
            marker_color="#dc2626",
            hovertemplate="%{x}<br>Tmax: %{y:.1f}°C<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=df["province"], y=df["ppt_annual"] / 100, name="Curah Hujan (×100 mm)",
            marker_color="#2563eb",
            hovertemplate="%{x}<br>PPT: %{customdata:.0f} mm<extra></extra>",
            customdata=df["ppt_annual"],
        ))
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", size=12),
            margin=dict(l=50, r=20, t=20, b=80), height=500,
            barmode="group",
            xaxis=dict(tickangle=-45, gridcolor="#eee"),
            yaxis=dict(gridcolor="#eee", range=[0, 40]),
            legend=dict(orientation="h", y=1.12),
        )
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = App(app_ui, server)
