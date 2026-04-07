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
CENTER_LAT, CENTER_LNG = -6.5, 118.0
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

TOOLTIP_STYLE = (
    "background:#1a1d27;color:#e4e6ef;"
    "font-family:'Plus Jakarta Sans',sans-serif;"
    "padding:8px 12px;border-radius:6px;border:1px solid #2e3345;"
    "font-size:13px;"
)

# ── CSS ──
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
:root {
    --bg: #0f1117; --sf: #1a1d27; --sf2: #242836; --bd: #2e3345;
    --tx: #e4e6ef; --tx2: #8b8fa3; --ac: #6c63ff;
    --ac-glow: rgba(108,99,255,.18); --ok: #22c55e; --wr: #f59e0b; --dg: #ef4444;
    --r: 12px;
}
* { box-sizing: border-box; }
body { font-family: 'Plus Jakarta Sans', sans-serif !important; background: var(--bg) !important; color: var(--tx) !important; }
.navbar, .navbar-default { background: var(--sf) !important; border-bottom: 1px solid var(--bd) !important; box-shadow: 0 1px 12px rgba(0,0,0,.35); }
.navbar-brand { font-weight: 800 !important; color: var(--tx) !important; letter-spacing: -.3px; }
.navbar-nav > li > a { color: var(--tx2) !important; font-weight: 500; transition: color .2s; }
.navbar-nav > li.active > a, .navbar-nav > li > a:hover { color: var(--ac) !important; }
.card, .well { background: var(--sf) !important; border: 1px solid var(--bd) !important; border-radius: var(--r) !important; box-shadow: 0 2px 12px rgba(0,0,0,.2); color: var(--tx) !important; }
.card-header { background: var(--sf2) !important; border-bottom: 1px solid var(--bd) !important; font-weight: 600; }
.metric-badge { background: linear-gradient(135deg, var(--ac), #8b5cf6); color: #fff; padding: 18px 28px; border-radius: var(--r); text-align: center; box-shadow: 0 4px 20px var(--ac-glow); }
.metric-badge .label { font-size: .78rem; opacity: .85; text-transform: uppercase; letter-spacing: .8px; }
.metric-badge .value { font-size: 2.2rem; font-weight: 800; line-height: 1.1; margin-top: 4px; }
.form-control, .selectize-input, .form-select { background: var(--sf2) !important; border: 1px solid var(--bd) !important; color: var(--tx) !important; border-radius: 8px !important; }
label.control-label { color: var(--tx2) !important; font-weight: 500; font-size: .82rem; }
.dtable { width: 100%; border-collapse: collapse; }
.dtable th { background: var(--sf2); color: var(--tx2); font-size: .78rem; text-transform: uppercase; letter-spacing: .5px; padding: 10px 12px; border-bottom: 2px solid var(--bd); }
.dtable td { padding: 8px 12px; border-bottom: 1px solid var(--bd); font-size: .88rem; }
.dtable tr:hover td { background: var(--sf2); }
.map-wrap { border-radius: var(--r); overflow: hidden; border: 1px solid var(--bd); }
.tab-content > .tab-pane { padding: 24px 20px; }
.about-section { max-width: 780px; margin: 0 auto; line-height: 1.75; }
.about-section h3 { color: var(--ac); margin-top: 28px; }
.about-section code { background: var(--sf2); padding: 2px 7px; border-radius: 5px; font-size: .88em; }
</style>
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
    height: int = 560,
    extra_fields: dict[str, dict[str, str]] | None = None,
) -> str:
    """
    Build a Folium choropleth and return its HTML string.

    Parameters
    ----------
    geojson_src : raw GeoJSON dict (will be deep-copied)
    lookup      : {feature_name: value} for coloring
    geojson_key : property name in GeoJSON features to match against lookup keys
    caption     : legend caption
    colors      : list of hex colors for LinearColormap
    vmin, vmax  : color scale range
    extra_fields: optional {feature_name: {field: value}} to inject into tooltips
    """
    geojson = copy.deepcopy(geojson_src)

    colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=caption)

    # Inject _value (and any extra fields) into feature properties for tooltip
    for feat in geojson["features"]:
        name = feat["properties"].get(geojson_key, "")
        val = lookup.get(name)
        feat["properties"]["_value"] = f"{val:,.1f}" if val is not None else "N/A"
        if extra_fields and name in extra_fields:
            for k, v in extra_fields[name].items():
                feat["properties"][k] = v

    def style_fn(feature):
        name = feature["properties"].get(geojson_key, "")
        val = lookup.get(name)
        return {
            "fillColor": colormap(val) if val is not None else "#333",
            "color": "#444",
            "weight": 0.5,
            "fillOpacity": 0.72,
        }

    def highlight_fn(feature):
        return {"weight": 2.5, "color": "#fff", "fillOpacity": 0.92}

    m = folium.Map(
        location=[CENTER_LAT, CENTER_LNG],
        zoom_start=ZOOM,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # Determine tooltip fields
    tooltip_fields = [geojson_key, "_value"]
    tooltip_aliases = ["Wilayah:", f"{caption}:"]

    folium.GeoJson(
        geojson,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            style=TOOLTIP_STYLE,
        ),
    ).add_to(m)

    colormap.add_to(m)

    return f'<div style="height:{height}px;">{m._repr_html_()}</div>'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UI DEFINITION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app_ui = ui.page_navbar(
    ui.head_content(ui.HTML(CUSTOM_CSS)),

    # ── PAGE 1: Jelajah Data ──────────────────────────
    ui.nav_panel(
        "🌏 Jelajah Data",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("explore_year", "Tahun", choices=YEAR_CHOICES, selected=str(YEARS[-1])),
                ui.input_radio_buttons(
                    "explore_var", "Variabel",
                    choices={
                        "tmean_annual": "🌡️ Suhu Rata-rata (°C)",
                        "ppt_annual": "🌧️ Curah Hujan (mm)",
                    },
                    selected="tmean_annual",
                ),
                ui.input_radio_buttons(
                    "explore_level", "Level Peta",
                    choices={"kab": "Kabupaten/Kota", "prov": "Provinsi"},
                    selected="kab",
                ),
                ui.hr(),
                ui.markdown("**Sumber**: TerraClimate (2021–2025)  \nResolusi ~4 km"),
                width=280, bg="#1a1d27",
            ),
            ui.div(ui.output_ui("explore_map"), class_="map-wrap"),
        ),
    ),

    # ── PAGE 2: Kredit Skoring ────────────────────────
    ui.nav_panel(
        "📊 Kredit Skoring",
        # Row 1: Filters + Badge
        ui.layout_columns(
            ui.card(
                ui.card_header("Filter"),
                ui.input_select("score_year", "Tahun", choices=YEAR_CHOICES, selected=str(YEARS[-1])),
                ui.input_radio_buttons(
                    "score_view", "Tampilan",
                    choices={"prov10": "Top 10 Provinsi", "kab50": "Top 50 Kab/Kota"},
                    selected="prov10",
                ),
            ),
            ui.card(ui.card_header("Skor Nasional"), ui.output_ui("national_badge")),
            ui.card(ui.card_header("Ringkasan"), ui.output_ui("score_summary")),
            col_widths=(3, 5, 4),
        ),
        # Row 2: Map + Table
        ui.layout_columns(
            ui.card(
                ui.card_header("Peta Skor Kredit per Provinsi"),
                ui.div(ui.output_ui("score_map"), class_="map-wrap"),
            ),
            ui.card(ui.card_header("Tabel Data"), ui.output_ui("score_table")),
            col_widths=(8, 4),
        ),
        # Row 3: Charts
        ui.layout_columns(
            ui.card(ui.card_header("Tren Skor Kredit Nasional"), ui.output_ui("trend_chart")),
            ui.card(ui.card_header("Indikator Iklim per Provinsi"), ui.output_ui("climate_bar_chart")),
            col_widths=(6, 6),
        ),
    ),

    # ── PAGE 3: Tentang ───────────────────────────────
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
            <pre style="background:var(--sf2);padding:16px;border-radius:8px;color:#a5b4fc;">credit_score = 700 − 120 × norm(tmax) − 80 × norm(ppt_std) + noise(σ=15)
Clipped to [300, 850]</pre>
            <p>Daerah dengan suhu ekstrem tinggi dan variabilitas curah hujan besar → skor kredit lebih rendah.</p>
            <h3>🛠️ Tech Stack</h3>
            <p>Python · PyShiny · xarray · GeoPandas · Folium · Plotly · rasterstats</p>
            <hr style="border-color:var(--bd);margin:32px 0;">
            <p style="color:var(--tx2);font-size:.85rem;text-align:center;">Built with ❤️ for climate risk analytics in Indonesia</p>
        </div>
        """),
    ),

    title="🇮🇩 Indonesia Climate Risk Dashboard",
    id="main_nav",
    navbar_options=ui.navbar_options(bg="#1a1d27"),
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SERVER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#e4e6ef", size=12),
)


def server(input: Inputs, output: Outputs, session: Session):

    # ────────── PAGE 1: Jelajah Data ──────────

    @output
    @render.ui
    def explore_map():
        year = int(input.explore_year())
        var = input.explore_var()
        level = input.explore_level()

        caption = "Suhu Rata-rata (°C)" if var == "tmean_annual" else "Curah Hujan (mm)"
        colors = CLIMATE_CMAPS[var]

        if level == "prov":
            df = DF_PROV_CLIMATE[DF_PROV_CLIMATE["year"] == year]
            # Build lookup: nama_prop (uppercase) → value
            lookup = {}
            for _, r in df.iterrows():
                np_name = PROV_TO_NAMAPROP.get(r["province"])
                if np_name:
                    lookup[np_name] = r[var]

            return ui.HTML(build_choropleth(
                GEOJSON_PROV, lookup, "nama_prop", caption, colors,
                vmin=df[var].min(), vmax=df[var].max(), height=600,
            ))
        else:
            # Kabupaten level — lookup by nama_kab
            df = DF_KAB[DF_KAB["year"] == year]
            lookup = dict(zip(df["kabupaten"], df[var]))

            return ui.HTML(build_choropleth(
                GEOJSON_KAB, lookup, "nama_kab", caption, colors,
                vmin=df[var].min(), vmax=df[var].max(), height=600,
            ))

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
        df = DF_PROV_SCORES[DF_PROV_SCORES["year"] == year]
        lookup = {}
        for _, r in df.iterrows():
            np_name = PROV_TO_NAMAPROP.get(r["province"])
            if np_name:
                lookup[np_name] = r["credit_score"]

        return ui.HTML(build_choropleth(
            GEOJSON_PROV, lookup, "nama_prop", "Credit Score", SCORE_CMAP,
            vmin=df["credit_score"].min() - 20, vmax=df["credit_score"].max() + 20,
            height=460,
        ))

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
            line=dict(color="#6c63ff", width=3),
            marker=dict(size=10, color="#6c63ff", line=dict(color="#fff", width=2)),
            fill="tozeroy", fillcolor="rgba(108,99,255,0.08)",
            hovertemplate="Tahun %{x}<br>Skor: %{y:.1f}<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            margin=dict(l=40, r=20, t=20, b=40), height=300,
            xaxis=dict(dtick=1, title="Tahun", gridcolor="#2e3345"),
            yaxis=dict(title="Skor Kredit", gridcolor="#2e3345", range=[400, 800]),
        )
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    @output
    @render.ui
    def climate_bar_chart():
        year = sel_year()
        df = DF_PROV_SCORES[DF_PROV_SCORES["year"] == year].sort_values("tmax_annual", ascending=False).head(15)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["province"], y=df["tmax_annual"], name="Tmax (°C)",
            marker_color="#ef4444",
            hovertemplate="%{x}<br>Tmax: %{y:.1f}°C<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=df["province"], y=df["ppt_annual"] / 100, name="PPT (×100 mm)",
            marker_color="#3b82f6",
            hovertemplate="%{x}<br>PPT: %{customdata:.0f} mm<extra></extra>",
            customdata=df["ppt_annual"],
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            margin=dict(l=40, r=20, t=20, b=80), height=300,
            barmode="group",
            xaxis=dict(tickangle=-45, gridcolor="#2e3345"),
            yaxis=dict(gridcolor="#2e3345"),
            legend=dict(orientation="h", y=1.12),
        )
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = App(app_ui, server)
