# 🇮🇩 Indonesia Climate Risk Credit Scoring Dashboard

A PyShiny dashboard for exploring TerraClimate data (2021–2025) and synthetic climate-informed credit scores across Indonesian provinces and kabupaten/kota.

## Project Structure

```
climate_dashboard/
├── app.py              # Main PyShiny app (UI + server)
├── data_loader.py      # Data loading module (cached on import)
├── scoring.py          # Credit score utilities
├── preprocess.py       # Offline preprocessing (NetCDF → CSV/GeoJSON)
├── requirements.txt    # Python dependencies
├── data/
│   ├── kabupaten_data.csv      # Kabupaten-level climate + credit scores
│   ├── province_climate.csv    # Province-level annual climate aggregates
│   ├── province_scores.csv     # Province-level aggregated credit scores
│   └── provinces.geojson       # Province boundaries
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-run preprocessing with full shapefile
python preprocess.py \
  --nc-dir /path/to/clipped_nc \
  --shp-path /path/to/Batas_Kabupaten_Kemendagri_2024.shp \
  --out-dir ./data

# 3. Run dashboard
shiny run app.py --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.

## Dashboard Pages

| Page | Description |
|------|-------------|
| **🌏 Jelajah Data** | Interactive choropleth map of annual temperature or precipitation by province. Year + variable selectors. |
| **📊 Kredit Skoring** | Credit score explorer: national badge, top provinces/kabupaten table, score map, trend line chart, climate indicator bar chart. |
| **ℹ️ Tentang** | Methodology, data sources, and formula documentation. |

## Adding Kabupaten Boundaries

The dashboard currently uses approximate province-level bounding boxes because the `.shp` file was not available during initial preprocessing. To upgrade to full kabupaten polygons:

1. Place all shapefile components (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`) in the project root
2. Re-run: `python preprocess.py --shp-path ./Batas_Kabupaten_Kemendagri_2024.shp`
3. This creates `data/kabupaten.geojson` with simplified polygons
4. Update `data_loader.py` to also load `kabupaten.geojson`
5. Update `app.py` map functions to use kabupaten-level GeoJSON

## Credit Score Formula

```
credit_score = 700
    − 120 × normalized(tmax_annual)
    − 80  × normalized(ppt_std)
    + random_noise(σ=15)

Clipped to [300, 850]
```

Higher temperature extremes and higher rainfall variability produce lower scores.

## Data Sources

- **TerraClimate** — Abatzoglou et al. (2018), ~4 km monthly climate
- **Admin boundaries** — Kemendagri 2024 (kabupaten/kota)
