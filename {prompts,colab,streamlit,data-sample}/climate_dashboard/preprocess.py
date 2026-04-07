"""
preprocess.py
=============
Run this ONCE offline to process raw TerraClimate NetCDF data into
dashboard-ready CSVs and GeoJSON.

Usage:
    python preprocess.py --nc-dir ./clipped_nc --shp-path ./Batas_Kabupaten_Kemendagri_2024.shp --out-dir ./data

Inputs:
    - clipped_nc/          → Province-clipped TerraClimate NetCDFs (ppt, tmax, tmin × 2021-2025)
    - *.shp (optional)     → Kabupaten/Kota admin boundaries (Kemendagri 2024)

Outputs (saved to --out-dir):
    - province_climate.csv → Province-level annual climate aggregates
    - kabupaten_data.csv   → Kabupaten-level climate + synthetic credit scores
    - province_scores.csv  → Province-level aggregated credit scores
    - kabupaten.geojson    → Kabupaten polygons (simplified for web)
    - provinces.geojson    → Province polygons (dissolved from kabupaten or from raster extents)
"""

import argparse
import os
import json
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box, mapping

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. Province-level climate from NetCDF rasters
# ──────────────────────────────────────────────
def process_netcdf(nc_dir: str) -> tuple[pd.DataFrame, dict]:
    """
    Read all province/variable/year NetCDF files and compute:
      - ppt_annual   : annual total precipitation (mm)
      - ppt_std      : monthly precipitation std (variability proxy)
      - tmax_annual  : annual mean of monthly max temperature (°C)
      - tmax_max     : absolute annual max temperature (°C)
      - tmin_annual  : annual mean of monthly min temperature (°C)
      - tmean_annual : (tmax_annual + tmin_annual) / 2

    Also extracts spatial bounding boxes per province for fallback geometry.
    """
    provinces = sorted(
        d for d in os.listdir(nc_dir) if os.path.isdir(os.path.join(nc_dir, d))
    )
    variables = ["ppt", "tmax", "tmin"]
    years = range(2021, 2026)

    records = []
    prov_bounds = {}

    for prov in provinces:
        prov_dir = os.path.join(nc_dir, prov)
        for year in years:
            row = {"province_folder": prov, "year": year}
            for var in variables:
                fname = os.path.join(prov_dir, f"{var}_{year}_{prov}.nc")
                if not os.path.exists(fname):
                    continue
                ds = xr.open_dataset(fname)
                data = ds[var]

                # Capture bounding box (once per province)
                if prov not in prov_bounds:
                    x, y = ds.x.values, ds.y.values
                    prov_bounds[prov] = dict(
                        xmin=float(x.min()), xmax=float(x.max()),
                        ymin=float(y.min()), ymax=float(y.max()),
                        xcen=float(x.mean()), ycen=float(y.mean()),
                    )

                if var == "ppt":
                    row["ppt_annual"] = float(data.sum(dim="time").mean(skipna=True))
                    monthly = data.mean(dim=["x", "y"], skipna=True)
                    row["ppt_std"] = float(monthly.std())
                elif var == "tmax":
                    row["tmax_annual"] = float(data.mean(dim="time").mean(skipna=True))
                    row["tmax_max"] = float(data.max(skipna=True))
                elif var == "tmin":
                    row["tmin_annual"] = float(data.mean(dim="time").mean(skipna=True))

                ds.close()

            row["tmean_annual"] = (
                row.get("tmax_annual", 0) + row.get("tmin_annual", 0)
            ) / 2
            records.append(row)

    df = pd.DataFrame(records)
    df["province"] = df["province_folder"].apply(_clean_province_name)
    return df, prov_bounds


def _clean_province_name(folder: str) -> str:
    name = folder.replace("_", " ").title()
    replacements = {
        "P A P U A": "Papua",
        "Dki": "DKI",
        "Daerah Istimewa Yogyakarta": "DI Yogyakarta",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


# ──────────────────────────────────────────────
# 2. Kabupaten data + synthetic credit scores
# ──────────────────────────────────────────────
def generate_kabupaten_data(
    df_prov: pd.DataFrame, shp_path: str | None
) -> tuple[pd.DataFrame, gpd.GeoDataFrame | None]:
    """
    Build kabupaten-level climate + credit score dataframe.
    If shapefile available, use kabupaten names from it.
    Returns (df_kab, gdf_kab_or_None).
    """
    gdf_kab = None

    if shp_path and os.path.exists(shp_path):
        gdf_kab = gpd.read_file(shp_path)
        kab_info = gdf_kab[["nama_prop", "nama_kab"]].copy()
        kab_info.columns = ["province_shp", "kabupaten"]
    else:
        # Fallback: read from .dbf if exists
        dbf = shp_path.replace(".shp", ".dbf") if shp_path else None
        if dbf and os.path.exists(dbf):
            tmp = gpd.read_file(dbf)
            kab_info = tmp[["nama_prop", "nama_kab"]].copy()
            kab_info.columns = ["province_shp", "kabupaten"]
        else:
            raise FileNotFoundError("No shapefile or .dbf found for kabupaten names.")

    # Map shapefile province names → folder names
    prov_names = df_prov[["province_folder", "province"]].drop_duplicates()
    prov_map = _build_province_mapping(
        kab_info["province_shp"].unique(), prov_names["province_folder"].unique()
    )
    kab_info["province_folder"] = kab_info["province_shp"].map(prov_map)
    kab_info = kab_info.merge(prov_names, on="province_folder", how="left")

    # Generate kabupaten-level records with noise around province means
    np.random.seed(42)
    years = sorted(df_prov["year"].unique())
    records = []

    for _, kr in kab_info.iterrows():
        prov_data = df_prov[df_prov["province_folder"] == kr["province_folder"]]
        for year in years:
            yr = prov_data[prov_data["year"] == year]
            if yr.empty:
                continue
            yr = yr.iloc[0]
            nt = np.random.normal(0, 0.5)
            np_ = np.random.normal(0, 80)
            tmax = yr["tmax_annual"] + nt
            tmin = yr["tmin_annual"] + nt * 0.8
            records.append(dict(
                province=kr["province"],
                province_folder=kr["province_folder"],
                kabupaten=kr["kabupaten"],
                year=year,
                tmax_annual=round(tmax, 2),
                tmin_annual=round(tmin, 2),
                tmean_annual=round((tmax + tmin) / 2, 2),
                ppt_annual=round(max(200, yr["ppt_annual"] + np_), 1),
                ppt_std=round(max(20, yr["ppt_std"] + np.random.normal(0, 15)), 2),
            ))

    df_kab = pd.DataFrame(records)

    # ── Credit score ──
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    norm = scaler.fit_transform(df_kab[["tmax_annual", "ppt_std"]].values)
    alpha, beta = 120, 80
    noise = np.random.normal(0, 15, len(df_kab))
    raw = 700 - alpha * norm[:, 0] - beta * norm[:, 1] + noise
    df_kab["credit_score"] = np.clip(raw, 300, 850).round(0).astype(int)

    return df_kab, gdf_kab


def _build_province_mapping(shp_names, folder_names):
    """Best-effort mapping from shapefile province names to NC folder names."""
    folder_set = set(folder_names)
    mapping = {}
    for name in shp_names:
        guess = name.lower().replace(" ", "_").replace("d.i.", "daerah_istimewa").replace(".", "")
        if guess in folder_set:
            mapping[name] = guess
        elif "papua" == guess:
            mapping[name] = "p_a_p_u_a"
        elif "kepulauan_riau" in guess:
            mapping[name] = "kepulauan_riau"
        elif "bangka" in guess:
            mapping[name] = "kepulauan_bangka_belitung"
        elif "yogyakarta" in guess:
            mapping[name] = "daerah_istimewa_yogyakarta"
        elif "jakarta" in guess:
            mapping[name] = "dki_jakarta"
        else:
            # Fuzzy fallback
            for f in folder_set:
                if guess[:6] in f:
                    mapping[name] = f
                    break
    return mapping


# ──────────────────────────────────────────────
# 3. GeoJSON generation
# ──────────────────────────────────────────────
def make_geojson(
    df_kab: pd.DataFrame,
    gdf_kab: gpd.GeoDataFrame | None,
    prov_bounds: dict,
    out_dir: str,
):
    """Create kabupaten and province GeoJSON files."""

    if gdf_kab is not None and "geometry" in gdf_kab.columns and gdf_kab.geometry.notna().any():
        # Simplify geometries for web performance
        gdf_kab = gdf_kab.to_crs("EPSG:4326")
        gdf_kab["geometry"] = gdf_kab.geometry.simplify(0.005, preserve_topology=True)
        gdf_kab[["nama_prop", "nama_kab", "geometry"]].to_file(
            os.path.join(out_dir, "kabupaten.geojson"), driver="GeoJSON"
        )
        # Dissolve to province
        gdf_prov = gdf_kab.dissolve(by="nama_prop").reset_index()
        gdf_prov["geometry"] = gdf_prov.geometry.simplify(0.01, preserve_topology=True)
        gdf_prov.rename(columns={"nama_prop": "province_shp"}).to_file(
            os.path.join(out_dir, "provinces.geojson"), driver="GeoJSON"
        )
        print("✓ Created kabupaten.geojson and provinces.geojson from shapefile")
    else:
        # Fallback: bounding-box provinces
        features = []
        for prov, b in prov_bounds.items():
            features.append(dict(
                type="Feature",
                properties=dict(province=_clean_province_name(prov), province_folder=prov),
                geometry=mapping(box(b["xmin"], b["ymin"], b["xmax"], b["ymax"])),
            ))
        with open(os.path.join(out_dir, "provinces.geojson"), "w") as f:
            json.dump(dict(type="FeatureCollection", features=features), f)
        print("⚠ Created approximate provinces.geojson (bounding boxes — no .shp found)")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess TerraClimate data for dashboard")
    parser.add_argument("--nc-dir", default="./clipped_nc", help="Path to clipped_nc folder")
    parser.add_argument("--shp-path", default=None, help="Path to kabupaten .shp file")
    parser.add_argument("--out-dir", default="./data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Processing NetCDF files…")
    df_prov, prov_bounds = process_netcdf(args.nc_dir)
    df_prov.to_csv(os.path.join(args.out_dir, "province_climate.csv"), index=False)

    print("Generating kabupaten data + credit scores…")
    # If no --shp-path given, try common locations
    shp = args.shp_path
    if not shp:
        for candidate in [
            "Batas_Kabupaten_Kemendagri_2024.shp",
            "data/Batas_Kabupaten_Kemendagri_2024.shp",
        ]:
            if os.path.exists(candidate):
                shp = candidate
                break
        if not shp:
            # Try .dbf fallback
            shp = "Batas_Kabupaten_Kemendagri_2024.shp"  # will fall back to .dbf

    df_kab, gdf_kab = generate_kabupaten_data(df_prov, shp)
    df_kab.to_csv(os.path.join(args.out_dir, "kabupaten_data.csv"), index=False)

    # Province scores (mean of kabupaten)
    df_scores = (
        df_kab.groupby(["province", "province_folder", "year"])
        .agg(credit_score=("credit_score", "mean"),
             tmax_annual=("tmax_annual", "mean"),
             ppt_annual=("ppt_annual", "mean"),
             tmean_annual=("tmean_annual", "mean"))
        .reset_index()
    )
    df_scores["credit_score"] = df_scores["credit_score"].round(0).astype(int)
    df_scores.to_csv(os.path.join(args.out_dir, "province_scores.csv"), index=False)

    print("Creating GeoJSON…")
    make_geojson(df_kab, gdf_kab, prov_bounds, args.out_dir)

    print(f"\n✅ Done! {len(df_kab)} kabupaten records, {len(df_prov)} province records.")
    print(f"   Credit score range: {df_kab['credit_score'].min()}–{df_kab['credit_score'].max()}")
    print(f"   Output: {args.out_dir}/")


if __name__ == "__main__":
    main()
