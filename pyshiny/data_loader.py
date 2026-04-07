"""
data_loader.py
--------------
Loads preprocessed climate data, credit scores, and GeoJSON boundaries.
All data is cached at import time for fast reactive access.
"""

import json
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_json(name: str) -> dict:
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


# ── Tabular data ──
DF_KAB: pd.DataFrame = pd.read_csv(os.path.join(DATA_DIR, "kabupaten_data.csv"))
DF_PROV_CLIMATE: pd.DataFrame = pd.read_csv(os.path.join(DATA_DIR, "province_climate.csv"))
DF_PROV_SCORES: pd.DataFrame = pd.read_csv(os.path.join(DATA_DIR, "province_scores.csv"))

# ── GeoJSON boundaries ──
GEOJSON_KAB: dict = _load_json("kabupaten.geojson")
GEOJSON_PROV: dict = _load_json("provinces.geojson")

# ── Convenience lookups ──
YEARS: list[int] = sorted(DF_KAB["year"].unique())
PROVINCES: list[str] = sorted(DF_KAB["province"].unique())

# ── Build mapping: province display name <-> nama_prop (uppercase in GeoJSON) ──
_prov_to_namaprop = {}
_geo_prov_names = set()
for feat in GEOJSON_PROV["features"]:
    _geo_prov_names.add(feat["properties"]["nama_prop"])

for prov_display in PROVINCES:
    upper = prov_display.upper()
    if upper in _geo_prov_names:
        _prov_to_namaprop[prov_display] = upper
    elif prov_display == "DI Yogyakarta":
        _prov_to_namaprop[prov_display] = "DAERAH ISTIMEWA YOGYAKARTA"
    elif prov_display == "DKI Jakarta":
        _prov_to_namaprop[prov_display] = "DKI JAKARTA"
    elif prov_display == "Papua":
        _prov_to_namaprop[prov_display] = "P A P U A"
    else:
        for gn in _geo_prov_names:
            if upper.replace(" ", "") in gn.replace(" ", ""):
                _prov_to_namaprop[prov_display] = gn
                break

PROV_TO_NAMAPROP: dict[str, str] = _prov_to_namaprop
NAMAPROP_TO_PROV: dict[str, str] = {v: k for k, v in _prov_to_namaprop.items()}
