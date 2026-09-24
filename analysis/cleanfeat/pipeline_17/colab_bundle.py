#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colab_bundle.py
===============
Shared helper for the Colab fast-path. A few analysis scripts read the raw
per-window CSVs directly (their own glob) instead of going through
DefenseDetector.load_simulation_data_enhanced. This module lets them load the
compact wide bundle produced by make_colab_dataset.py instead, when the
DCFM_DATA_BUNDLE environment variable is set.

It reconstructs the same TALL dataframe those scripts expect (columns:
scenario, file_source, defense_active, Metric, Value, Duration, ...), so their
downstream to_wide()/groupby logic works unchanged.
"""
import os
import pandas as pd

_META = ["scenario", "file_source", "defense_active",
         "_Duration", "_StartTime", "_EndTime"]


def bundle_active() -> str:
    """Return the bundle dir if DCFM_DATA_BUNDLE is set and non-empty, else ''."""
    return os.environ.get("DCFM_DATA_BUNDLE", "").strip()


def detect_cfg(data_root: str) -> str:
    """Resolve 'static' or 'mobile' from a path string; error if ambiguous.

    Refuses to silently default — an unrecognizable data_root would otherwise
    load the wrong configuration without any warning.
    """
    s = str(data_root).lower()
    has_static, has_mobile = "static" in s, "mobile" in s
    if has_mobile and not has_static:
        return "mobile"
    if has_static and not has_mobile:
        return "static"
    raise ValueError(
        f"Cannot tell static vs mobile from data_root={data_root!r}. "
        "Pass a path containing exactly one of 'static' / 'mobile'.")


def find_wide(bundle_dir: str, data_root: str):
    """Pick wide_static.* or wide_mobile.* based on the data_root string."""
    cfg = detect_cfg(data_root)
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = os.path.join(bundle_dir, f"wide_{cfg}{ext}")
        if os.path.exists(p):
            return p, cfg
    raise FileNotFoundError(f"No wide_{cfg}.* under {bundle_dir}")


def load_bundle_tall(bundle_dir: str, data_root: str) -> pd.DataFrame:
    """Reconstruct the tall (Metric/Value) frame from the wide bundle."""
    path, _cfg = find_wide(bundle_dir, data_root)
    wide = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    metric_cols = [c for c in wide.columns if c not in _META]
    tall = wide.melt(
        id_vars=[c for c in _META if c in wide.columns],
        value_vars=metric_cols, var_name="Metric", value_name="Value",
    )
    tall = tall.rename(columns={"_Duration": "Duration",
                                "_StartTime": "StartTime",
                                "_EndTime": "EndTime"})
    tall["Scenario"] = tall["scenario"]
    return tall
