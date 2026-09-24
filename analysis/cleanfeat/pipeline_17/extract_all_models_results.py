"""Extract test accuracy/AUC/F1 for all 13 models from best_models.pkl"""
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.stats import beta as beta_dist

def wilson_ci(p_hat, n, alpha=0.05):
    z = 1.959963984540054  # z_{0.975}
    denom = 1 + z**2/n
    centre = (p_hat + z**2/(2*n)) / denom
    half = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denom
    return centre - half, centre + half

def evaluate_models(config_dir, X_test, y_test):
    pkl = joblib.load(config_dir / "best_models.pkl")
    rows = []
    for name, model in pkl.items():
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        n = len(y_test)
        lo, hi = wilson_ci(acc, n)
        rows.append({
            "model": name, "test_accuracy": acc, "test_ci_low": lo, "test_ci_high": hi,
            "test_auc": auc, "test_f1": f1, "n_test": n
        })
    return pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)

# Need test set --- adjust path if different
# Search known locations:
base = Path.home() / "ns3/Final_Project_NS3-master/strict_observable_v2"
for split_dir_name in ["data_splits", "splits", "processed_data", "splits_static"]:
    p = base / split_dir_name
    if p.exists():
        print(f"Found splits dir: {p}")
        for f in p.iterdir():
            print(f"  {f.name}")
        break
else:
    print("No splits dir found; will need to locate X_test/y_test manually.")
    print("Check what's in base directory:")
    for f in sorted(base.iterdir())[:30]:
        print(f"  {f.name}")
