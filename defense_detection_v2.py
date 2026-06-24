#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
Binary Defense Detection Pipeline (v2 - strict observability)
========================================================================

in OLSR-based Mobile Ad Hoc Networks (MANETs).

VERSION 2 NOTES (strict observability):
This is a strict-observability variant of defense_detection.py. The raw ns-3 CSVs
are READ in full, but the following are NOT used anywhere in the pipeline because a
passive adversary cannot derive them from observed wireless transmissions.

Removed base metrics (read from the CSV, used nowhere — appear in no feature list):
  - HelloMessageRate, ControlPacketRate         - HELLO is 1-hop only, not flooded
  - AverageRoutingTableSize                      - needs a HELLO-derived neighbor graph
  - TotalEnergyConsumption, EnergyEfficiency     - hardware-dependent
  - MACDropRateAvg/Max/Std, MACDropPacketRate    - MAC-layer internal counters
  - AverageNeighborCount, MprNeighborRatio       - not encoded in OLSR messages
  - AverageNodeSpeed, NodeSpeedStd               - physical mobility, not observable

Removed engineered features (they depended on the removed base metrics):
  - Control_Data_Ratio, HELLO_TC_Ratio, Energy_Per_Bit, Total_Overhead,
    MPR_Load, MAC_Drop_Impact
  - Interaction pairs (HELLO,TC), (DataPacketRate,ControlPacketRate),
    (AverageMprCount,AverageNeighborCount)
  - CDR and Total_Traffic are redefined without HELLO.

Run from inside strict_observable_v2/ working directory.
Default results directory: ./results/ (created automatically if missing).

Features:
- Advanced network feature engineering (domain-specific)
- Multi-criteria feature selection (MI, F-stat, RF, ET)
- Data augmentation (SMOTE variants)
- Diverse ensemble of classifiers with threshold optimization
- Calibrated probability predictions
- Group-based train/test splitting to prevent data leakage

========================================================================
"""
import os
# Avoid OpenMP shared memory issues in constrained environments
os.environ.setdefault("KMP_DISABLE_SHARED_MEM", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import glob
import warnings
import pickle
import datetime
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew, kurtosis, entropy

# Core ML
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV, GroupShuffleSplit
)
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PowerTransformer,
    QuantileTransformer, MinMaxScaler
)
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, f_classif, chi2,
    RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

import joblib
warnings.filterwarnings("ignore")

# Limit CPU usage to half of available cores to prevent freezing
import os
MAX_JOBS = int(os.environ.get("MAX_JOBS", max(1, os.cpu_count() // 2)))

# Check GPU availability
def check_gpu():
    """Check if GPU is available for acceleration"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("[GPU] CUDA not available, using CPU")
            return False
    except:
        print("[GPU] PyTorch not available, using CPU")
        return False

HAS_GPU = check_gpu()

# Gradient Boosting
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False
    print("[WARNING] XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except:
    HAS_LGBM = False
    print("[WARNING] LightGBM not available")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False
    print("[WARNING] CatBoost not available")

# Data Augmentation
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
    HAS_IMBLEARN = True
except:
    HAS_IMBLEARN = False
    print("[WARNING] imbalanced-learn not available")


@dataclass
class Config:
    """Configuration for defense detection pipeline"""

    data_root: str = "./simulations/features_static/"
    results_dir: str = "./results"

    random_state: int = 42
    test_size: float = 0.20  # Smaller test = more training data

    # Feature engineering
    use_temporal_features: bool = False  # DISABLED: Causes data leakage!
    use_advanced_network_features: bool = True  # Domain expertise
    use_statistical_features: bool = True
    use_interaction_features: bool = True
    max_interaction_order: int = 3  # Up to 3-way interactions
    observable_only: bool = False  # Use only metrics observable by passive listening
    report_only: bool = False  # Use only metrics listed in the report

    # Feature selection
    n_features_target: int = 150
    use_multiple_selection_methods: bool = True

    # Splitting
    group_split_by_file_source: bool = True
    limit_runs: Optional[int] = None  # Limit unique file_source runs (when grouping)
    train_size: float = 0.8  # Used when validation is disabled
    use_validation: bool = True

    # Data augmentation
    use_aggressive_augmentation: bool = True
    augmentation_strategies: List[str] = None  # Will use multiple
    target_balance_ratio: float = 1.0

    # Models
    use_diverse_ensemble: bool = True
    n_ensemble_models: int = 15  # More diversity
    safe_mode: bool = False  # Stability-first mode: avoids GPU/OMP-heavy models

    # Optimization
    optimize_thresholds: bool = True
    use_calibration: bool = True
    use_nested_cv: bool = True
    cv_folds: int = 10

    # Verbosity
    verbose: int = 1
    force_visuals: bool = False  # Allow visualizations even in safe mode

    def __post_init__(self):
        if self.augmentation_strategies is None:
            self.augmentation_strategies = ['smote', 'borderline', 'svmsmote']
        if os.environ.get("DEFENSE_DETECT_SAFE_MODE", "").strip().lower() in {"1", "true", "yes"}:
            self.safe_mode = True
        if os.environ.get("DEFENSE_DETECT_FORCE_VISUALS", "").strip().lower() in {"1", "true", "yes"}:
            self.force_visuals = True


class DefenseDetector:
    """ML-based defense detection pipeline"""

    REPORT_METRICS = [
        "PacketDeliveryRatio", "PacketLossRatio", "AverageEndToEndDelay", "AverageJitter",
        "Throughput", "AverageHopCount",
        "NormalizedRoutingLoad", "RoutingOverheadRatio", "AverageAdvertisedLinksPerTCMessage",
        "TcMessageRate", "MidMessageRate",
        "HnaMessageRate", "DataPacketRate", "CDR", "TDR"
    ]

    OBSERVABLE_METRICS = [
        "PacketDeliveryRatio", "PacketLossRatio", "AverageEndToEndDelay", "AverageJitter",
        "Throughput", "AverageHopCount",
        "TcMessageRate", "MidMessageRate",
        "HnaMessageRate", "DataPacketRate", "CDR", "TDR"
    ]

    METRICS = [
        # Control-plane metrics observable from OLSR messages flooded across the network
        # (TC, MID, HNA are flooded by MPRs; HELLO is NOT flooded — only 1-hop)
        "TcMessageRate", "MidMessageRate", "HnaMessageRate",
        "AverageAdvertisedLinksPerTCMessage",
        "NormalizedRoutingLoad", "RoutingOverheadRatio", "RoutingOverheadBytesRatio",
        # Data-plane metrics (observable from traffic)
        "PacketDeliveryRatio", "PacketLossRatio", "AverageEndToEndDelay", "AverageJitter",
        "Throughput", "AverageHopCount", "DataPacketRate", "RxTxPacketRatio",
        "FlowCount", "AvgFlowDuration", "FlowDurationStd", "AvgFlowThroughput",
        "AvgFlowDelay", "AvgFlowJitter", "AvgFlowLossRate", "FlowThroughputStd",
        "FlowDelayStd", "FlowJitterStd", "FlowLossRateStd",
        "AvgTxBytesPerFlow", "AvgRxBytesPerFlow", "AvgTxPacketsPerFlow", "AvgRxPacketsPerFlow",
        "AvgTxPacketSize", "AvgRxPacketSize",
        # Inferable from TC messages (flooded across network)
        "AverageMprCount",
    ]

    def __init__(self, config: Config):
        self.config = config
        self.random_state = config.random_state
        np.random.seed(self.random_state)

        if config.report_only:
            self.metrics = self.REPORT_METRICS
        elif config.observable_only:
            self.metrics = self.OBSERVABLE_METRICS
        else:
            self.metrics = self.METRICS
        self.scaler = None
        self.feature_names = None
        self.models = {}
        self.optimal_thresholds = {}

    def log(self, msg: str, level: int = 1):
        if self.config.verbose >= level:
            print(msg)

    # ================================================================
    # ENHANCED DATA LOADING - WITH TEMPORAL FEATURES!
    # ================================================================
    def _load_bundle_tall(self, bundle_dir: str) -> pd.DataFrame:
        """Colab fast-path: reconstruct the tall dataframe from the compact
        wide bundle (one row per measurement window) produced by
        make_colab_dataset.py, instead of globbing ~40k raw CSVs.

        Selected by setting the DCFM_DATA_BUNDLE environment variable to the
        bundle directory (containing wide_static.* and wide_mobile.*). The
        frame is melt-expanded so preprocess_data_enhanced works unchanged.
        """
        s = self.config.data_root.lower()
        if ("mobile" in s) == ("static" in s):
            raise ValueError(
                f"Cannot tell static vs mobile from data_root={self.config.data_root!r}. "
                "Pass a path containing exactly one of 'static' / 'mobile'.")
        cfg = "mobile" if "mobile" in s else "static"
        candidates = [
            os.path.join(bundle_dir, f"wide_{cfg}.parquet"),
            os.path.join(bundle_dir, f"wide_{cfg}.csv.gz"),
            os.path.join(bundle_dir, f"wide_{cfg}.csv"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(
                f"DCFM_DATA_BUNDLE={bundle_dir} but no wide_{cfg}.* found there")
        self.log(f"\n[1/14] Loading data from bundle: {os.path.basename(path)}")
        wide = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        meta = ["scenario", "file_source", "defense_active",
                "_Duration", "_StartTime", "_EndTime"]
        metric_cols = [c for c in wide.columns if c not in meta]
        tall = wide.melt(
            id_vars=[c for c in meta if c in wide.columns],
            value_vars=metric_cols, var_name="Metric", value_name="Value",
        )
        tall = tall.rename(columns={"_Duration": "Duration",
                                    "_StartTime": "StartTime",
                                    "_EndTime": "EndTime"})
        tall["Scenario"] = tall["scenario"]
        self.log(f"  Reconstructed tall frame: {len(tall)} rows from {len(wide)} windows")
        return tall

    def load_simulation_data_enhanced(self) -> pd.DataFrame:
        """Load data INCLUDING temporal features (StartTime, EndTime, Duration)"""
        self.log("\n" + "="*70)
        self.log("Defense Detection Pipeline")
        self.log("="*70)

        # Colab/git fast-path: load the compact wide bundle instead of the raw
        # per-window CSVs when DCFM_DATA_BUNDLE is set (see make_colab_dataset.py).
        bundle = os.environ.get("DCFM_DATA_BUNDLE", "").strip()
        if bundle:
            return self._load_bundle_tall(bundle)

        self.log("\n[1/14] Loading data with TEMPORAL features...")

        scenarios = {
            "baseline": 0,
            "attack_only": 0,
            "defense_only": 1,
            "defense_vs_attack": 1
        }

        parts = []
        for scenario, label in scenarios.items():
            scenario_dir = os.path.join(self.config.data_root, scenario)
            if not os.path.exists(scenario_dir):
                continue

            csv_files = glob.glob(os.path.join(scenario_dir, "*.csv"))
            self.log(f"  {scenario}: {len(csv_files)} files", level=2)

            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path)

                    # Check if temporal columns exist
                    if "StartTime" in df.columns and "EndTime" in df.columns:
                        df["defense_active"] = label
                        df["scenario"] = scenario
                        df["file_source"] = os.path.basename(csv_path)
                        parts.append(df)
                    elif "Metric" in df.columns and "Value" in df.columns:
                        df["defense_active"] = label
                        df["scenario"] = scenario
                        df["file_source"] = os.path.basename(csv_path)
                        parts.append(df)
                except Exception as e:
                    continue

        if not parts:
            raise ValueError(f"No CSV files found in {self.config.data_root}")

        tall_df = pd.concat(parts, ignore_index=True)
        self.log(f"  Loaded {len(tall_df)} records from {len(parts)} files")

        return tall_df

    def preprocess_data_enhanced(self, tall_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Convert to wide format WITH temporal features"""
        self.log("\n[2/14] Preprocessing with temporal extraction...")

        # Group by file to aggregate
        grouped = tall_df.groupby(["scenario", "file_source"])

        features_list = []
        labels_list = []
        group_list = []

        for (scenario, file_source), group in grouped:
            feature_row = {}

            # Extract metric values
            for metric in self.metrics:
                metric_data = group[group["Metric"] == metric]
                if not metric_data.empty:
                    feature_row[metric] = metric_data["Value"].mean()
                else:
                    feature_row[metric] = 0.0

            # תמיד חלץ Duration לצורך חישוב CDR
            if "Duration" in group.columns:
                feature_row["_measurement_duration"] = group["Duration"].mean()
            elif "StartTime" in group.columns and "EndTime" in group.columns:
                feature_row["_measurement_duration"] = (group["EndTime"] - group["StartTime"]).mean()
            else:
                feature_row["_measurement_duration"] = 40.0  # fallback
    
            # Extract temporal features
            if self.config.use_temporal_features and "StartTime" in group.columns:
                feature_row["StartTime_mean"] = group["StartTime"].mean()
                feature_row["EndTime_mean"] = group["EndTime"].mean()
                feature_row["Duration_mean"] = group["Duration"].mean()
                feature_row["Duration_std"] = group["Duration"].std() if len(group) > 1 else 0
                feature_row["Duration_max"] = group["Duration"].max()
                feature_row["Duration_min"] = group["Duration"].min()
                feature_row["TimeRange"] = group["EndTime"].max() - group["StartTime"].min()

            # Label
            label = group["defense_active"].iloc[0]

            features_list.append(feature_row)
            labels_list.append(label)
            #group_list.append(f"{scenario}_{file_source}")
            group_list.append(file_source)
            
        X = pd.DataFrame(features_list).fillna(0.0)
        y = pd.Series(labels_list, dtype=int)

        self.log(f"  Shape: {X.shape}")
        self.log(f"  Classes: {y.value_counts().to_dict()}")

        return X, y, group_list

    # ================================================================
    # ADVANCED FEATURE ENGINEERING
    # ================================================================
    def engineer_advanced_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create advanced network security features"""
        self.log("\n[3/14] Engineering advanced features...")

        X_eng = X.copy()
        eps = 1e-10
        zeros = pd.Series(0.0, index=X.index)

        def col(name: str):
            return X[name] if name in X.columns else zeros

        # 1. Network Performance Indicators
        if self.config.use_advanced_network_features:
            # Quality of Service metrics
            X_eng["QoS_Score"] = (
                col("PacketDeliveryRatio") * 0.4 +
                (1 - col("PacketLossRatio")) * 0.3 +
                (1 / (col("AverageEndToEndDelay") + eps)) * 0.15 +
                (1 / (col("AverageJitter") + eps)) * 0.15
            )

            # Network efficiency
            X_eng["Network_Efficiency"] = col("Throughput") / (col("RoutingOverheadRatio") + eps)
            X_eng["Delivery_Efficiency"] = col("PacketDeliveryRatio") / (col("NormalizedRoutingLoad") + eps)

            # CDR: Topology Control rate normalized by data packet rate
            measurement_duration = col("_measurement_duration")
            data_packet_rate = (col("AvgTxPacketsPerFlow") * col("FlowCount")) / measurement_duration
            X_eng["CDR"] = col("TcMessageRate") / (data_packet_rate + eps)
            # TDR: Topology Dissemination Rate = TC rate * avg links per TC
            X_eng["TDR"] = (col("TcMessageRate") * col("AverageAdvertisedLinksPerTCMessage"))

            # Overhead indicators
            X_eng["Overhead_Per_Hop"] = col("RoutingOverheadRatio") / (col("AverageHopCount") + eps)

            # Traffic intensity
            X_eng["Total_Traffic"] = (
                col("TcMessageRate") +
                col("MidMessageRate") + col("HnaMessageRate") +
                col("DataPacketRate")
            )

            # Delay metrics
            X_eng["Delay_Per_Hop"] = col("AverageEndToEndDelay") / (col("AverageHopCount") + eps)
            X_eng["Jitter_Delay_Ratio"] = col("AverageJitter") / (col("AverageEndToEndDelay") + eps)

            # Defense indicators (anomaly patterns)
            X_eng["Routing_Anomaly"] = col("NormalizedRoutingLoad") * col("RoutingOverheadRatio")
            X_eng["Traffic_Anomaly"] = X_eng["Total_Traffic"] * (1 - col("PacketDeliveryRatio"))
            X_eng["Performance_Degradation"] = col("AverageEndToEndDelay") * col("PacketLossRatio")

        # 2. Statistical transformations
        if self.config.use_statistical_features:
            # Log transformations
            for col in self.metrics:
                if col in X.columns and (X[col] >= 0).all():
                    X_eng[f"log1p_{col}"] = np.log1p(X[col])
                    X_eng[f"sqrt_{col}"] = np.sqrt(X[col] + eps)

            # Power transformations for key metrics
            key_metrics = ["Throughput", "AverageEndToEndDelay", "AverageJitter", "RoutingOverheadRatio"]
            for col in key_metrics:
                if col in X.columns:
                    X_eng[f"{col}_squared"] = X[col] ** 2
                    X_eng[f"{col}_cubed"] = X[col] ** 3

            # Row-wise statistics
            numeric_cols = X[self.metrics].select_dtypes(include=[np.number]).columns
            X_eng["row_mean"] = X[numeric_cols].mean(axis=1)
            X_eng["row_std"] = X[numeric_cols].std(axis=1)
            X_eng["row_median"] = X[numeric_cols].median(axis=1)
            X_eng["row_max"] = X[numeric_cols].max(axis=1)
            X_eng["row_min"] = X[numeric_cols].min(axis=1)
            X_eng["row_range"] = X_eng["row_max"] - X_eng["row_min"]
            X_eng["row_cv"] = X_eng["row_std"] / (X_eng["row_mean"] + eps)
            X_eng["row_skew"] = X[numeric_cols].apply(lambda x: skew(x.values), axis=1)
            X_eng["row_kurtosis"] = X[numeric_cols].apply(lambda x: kurtosis(x.values), axis=1)

            # Percentiles
            X_eng["row_q25"] = X[numeric_cols].quantile(0.25, axis=1)
            X_eng["row_q75"] = X[numeric_cols].quantile(0.75, axis=1)
            X_eng["row_iqr"] = X_eng["row_q75"] - X_eng["row_q25"]

        # 3. Interaction features
        if self.config.use_interaction_features:
            # Critical 2-way interactions
            critical_pairs = [
                ("Throughput", "AverageEndToEndDelay"),
                ("PacketDeliveryRatio", "PacketLossRatio"),
                ("RoutingOverheadRatio", "NormalizedRoutingLoad"),
                ("AverageEndToEndDelay", "AverageJitter"),
                ("Throughput", "RoutingOverheadRatio"),
            ]

            for feat1, feat2 in critical_pairs:
                if feat1 in X.columns and feat2 in X.columns:
                    X_eng[f"interact_{feat1}_{feat2}"] = X[feat1] * X[feat2]
                    X_eng[f"ratio_{feat1}_{feat2}"] = X[feat1] / (X[feat2] + eps)

        # 4. Temporal features (if available)
        if "Duration_mean" in X.columns:
            X_eng["Duration_CV"] = X["Duration_std"] / (X["Duration_mean"] + eps)
            X_eng["Duration_Range"] = X["Duration_max"] - X["Duration_min"]
            X_eng["Time_Efficiency"] = X["Throughput"] / (X["Duration_mean"] + eps)

        # Clean up
        X_eng = X_eng.replace([np.inf, -np.inf], 0).fillna(0)
        X_eng = X_eng.drop(columns=["_measurement_duration"], errors="ignore")  # ← כאן

        self.log(f"  Created {X_eng.shape[1]} features (from {X.shape[1]})")

        return X_eng

    # ================================================================
    # INTELLIGENT FEATURE SELECTION
    # ================================================================
    def select_features_intelligent(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                    y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Multi-criteria feature selection"""
        self.log("\n[4/14] Intelligent feature selection...")

        # Remove zero-variance features
        var_selector = VarianceThreshold(threshold=0.01)
        X_train_var = var_selector.fit_transform(X_train)
        X_test_var = var_selector.transform(X_test)

        remaining_cols = X_train.columns[var_selector.get_support()]
        X_train_var = pd.DataFrame(X_train_var, columns=remaining_cols, index=X_train.index)
        X_test_var = pd.DataFrame(X_test_var, columns=remaining_cols, index=X_test.index)

        # Remove highly correlated features (|r| > 0.95)
        corr_matrix = X_train_var.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        X_train_var = X_train_var.drop(columns=to_drop)
        X_test_var = X_test_var.drop(columns=to_drop)
        remaining_cols = X_train_var.columns
        self.log(f"  Removed {len(to_drop)} highly correlated features (|r|>0.95), {len(remaining_cols)} remaining")

        # Multiple selection methods
        n_features = min(self.config.n_features_target, X_train_var.shape[1])

        # Method 1: Mutual Information
        mi_selector = SelectKBest(mutual_info_classif, k=n_features)
        mi_selector.fit(X_train_var, y_train)
        mi_scores = pd.Series(mi_selector.scores_, index=X_train_var.columns)

        # Method 2: F-statistic
        f_selector = SelectKBest(f_classif, k=n_features)
        f_selector.fit(X_train_var, y_train)
        f_scores = pd.Series(f_selector.scores_, index=X_train_var.columns)

        # Method 3: Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=MAX_JOBS)
        rf.fit(X_train_var, y_train)
        rf_scores = pd.Series(rf.feature_importances_, index=X_train_var.columns)

        # Method 4: ExtraTrees importance
        et = ExtraTreesClassifier(n_estimators=100, random_state=self.random_state, n_jobs=MAX_JOBS)
        et.fit(X_train_var, y_train)
        et_scores = pd.Series(et.feature_importances_, index=X_train_var.columns)

        # Combine using rank aggregation
        mi_ranks = mi_scores.rank(ascending=False)
        f_ranks = f_scores.rank(ascending=False)
        rf_ranks = rf_scores.rank(ascending=False)
        et_ranks = et_scores.rank(ascending=False)

        combined_ranks = (mi_ranks + f_ranks + rf_ranks + et_ranks) / 4

        # Select top features
        selected_features = combined_ranks.nsmallest(n_features).index.tolist()

        self.feature_names = selected_features

        X_train_final = X_train_var[selected_features]
        X_test_final = X_test_var[selected_features]

        self.log(f"  Selected {len(selected_features)} features from {X_train.shape[1]}")

        return X_train_final, X_test_final

    # ================================================================
    # AGGRESSIVE DATA AUGMENTATION
    # ================================================================
    def augment_data_aggressive(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Use multiple augmentation strategies"""
        self.log("\n[5/14] Aggressive data augmentation...")

        if not self.config.use_aggressive_augmentation or not HAS_IMBLEARN:
            return X_train, y_train

        original_size = len(X_train)

        # Strategy 1: SMOTE
        try:
            smote = SMOTE(sampling_strategy=self.config.target_balance_ratio,
                         k_neighbors=5, random_state=self.random_state)
            X_smote, y_smote = smote.fit_resample(X_train, y_train)
        except:
            X_smote, y_smote = X_train.values, y_train.values

        # Strategy 2: Borderline SMOTE
        try:
            borderline = BorderlineSMOTE(sampling_strategy=self.config.target_balance_ratio,
                                        k_neighbors=5, random_state=self.random_state)
            X_border, y_border = borderline.fit_resample(X_train, y_train)
        except:
            X_border, y_border = X_smote, y_smote

        # Strategy 3: SVM SMOTE
        try:
            svm_smote = SVMSMOTE(sampling_strategy=self.config.target_balance_ratio,
                                k_neighbors=5, random_state=self.random_state)
            X_svm, y_svm = svm_smote.fit_resample(X_train, y_train)
        except:
            X_svm, y_svm = X_border, y_border

        # Combine all augmented data
        X_combined = np.vstack([X_smote, X_border, X_svm])
        y_combined = np.hstack([y_smote, y_border, y_svm])

        # Remove duplicates
        combined_df = pd.DataFrame(X_combined, columns=X_train.columns)
        combined_df['label'] = y_combined
        combined_df = combined_df.drop_duplicates()

        y_aug = combined_df['label'].astype(int)
        X_aug = combined_df.drop('label', axis=1)

        self.log(f"  Augmented: {original_size} → {len(X_aug)} samples")

        return X_aug, y_aug

    # ================================================================
    # DIVERSE ENSEMBLE TRAINING
    # ================================================================
    def train_diverse_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train highly diverse ensemble"""
        self.log("\n[6/14] Training diverse ensemble (15 models)...")

        models = {}
        n_jobs = 1 if self.config.safe_mode else MAX_JOBS

        if self.config.safe_mode:
            self.log("  Safe mode enabled: disabling GPU/OMP-heavy models and forcing single-threaded training", level=2)

        # Gradient Boosting Models
        if HAS_LGBM and not self.config.safe_mode:
            self.log("  Training LightGBM variants...", level=2)
            # Use GPU if available
            lgbm_device = 'gpu' if HAS_GPU else 'cpu'

            # Variant 1: Balanced
            models['LGBM_balanced'] = LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=8,
                num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
                device=lgbm_device,
                random_state=self.random_state, n_jobs=n_jobs, verbose=-1
            )
            # Variant 2: Deep
            models['LGBM_deep'] = LGBMClassifier(
                n_estimators=800, learning_rate=0.03, max_depth=12,
                num_leaves=127, subsample=0.7, colsample_bytree=0.7,
                min_child_samples=10, reg_alpha=0.5, reg_lambda=0.5,
                device=lgbm_device,
                random_state=self.random_state + 1, n_jobs=n_jobs, verbose=-1
            )
            if HAS_GPU:
                self.log("    Using GPU acceleration for LightGBM", level=2)

        if HAS_XGB and not self.config.safe_mode:
            self.log("  Training XGBoost variants...", level=2)
            # Use GPU if available
            xgb_device = 'cuda' if HAS_GPU else 'cpu'
            xgb_tree_method = 'gpu_hist' if HAS_GPU else 'hist'

            models['XGB_balanced'] = XGBClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=7,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                gamma=0.1, reg_alpha=0.1, reg_lambda=0.1,
                tree_method=xgb_tree_method, device=xgb_device,
                random_state=self.random_state, n_jobs=n_jobs, verbosity=0
            )
            models['XGB_conservative'] = XGBClassifier(
                n_estimators=1200, learning_rate=0.02, max_depth=5,
                subsample=0.9, colsample_bytree=0.9, min_child_weight=10,
                gamma=0.5, reg_alpha=1.0, reg_lambda=1.0,
                tree_method=xgb_tree_method, device=xgb_device,
                random_state=self.random_state + 2, n_jobs=n_jobs, verbosity=0
            )
            if HAS_GPU:
                self.log("    Using GPU acceleration for XGBoost", level=2)

        if HAS_CATBOOST and not self.config.safe_mode:
            self.log("  Training CatBoost variants...", level=2)
            # NOTE: CatBoost GPU doesn't work well with parallel sklearn ensembles
            # Using CPU mode to avoid "device already requested" errors
            cb_task_type = 'CPU'

            models['CatBoost_balanced'] = CatBoostClassifier(
                iterations=1000, learning_rate=0.05, depth=7,
                l2_leaf_reg=3, border_count=128,
                task_type=cb_task_type,
                random_state=self.random_state, verbose=0
            )
            models['CatBoost_deep'] = CatBoostClassifier(
                iterations=800, learning_rate=0.03, depth=9,
                l2_leaf_reg=5, border_count=254,
                task_type=cb_task_type,
                random_state=self.random_state + 3, verbose=0
            )
            self.log("    Using CPU for CatBoost (GPU conflicts with parallel ensembles)", level=2)

        # Tree-based models
        self.log("  Training tree ensembles...", level=2)
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt',
            random_state=self.random_state, n_jobs=n_jobs
        )

        models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=500, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt',
            random_state=self.random_state + 4, n_jobs=n_jobs
        )

        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, max_features='sqrt',
            random_state=self.random_state
        )

        def _make_bagging(estimator, **kwargs):
            try:
                return BaggingClassifier(estimator=estimator, **kwargs)
            except TypeError:
                return BaggingClassifier(base_estimator=estimator, **kwargs)

        def _make_adaboost(estimator, **kwargs):
            try:
                return AdaBoostClassifier(estimator=estimator, **kwargs)
            except TypeError:
                return AdaBoostClassifier(base_estimator=estimator, **kwargs)

        # Bagging ensembles
        models['Bagging_RF'] = _make_bagging(
            DecisionTreeClassifier(max_depth=15),
            n_estimators=100, max_samples=0.8, max_features=0.8,
            random_state=self.random_state, n_jobs=n_jobs
        )

        models['Bagging_ET'] = _make_bagging(
            ExtraTreesClassifier(n_estimators=10, max_depth=10),
            n_estimators=50, max_samples=0.8,
            random_state=self.random_state + 5, n_jobs=n_jobs
        )

        # Boosting
        models['AdaBoost'] = _make_adaboost(
            DecisionTreeClassifier(max_depth=5),
            n_estimators=200, learning_rate=0.1,
            random_state=self.random_state
        )

        # Linear models
        logreg_solver = 'liblinear' if self.config.safe_mode else 'lbfgs'
        models['LogisticRegression'] = LogisticRegression(
            C=1.0, penalty='l2', solver=logreg_solver,
            random_state=self.random_state, max_iter=1000, n_jobs=n_jobs
        )

        if not self.config.safe_mode:
            models['Ridge'] = RidgeClassifier(
                alpha=1.0, random_state=self.random_state
            )

        # SVM
        models['SVM_rbf'] = SVC(
            C=1.0, kernel='rbf', gamma='scale', probability=True,
            random_state=self.random_state
        )

        # Train all models
        for name, model in models.items():
            self.log(f"    Training {name}...", level=2)
            model.fit(X_train, y_train)

        self.log(f"  Trained {len(models)} diverse models")

        return models

    # ================================================================
    # THRESHOLD OPTIMIZATION
    # ================================================================
    def optimize_thresholds(self, models: Dict[str, Any], X_val: pd.DataFrame,
                           y_val: pd.Series) -> Dict[str, float]:
        """Find optimal decision threshold for each model"""
        self.log("\n[7/14] Optimizing decision thresholds...")

        thresholds = {}

        for name, model in models.items():
            try:
                y_prob = model.predict_proba(X_val)[:, 1]

                # Try thresholds from 0.1 to 0.9
                best_threshold = 0.5
                best_accuracy = 0

                for threshold in np.arange(0.1, 0.9, 0.01):
                    y_pred = (y_prob >= threshold).astype(int)
                    accuracy = accuracy_score(y_val, y_pred)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold

                thresholds[name] = best_threshold
                self.log(f"  {name}: threshold={best_threshold:.3f}, acc={best_accuracy:.4f}", level=2)
            except:
                thresholds[name] = 0.5

        return thresholds

    # ================================================================
    # CALIBRATED PREDICTIONS
    # ================================================================
    def calibrate_models(self, models: Dict[str, Any], X_cal: pd.DataFrame,
                        y_cal: pd.Series) -> Dict[str, Any]:
        """Calibrate probability predictions"""
        self.log("\n[8/14] Calibrating probability predictions...")

        calibrated_models = {}

        for name, model in models.items():
            try:
                self.log(f"  Calibrating {name}...", level=2)
                calibrated = CalibratedClassifierCV(model, cv=5, method='isotonic')
                calibrated.fit(X_cal, y_cal)
                calibrated_models[name] = calibrated
            except:
                calibrated_models[name] = model

        self.log("  Calibration complete")

        return calibrated_models

    # ================================================================
    # SUPER ENSEMBLE
    # ================================================================
    def create_super_ensemble(self, models: Dict[str, Any], X_train: pd.DataFrame,
                             y_train: pd.Series) -> Any:
        """Create stacked ensemble from all models"""
        self.log("\n[9/14] Building super ensemble...")

        # Base estimators
        base_estimators = [(name, model) for name, model in list(models.items())[:10]]

        # Meta-learner
        meta_learner = LogisticRegression(
            C=1.0, random_state=self.random_state, max_iter=1000
        )

        # Stacking
        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=MAX_JOBS,
            verbose=0
        )

        self.log("  Training stacking ensemble...")
        stacking.fit(X_train, y_train)

        return stacking

    # ================================================================
    # VOTING ENSEMBLE
    # ================================================================
    def create_voting_ensemble(self, models: Dict[str, Any]) -> Any:
        """Create soft voting ensemble"""
        self.log("\n[10/14] Building voting ensemble...")

        # Use top models
        estimators = [(name, model) for name, model in list(models.items())[:10]]

        voting = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=MAX_JOBS
        )

        return voting

    # ================================================================
    # EVALUATION
    # ================================================================
    def evaluate_model(self, name: str, model: Any, X_test: pd.DataFrame,
                      y_test: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
        """Comprehensive evaluation"""

        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
        except:
            y_pred = model.predict(X_test)
            y_prob = np.zeros(len(y_test)) + 0.5

        return {
            'Model': name,
            'Threshold': threshold,
            'AUC': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'Kappa': cohen_kappa_score(y_test, y_pred)
        }

    # ================================================================
    # VISUALIZATION
    # ================================================================
    def create_visualizations(self, results_df: pd.DataFrame, X_test: pd.DataFrame,
                              y_test: pd.Series, best_model: Any):
        """Create comprehensive visualizations"""
        if self.config.safe_mode and not self.config.force_visuals:
            self.log("\n[12/14] Safe mode: skipping visualizations to avoid OpenMP shared memory issues...")
            return
        self.log("\n[12/14] Creating visualizations...")

        os.makedirs(self.config.results_dir, exist_ok=True)

        # Model comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        results_sorted = results_df.sort_values('Accuracy', ascending=True).tail(15)

        bars = ax.barh(results_sorted['Model'], results_sorted['Accuracy'], color='steelblue')

        # Color code by performance
        for i, (idx, row) in enumerate(results_sorted.iterrows()):
            if row['Accuracy'] >= 0.90:
                bars[i].set_color('green')
            elif row['Accuracy'] >= 0.85:
                bars[i].set_color('orange')

        ax.axvline(x=0.90, color='red', linestyle='--', linewidth=2, label='90% Target')
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Confusion matrix for best model
        try:
            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix (Best Model)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.log(f"  Skipped confusion matrix: {e}", level=2)

        # Correlation between target and features (test split)
        try:
            corr_df = X_test.copy()
            corr_df['target'] = y_test.values
            corr = corr_df.corr(numeric_only=True)['target'].drop('target')
            corr_abs = corr.abs().sort_values(ascending=False)
            top_n = min(20, len(corr_abs))
            top_corr = corr.loc[corr_abs.index[:top_n]]

            fig, ax = plt.subplots(figsize=(10, 6))
            top_corr.sort_values().plot(kind='barh', ax=ax, color='teal')
            ax.set_xlabel('Correlation with target')
            ax.set_title('Top Feature Correlations (Test Split)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.results_dir, 'feature_target_correlation.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.log(f"  Skipped correlation plot: {e}", level=2)

        self.log(f"  Saved visualizations to {self.config.results_dir}")

    # ================================================================
    # MAIN PIPELINE
    # ================================================================
    def run_full_pipeline(self):
        """Execute complete pipeline"""

        start_time = datetime.datetime.now()

        # Load data with temporal features
        tall_df = self.load_simulation_data_enhanced()
        X, y, groups = self.preprocess_data_enhanced(tall_df)

        # Feature engineering
        X_eng = self.engineer_advanced_features(X)

        # Optional: limit to N unique runs (file_source)
        if self.config.limit_runs is not None:
            if not self.config.group_split_by_file_source:
                raise ValueError("--limit-runs requires --group-by-file")
            unique_runs = sorted(set(groups))
            if self.config.limit_runs > len(unique_runs):
                raise ValueError(f"limit_runs={self.config.limit_runs} exceeds available runs={len(unique_runs)}")
            rng = np.random.RandomState(self.random_state)
            selected_runs = set(rng.choice(unique_runs, size=self.config.limit_runs, replace=False).tolist())
            mask = [g in selected_runs for g in groups]
            X_eng = X_eng.loc[mask]
            y = y.loc[mask]
            groups = [g for g in groups if g in selected_runs]

        # Split: 60% train, 20% validation, 20% test (default)
        if self.config.group_split_by_file_source:
            groups_arr = np.array(groups)
            if self.config.use_validation:
                splitter = GroupShuffleSplit(
                    n_splits=1, test_size=self.config.test_size, random_state=self.random_state
                )
                train_val_idx, test_idx = next(splitter.split(X_eng, y, groups_arr))
                X_temp = X_eng.iloc[train_val_idx]
                y_temp = y.iloc[train_val_idx]
                X_test = X_eng.iloc[test_idx]
                y_test = y.iloc[test_idx]

                groups_temp = groups_arr[train_val_idx]
                splitter_val = GroupShuffleSplit(
                    n_splits=1, test_size=0.25, random_state=self.random_state
                )
                train_idx, val_idx = next(splitter_val.split(X_temp, y_temp, groups_temp))
                X_train = X_temp.iloc[train_idx]
                y_train = y_temp.iloc[train_idx]
                X_val = X_temp.iloc[val_idx]
                y_val = y_temp.iloc[val_idx]
            else:
                splitter = GroupShuffleSplit(
                    n_splits=1, test_size=1.0 - self.config.train_size, random_state=self.random_state
                )
                train_idx, test_idx = next(splitter.split(X_eng, y, groups_arr))
                X_train = X_eng.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X_eng.iloc[test_idx]
                y_test = y.iloc[test_idx]
                X_val = None
                y_val = None
        else:
            if self.config.use_validation:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X_eng, y, test_size=self.config.test_size,
                    random_state=self.random_state, stratify=y
                )

                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.25,
                    random_state=self.random_state, stratify=y_temp
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_eng, y, test_size=1.0 - self.config.train_size,
                    random_state=self.random_state, stratify=y
                )
                X_val = None
                y_val = None

        # Scale
        self.scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns, index=X_train.index
        )
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns, index=X_val.index
            )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index
        )

        # Feature selection
        X_train_sel, X_val_sel = self.select_features_intelligent(
            X_train_scaled, X_val_scaled if X_val_scaled is not None else X_test_scaled, y_train
        )
        X_test_sel = X_test_scaled[self.feature_names]

        # Data augmentation
        X_train_aug, y_train_aug = self.augment_data_aggressive(X_train_sel, y_train)

        # Train diverse ensemble
        models = self.train_diverse_ensemble(X_train_aug, y_train_aug)

        # Optimize thresholds
        if self.config.use_validation and self.config.optimize_thresholds:
            self.optimal_thresholds = self.optimize_thresholds(models, X_val_sel, y_val)
        else:
            self.optimal_thresholds = {name: 0.5 for name in models.keys()}

        # Calibrate models
        if self.config.use_validation and self.config.use_calibration:
            models = self.calibrate_models(models, X_val_sel, y_val)

        # Create ensembles
        stacking_ensemble = self.create_super_ensemble(models, X_train_aug, y_train_aug)
        models['Stacking_Ensemble'] = stacking_ensemble
        self.optimal_thresholds['Stacking_Ensemble'] = 0.5

        # Evaluate all models
        self.log("\n[11/14] Evaluating all models on test set...")

        results = []
        for name, model in models.items():
            threshold = self.optimal_thresholds.get(name, 0.5)
            result = self.evaluate_model(name, model, X_test_sel, y_test, threshold)
            try:
                y_prob = model.predict_proba(X_test_sel)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)
            except Exception:
                y_pred = model.predict(X_test_sel)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            result["TN"] = tn
            result["FP"] = fp
            result["FN"] = fn
            result["TP"] = tp
            results.append(result)

        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

        # Save results
        self.log("\n[13/14] Saving results...")
        os.makedirs(self.config.results_dir, exist_ok=True)

        results_csv = os.path.join(self.config.results_dir, "results.csv")
        results_df.to_csv(results_csv, index=False)

        # Save best model
        best_model_name = results_df.iloc[0]['Model']
        best_model = models[best_model_name]

        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'threshold': self.optimal_thresholds[best_model_name],
            'all_models': models,
            'all_thresholds': self.optimal_thresholds
        }

        model_path = os.path.join(self.config.results_dir, f"best_model_{best_model_name}.pkl")
        joblib.dump(model_data, model_path)

        # Save confusion matrix for best model
        try:
            y_pred_best = best_model.predict(X_test_sel)
            cm = confusion_matrix(y_test, y_pred_best)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual_0", "Actual_1"],
                columns=["Pred_0", "Pred_1"]
            )
            cm_path = os.path.join(self.config.results_dir, "confusion_matrix.csv")
            cm_df.to_csv(cm_path, index=True)
        except Exception as e:
            self.log(f"  Skipped confusion matrix save: {e}", level=2)

        # Visualizations
        self.create_visualizations(results_df, X_test_sel, y_test, best_model)

        # Final report
        elapsed = datetime.datetime.now() - start_time

        self.log("\n" + "="*70)
        self.log("FINAL RESULTS")
        self.log("="*70)
        print(results_df.to_string(index=False))

        best_accuracy = results_df.iloc[0]['Accuracy']

        self.log("\n" + "="*70)
        self.log("SUMMARY")
        self.log("="*70)
        self.log(f"Best Model: {best_model_name}")
        self.log(f"Best Accuracy: {best_accuracy*100:.2f}%")
        self.log(f"Optimal Threshold: {self.optimal_thresholds[best_model_name]:.3f}")
        self.log(f"Target Achieved: {'YES! ✓✓✓' if best_accuracy >= 0.90 else 'Not yet'}")
        self.log(f"Gap to 90%: {(0.90 - best_accuracy)*100:+.2f}%")
        self.log(f"Total Runtime: {elapsed}")
        self.log(f"Results saved to: {self.config.results_dir}")
        self.log("="*70 + "\n")

        return results_df, models


def build_config_from_args() -> Config:
    parser = argparse.ArgumentParser(description="Run defense detection pipeline.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Base input data directory (e.g., ./simulations/features_extended/).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory to write model results.",
    )
    parser.add_argument(
        "--group-by-file",
        action="store_true",
        help="Split by file_source so each simulation run stays in one split.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Limit to N unique file_source runs (only with --group-by-file).",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        help="Train split size when validation is disabled (default: 0.8).",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable validation split (use only train/test).",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable data augmentation (use raw training data only).",
    )
    parser.add_argument(
        "--use-temporal",
        action="store_true",
        help="Enable temporal features (StartTime/EndTime/Duration).",
    )
    parser.add_argument(
        "--observable-only",
        action="store_true",
        help="Use only metrics observable by passive listening.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Use only metrics listed in the project report.",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Stability-first mode (disable GPU/OMP-heavy models, force single-threaded training).",
    )
    parser.add_argument(
        "--force-visuals",
        action="store_true",
        help="Generate visualizations even in safe mode (may require OpenMP shared memory).",
    )
    args = parser.parse_args()

    config = Config()
    if args.data_root:
        config.data_root = args.data_root
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.group_by_file:
        config.group_split_by_file_source = True
    if args.limit_runs is not None:
        config.limit_runs = args.limit_runs
    if args.train_size is not None:
        config.train_size = args.train_size
    if args.no_validation:
        config.use_validation = False
    if args.no_augmentation:
        config.use_aggressive_augmentation = False
    if args.use_temporal:
        config.use_temporal_features = True
    if args.observable_only:
        config.observable_only = True
    if args.report_only:
        config.report_only = True
    if args.safe_mode:
        config.safe_mode = True
    if args.force_visuals:
        config.force_visuals = True

    return config


def main():
    """Main entry point"""

    config = build_config_from_args()
    detector = DefenseDetector(config)

    try:
        results, models = detector.run_full_pipeline()
        return results, models
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR!")
        print(f"{'='*70}")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        sys.exit(1)


if __name__ == "__main__":
    results, models = main()