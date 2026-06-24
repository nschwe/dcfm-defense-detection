
import sys, os, json, time, traceback

# Set cwd explicitly to make sure imports work
WORKING_DIR = sys.argv[4]
os.chdir(WORKING_DIR)
sys.path.insert(0, WORKING_DIR)

# Match imports from unified_hp_search_v2
from defense_detection_v2 import Config
import unified_hp_search_v2 as u

C_VAL = float(sys.argv[1])
GAMMA_VAL = float(sys.argv[2])
OUTPUT_JSON = sys.argv[3]

# Choose max_iter based on C: higher C may need more iterations
# For C=10^6, allow up to 500k iterations (libsvm default is -1 = unlimited)
SVM_MAX_ITER = 500_000

result = {
    "C": C_VAL,
    "gamma": GAMMA_VAL,
    "status": "started",
    "timestamp": time.time(),
    "max_iter": SVM_MAX_ITER,
}

def save_result():
    """Write result to JSON. Called at every checkpoint."""
    try:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        print(f"[worker] Failed to save result: {e}", flush=True)

try:
    cfg = Config()
    cfg.data_root = "../simulations/features_static"
    cfg.results_dir = "/tmp/_svm_c1e6_workspace"
    os.makedirs(cfg.results_dir, exist_ok=True)

    print(f"[worker] cwd: {os.getcwd()}", flush=True)
    print(f"[worker] Loading data...", flush=True)
    t_load = time.time()
    data = u.prepare_data(cfg)
    result["prepare_data_seconds"] = float(time.time() - t_load)
    save_result()

    X_tr = data["X_train_smote"]
    y_tr = data["y_train_smote"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    print(f"[worker] X_train: {X_tr.shape}, X_test: {X_test.shape}", flush=True)
    result["n_features"] = int(X_tr.shape[1])
    result["n_train"] = int(X_tr.shape[0])
    result["n_test"] = int(X_test.shape[0])

    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    print(f"[worker] Training SVM with C={C_VAL}, gamma={GAMMA_VAL}, max_iter={SVM_MAX_ITER}", flush=True)
    t_start = time.time()

    # Single-fit to test convergence first
    svm_quick = SVC(C=C_VAL, gamma=GAMMA_VAL, kernel="rbf",
                    random_state=42, max_iter=SVM_MAX_ITER)
    svm_quick.fit(X_tr, y_tr)
    t_fit = time.time() - t_start

    # Check if SVM hit max_iter (== did NOT converge)
    n_iter_actual = svm_quick.n_iter_
    if hasattr(n_iter_actual, "__iter__"):
        # In binary classification it's an array of length 1
        n_iter_actual = int(n_iter_actual[0]) if len(n_iter_actual) > 0 else -1
    else:
        n_iter_actual = int(n_iter_actual)
    converged = (n_iter_actual < SVM_MAX_ITER)

    print(f"[worker] Single fit took {t_fit:.1f}s, n_iter={n_iter_actual} (max={SVM_MAX_ITER})", flush=True)
    print(f"[worker] Converged: {converged}", flush=True)
    result["single_fit_seconds"] = float(t_fit)
    result["n_iter"] = n_iter_actual
    result["converged"] = bool(converged)
    save_result()

    # Quick test_score on the trained single fit
    quick_test_acc = svm_quick.score(X_test, y_test)
    print(f"[worker] Quick test accuracy (no CV): {quick_test_acc:.4f}", flush=True)
    result["quick_test_accuracy"] = float(quick_test_acc)
    save_result()

    # If single fit took >300s, skip CV (would be ~3x = >900s)
    # Also skip if did not converge — CV result would be unreliable
    if t_fit > 300:
        print(f"[worker] Single fit too slow ({t_fit:.0f}s); skipping CV", flush=True)
        result["status"] = "single_fit_only_slow"
    elif not converged:
        print(f"[worker] Did not converge; skipping CV", flush=True)
        result["status"] = "did_not_converge"
    else:
        print(f"[worker] Running 3-fold CV...", flush=True)
        t_cv_start = time.time()
        cv_scores = cross_val_score(
            SVC(C=C_VAL, gamma=GAMMA_VAL, kernel="rbf",
                random_state=42, max_iter=SVM_MAX_ITER),
            X_tr, y_tr,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=1,  # Single-threaded; we already use the timeout mechanism
        )
        t_cv = time.time() - t_cv_start
        result["cv_accuracy"] = float(cv_scores.mean())
        result["cv_std"] = float(cv_scores.std())
        result["cv_scores"] = [float(s) for s in cv_scores]
        result["cv_seconds"] = float(t_cv)
        print(f"[worker] CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}", flush=True)
        print(f"[worker] CV took {t_cv:.0f}s", flush=True)
        result["status"] = "ok"

    print(f"[worker] DONE in {time.time() - t_start:.0f}s total", flush=True)
    save_result()

except Exception as e:
    result["status"] = "error"
    result["error"] = str(e)
    result["traceback"] = traceback.format_exc()
    print(f"[worker] ERROR: {e}", flush=True)
    save_result()

print(f"[worker] Result saved to {OUTPUT_JSON}", flush=True)
