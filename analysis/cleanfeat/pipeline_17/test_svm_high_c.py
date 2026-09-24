#!/usr/bin/env python3
"""
Test SVM with C=10^6 (and selected gammas) to verify whether the C=10^5 boundary
in the FOCUSED search reflects a true performance plateau or a missed optimum.

Each (C, gamma) combination runs in a SEPARATE PROCESS with a hard timeout
(default 1800s = 30 min). If a process exceeds the timeout, all child processes
are killed via process group, and the configuration is recorded as "timeout".

Output: results/hp_search_svm_c1e6/{static}/svm_c1e6_results.json
        Per-config logs: results/hp_search_svm_c1e6/{static}/log_c{C}_g{gamma}.txt

Run from strict_observable_v2/ directory:
    cd ~/ns3/Final_Project_NS3-master/strict_observable_v2
    OMP_NUM_THREADS=12 MAX_JOBS=12 \
        nohup python3 -u test_svm_high_c.py > test_svm_high_c.log 2>&1 &
    disown
"""

import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path

# CRITICAL: MAX_JOBS must be set before any imports below
if "MAX_JOBS" not in os.environ:
    print("[fatal] MAX_JOBS env var must be set.")
    print("        Try: MAX_JOBS=12 python3 test_svm_high_c.py")
    sys.exit(1)

print(f"[setup] MAX_JOBS env: {os.environ['MAX_JOBS']}")
print(f"[setup] cwd:          {os.getcwd()}")

# Verify cwd contains the modules we'll need
REQUIRED_MODULES = ["defense_detection_v2.py", "unified_hp_search_v2.py"]
for m in REQUIRED_MODULES:
    if not Path(m).exists():
        print(f"[fatal] {m} not found in cwd; run this script from strict_observable_v2/")
        sys.exit(1)
print(f"[setup] required modules present")

# Configurations to test
TEST_CONFIGS = [
    # (C, gamma) pairs to evaluate at C=10^6
    (1_000_000, 1e-5),
    (1_000_000, 1e-4),
    (1_000_000, 1e-3),
]

# Per-process timeout (seconds). 30 minutes per config.
PER_CONFIG_TIMEOUT_SEC = 1800

OUTPUT_DIR = Path("results/hp_search_svm_c1e6/static")
WORKING_DIR = os.getcwd()  # Capture early for subprocess


# ----------------------------------------------------------------------
# Subprocess worker — runs ONE (C, gamma) in isolation. The worker:
#   1. Sets cwd explicitly
#   2. Loads data via prepare_data
#   3. Single-fit SVM, checks convergence (n_iter_ vs max_iter)
#   4. If single fit was fast enough, runs 3-fold CV
#   5. Writes result to JSON
# ----------------------------------------------------------------------
WORKER_CODE = '''
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
'''


def run_one_config(C: float, gamma: float, output_dir: Path) -> dict:
    """Run a single (C, gamma) combination in a subprocess with timeout.

    Uses preexec_fn to put the subprocess in its own process group so we can
    kill all descendants (including any sklearn worker pools) on timeout.
    """
    print(f"\n{'='*70}")
    print(f"  Testing C={C:g}, gamma={gamma:g}")
    print(f"{'='*70}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = output_dir / f"result_C{C:g}_g{gamma:g}.json"
    log_file = output_dir / f"log_C{C:g}_g{gamma:g}.txt"

    # Initialize a placeholder result. Worker will overwrite incrementally.
    placeholder = {
        "C": C,
        "gamma": gamma,
        "status": "did_not_complete",
    }
    with open(result_json, "w") as f:
        json.dump(placeholder, f, indent=2)

    # Write worker script (same content for each config; ok to rewrite)
    worker_path = output_dir / "_worker.py"
    with open(worker_path, "w") as f:
        f.write(WORKER_CODE)

    # Launch subprocess in its own process group
    cmd = [
        sys.executable, "-u", str(worker_path),
        str(C), str(gamma), str(result_json), WORKING_DIR,
    ]
    print(f"[main] Spawning subprocess (timeout={PER_CONFIG_TIMEOUT_SEC}s, cwd={WORKING_DIR})...")
    t_start = time.time()

    proc = None
    try:
        with open(log_file, "w") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf, stderr=subprocess.STDOUT,
                cwd=WORKING_DIR,
                preexec_fn=os.setsid,  # New process group; kill via killpg
            )
            try:
                returncode = proc.wait(timeout=PER_CONFIG_TIMEOUT_SEC)
                elapsed = time.time() - t_start
                print(f"[main] Subprocess finished in {elapsed:.0f}s, returncode={returncode}")
            except subprocess.TimeoutExpired:
                elapsed = time.time() - t_start
                print(f"[main] TIMEOUT after {elapsed:.0f}s — killing process group")
                # Send SIGTERM to entire process group, wait briefly, then SIGKILL
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    time.sleep(2)
                    if proc.poll() is None:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        time.sleep(1)
                except ProcessLookupError:
                    pass  # Already dead
                except Exception as e:
                    print(f"[main] Error killing process group: {e}")
                # Update result file
                timeout_result = {
                    "C": C,
                    "gamma": gamma,
                    "status": "timeout",
                    "timeout_sec": PER_CONFIG_TIMEOUT_SEC,
                    "elapsed_sec": elapsed,
                }
                with open(result_json, "w") as f:
                    json.dump(timeout_result, f, indent=2)
                return timeout_result

        # Read whatever the worker last wrote
        with open(result_json) as f:
            return json.load(f)

    except Exception as e:
        elapsed = time.time() - t_start
        print(f"[main] Exception: {e}")
        # Best-effort kill
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        error_result = {
            "C": C,
            "gamma": gamma,
            "status": "main_exception",
            "error": str(e),
            "elapsed_sec": elapsed,
        }
        with open(result_json, "w") as f:
            json.dump(error_result, f, indent=2)
        return error_result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("  SVM C=10^6 BOUNDARY TEST")
    print("="*70)
    print(f"  Configs to test:    {len(TEST_CONFIGS)}")
    print(f"  Per-config timeout: {PER_CONFIG_TIMEOUT_SEC}s ({PER_CONFIG_TIMEOUT_SEC/60:.0f} min)")
    print(f"  Max total time:     {len(TEST_CONFIGS) * PER_CONFIG_TIMEOUT_SEC / 60:.0f} min")
    print(f"  Output:             {OUTPUT_DIR}")
    print(f"  Working directory:  {WORKING_DIR}")

    # Reference: previous best from FOCUSED
    reference = {
        "C": 100000,
        "gamma": 0.0001,
        "test_accuracy": 0.8780,
        "cv_accuracy": 0.8810,
        "source": "results/hp_search_focused/static",
    }
    print(f"\n  Reference (FOCUSED best): C={reference['C']}, gamma={reference['gamma']}, "
          f"test_acc={reference['test_accuracy']:.4f}, cv_acc={reference['cv_accuracy']:.4f}")

    all_results = {"reference": reference, "tests": []}

    for C, gamma in TEST_CONFIGS:
        result = run_one_config(C, gamma, OUTPUT_DIR)
        all_results["tests"].append(result)
        print(f"  Status: {result.get('status')}")
        # Print quick test accuracy if available
        if "quick_test_accuracy" in result:
            print(f"  Quick test accuracy: {result['quick_test_accuracy']:.4f}")
        if "cv_accuracy" in result and result["cv_accuracy"] is not None:
            print(f"  CV accuracy:         {result['cv_accuracy']:.4f}")

        # Save accumulated results after each test
        with open(OUTPUT_DIR / "svm_c1e6_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final summary
    print("\n\n" + "="*82)
    print("  SUMMARY: SVM C=10^6 boundary test")
    print("="*82)
    print(f"  Reference (C=10^5, gamma=10^-4): test_acc={reference['test_accuracy']:.4f}, "
          f"cv_acc={reference['cv_accuracy']:.4f}")
    print()
    print(f"  {'C':<10}{'gamma':<10}{'status':<22}{'test':<10}{'cv':<10}{'Δtest':<10}{'Δcv':<8}")
    print("  " + "-"*80)

    best_test_acc = reference["test_accuracy"]
    best_cv_acc = reference["cv_accuracy"]
    best_params_test = None
    best_params_cv = None

    for r in all_results["tests"]:
        C_str = f"{r['C']:g}"
        g_str = f"{r['gamma']:g}"
        status = r.get("status", "?")[:21]

        test_acc = r.get("quick_test_accuracy")
        cv_acc = r.get("cv_accuracy")

        test_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        cv_str = f"{cv_acc:.4f}" if cv_acc is not None else "N/A"

        if test_acc is not None:
            d_test = test_acc - reference["test_accuracy"]
            d_test_str = f"{'+' if d_test >= 0 else ''}{d_test:.4f}"
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_params_test = (r["C"], r["gamma"])
        else:
            d_test_str = "N/A"

        if cv_acc is not None:
            d_cv = cv_acc - reference["cv_accuracy"]
            d_cv_str = f"{'+' if d_cv >= 0 else ''}{d_cv:.4f}"
            if cv_acc > best_cv_acc:
                best_cv_acc = cv_acc
                best_params_cv = (r["C"], r["gamma"])
        else:
            d_cv_str = "N/A"

        print(f"  {C_str:<10}{g_str:<10}{status:<22}{test_str:<10}{cv_str:<10}{d_test_str:<10}{d_cv_str:<8}")

    print()
    # Verdict based on CV accuracy (more reliable than single-fit test)
    if best_params_cv:
        d = best_cv_acc - reference["cv_accuracy"]
        if d >= 0.005:
            print(f"  >>> IMPROVEMENT (CV-based): C={best_params_cv[0]:g}, gamma={best_params_cv[1]:g}")
            print(f"      cv_acc={best_cv_acc:.4f} (Δ=+{d:.4f}). Consider updating Tables XII/XIII.")
        elif d >= 0:
            print(f"  Best new (CV-based): C={best_params_cv[0]:g}, gamma={best_params_cv[1]:g}, "
                  f"cv_acc={best_cv_acc:.4f} (Δ=+{d:.4f}, within noise)")
            print(f"  No improvement justifying paper changes.")
        else:
            print(f"  Best new config no better than reference. Reference (C=10^5) stands.")
    elif best_params_test:
        d = best_test_acc - reference["test_accuracy"]
        print(f"  Best new (test-only, CV failed/skipped): C={best_params_test[0]:g}, "
              f"gamma={best_params_test[1]:g}, test_acc={best_test_acc:.4f} (Δ={'+' if d>=0 else ''}{d:.4f})")
    else:
        print(f"  No configuration completed successfully.")
        print(f"  Justification for paper: 'Extending the SVM grid to C=10^6 did not yield")
        print(f"  configurations that converged within practical computational budgets.'")

    print(f"\n  Full results: {OUTPUT_DIR / 'svm_c1e6_results.json'}")


if __name__ == "__main__":
    main()