"""
Checkpoint utilities for robust JSON-based result saving.

Usage pattern for experiment scripts:

    from checkpoint_utils import atomic_save_json, load_json_if_exists

    json_path = os.path.join(save_dir, "data", "results_xxx.json")
    all_results = load_json_if_exists(json_path)   # {} if file missing/corrupt

    for key, seed in combinations:
        if key in all_results and str(seed) in all_results[key]:
            continue                                # resume: skip completed
        hist = run_experiment(...)
        all_results.setdefault(key, {})[str(seed)] = hist
        atomic_save_json(all_results, json_path)    # checkpoint after each unit

Features:
  • atomic_save_json: writes to .tmp then os.replace(), so a crash mid-write
    cannot truncate the existing JSON. POSIX rename is atomic.
  • load_json_if_exists: safe load; missing or corrupt → returns {}.

This utility does NOT touch any learning logic — it only replaces the final
"json.dump at the end" pattern with "json.dump after every completed unit".
"""
import json
import os


def atomic_save_json(data, path: str):
    """Atomic JSON write: stage to path+'.tmp', then os.replace() swap in.

    Parameters
    ----------
    data : any JSON-serializable object
    path : destination file path. Parent directory is created if needed.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)   # atomic on POSIX


def load_json_if_exists(path: str) -> dict:
    """Load a JSON dict from `path` or return {} if missing / corrupt.

    Designed for resume: call at the start of main(); the returned dict holds
    already-completed units keyed by whatever scheme the script uses.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"[resume] {path} is not a dict, starting fresh")
            return {}
        return data
    except Exception as e:
        print(f"[resume] failed to load {path} ({e}); starting fresh")
        return {}
