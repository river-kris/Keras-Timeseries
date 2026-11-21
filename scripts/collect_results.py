"""Collect results/<run>/metrics.json files into Markdown benchmark tables.

Usage (from the repo root):
    python scripts/collect_results.py            # print tables
    python scripts/collect_results.py --csv      # also write results/summary.csv
"""
import csv
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

COLUMNS = {
    "transformer": [
        ("run_name", "Run"),
        ("test_accuracy", "Test acc"),
        ("test_loss", "Test loss"),
        ("best_val_accuracy", "Best val acc"),
        ("best_epoch", "Best epoch"),
        ("epochs_run", "Epochs run"),
        ("params", "Params"),
        ("train_time_min", "Train time (min)"),
    ],
    "lstm": [
        ("run_name", "Run"),
        ("best_val_mse", "Val MSE"),
        ("best_val_mae", "Val MAE"),
        ("best_val_mae_degC", "Val MAE (°C)"),
        ("best_epoch", "Best epoch"),
        ("epochs_run", "Epochs run"),
        ("params", "Params"),
        ("train_time_min", "Train time (min)"),
    ],
}


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def load_runs():
    runs = []
    for path in sorted(RESULTS.glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        cfg_path = path.with_name("config.json")
        m["_config"] = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        runs.append(m)
    # baseline first, then alphabetical
    runs.sort(key=lambda m: (m.get("model", ""), "baseline" not in m["run_name"], m["run_name"]))
    return runs


def table(runs, model):
    cols = COLUMNS[model]
    rows = [m for m in runs if m.get("model") == model]
    if not rows:
        return f"_No {model} runs in results/ yet._\n"
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for m in rows:
        lines.append("| " + " | ".join(fmt(m.get(k, "")) for k, _ in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    runs = load_runs()
    print("### Transformer classification (FordA)\n")
    print(table(runs, "transformer"))
    print("### LSTM forecasting (Jena Climate)\n")
    print(table(runs, "lstm"))

    if "--csv" in sys.argv and runs:
        keys = sorted({k for m in runs for k in m if not k.startswith("_") and k != "confusion_matrix"})
        out = RESULTS / "summary.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(runs)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
