#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any

PREFERRED_METRICS = [
    "test_loss",
    "test_avg_cossim_top20_de",
    "test_avg_cossim_top20_unionde",
    "test_perturbations_with_effect_avg_cossim_top20_unionde",
    "test_pn_mse_top20_de",
    "test_perturbations_with_effect_pn_mse_top20_unionde",
]


def parse_field(field_spec: str) -> tuple[str, str]:
    if "=" in field_spec:
        name, path = field_spec.split("=", 1)
        return name, path
    return field_spec, field_spec


def nested_get(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


parser = argparse.ArgumentParser(description="Summarize PRESAGE grid-search runs.")
parser.add_argument("root", type=Path, help="Grid-search output directory")
parser.add_argument(
    "--field",
    action="append",
    default=[],
    help="Summary column mapping in the form name=config.path",
)
parser.add_argument(
    "--preview",
    type=int,
    default=5,
    help="Number of top rows to print after writing the CSV",
)
args = parser.parse_args()

root = args.root
field_specs = [parse_field(item) for item in args.field]
rows: list[dict[str, Any]] = []
all_keys: set[str] = set()

if not root.exists():
    raise SystemExit(f"Grid-search directory does not exist: {root}")

for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    cfg_file = run_dir / "config_used.json"
    summary_file = run_dir / "training_summary.json"
    test_files = sorted(run_dir.glob("test_metrics_*.json"))
    if not (cfg_file.exists() and summary_file.exists() and test_files):
        continue

    cfg = load_json(cfg_file)
    summary = load_json(summary_file)
    test_metrics = load_json(test_files[0])
    if isinstance(test_metrics, list):
        test_metrics = test_metrics[0] if test_metrics else {}

    row = {
        "run_dir": str(run_dir),
        "tag": nested_get(cfg, "cli_args.tag"),
        "best_val_loss": summary.get("best_val_loss"),
    }
    for name, path in field_specs:
        row[name] = nested_get(cfg, path)

    for key in PREFERRED_METRICS:
        if key in test_metrics:
            row[key] = test_metrics[key]
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            row.setdefault(key, value)

    rows.append(row)
    all_keys.update(row)

rows.sort(
    key=lambda row: float("inf")
    if row.get("best_val_loss") is None
    else float(row["best_val_loss"])
)

base_fields = [
    "run_dir",
    "tag",
    "best_val_loss",
    *PREFERRED_METRICS,
    *(name for name, _ in field_specs),
]
fieldnames: list[str] = []
for key in base_fields + sorted(k for k in all_keys if k not in base_fields):
    if key not in fieldnames:
        fieldnames.append(key)

out_file = root / "grid_summary.csv"
with out_file.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"saved summary: {out_file}")
for row in rows[: args.preview]:
    print(row)
