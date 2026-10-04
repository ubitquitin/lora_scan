#!/usr/bin/env python
"""Flatten runs/*/metrics.json into results/runs.parquet (one row per run)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from lorascan import runs  # noqa: E402

df = pd.DataFrame(runs.collect_results())
out = runs.ROOT / "results"
out.mkdir(exist_ok=True)
df.to_parquet(out / "runs.parquet")
print(df.groupby("experiment_id")["accuracy"].agg(["mean", "std", "count"]) if len(df) else "no completed runs")
