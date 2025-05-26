#!/usr/bin/env python3
"""
tune_walk_forward.py

Grid-search hyperparameters for walk_forward_backtest.py,
optimizing a composite of mean_sharpe * mean_calmar.
"""

import argparse
import itertools
import json
import subprocess
import tempfile
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Tune walk-forward hyperparameters")
    # core data & backtest args
    p.add_argument("--sep-master",   required=True, help="SEP parquet")
    p.add_argument("--universe-csv", required=True, help="Universe CSV")
    p.add_argument("--horizon",      default="5d")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window",  type=int, default=21)
    p.add_argument("--step-size",    type=int, default=21)
    p.add_argument("--purge-days",   type=int, default=4)
    p.add_argument("--embargo-days", type=int, default=2)
    p.add_argument("--backend",      choices=["xgb","dummy","torch"], default="xgb")
    p.add_argument("--device",       choices=["cpu","gpu"], default="gpu")
    p.add_argument("--threshold",    type=float, default=0.6)
    # tuning‐specific
    p.add_argument("--hyperparams", required=True,
                   help="JSON string mapping param names to lists of values, e.g. '{\"learning_rate\":[0.01,0.1],\"max_depth\":[3,5]}'")
    p.add_argument("--out-file", default="tuning_results.csv",
                   help="Where to save the grid‐search results")
    return p.parse_args()

def main():
    args = parse_args()

    # load the grid
    grid = json.loads(args.hyperparams)
    keys, values = zip(*grid.items())
    combos = list(itertools.product(*values))

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))

        # temp files for this trial's backtest outputs
        detail_tmp  = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        summary_tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)

        cmd = [
            "python", "walk_forward_backtest.py",
            "--sep-master",   args.sep_master,
            "--universe-csv", args.universe_csv,
            "--horizon",      args.horizon,
            "--train-window", str(args.train_window),
            "--test-window",  str(args.test_window),
            "--step-size",    str(args.step_size),
            "--purge-days",   str(args.purge_days),
            "--embargo-days", str(args.embargo_days),
            "--backend",      args.backend,
            "--device",       args.device,
            "--threshold",    str(args.threshold),
            "--out-detail",   detail_tmp.name,
            "--out-summary",  summary_tmp.name,
        ]
        # inject hyperparams for this trial
        if args.backend == "xgb":
            # pack all of the combos into one JSON blob
            cmd += ["--xgb-params", json.dumps(params)]
        else:
            for k, v in params.items():
                cmd += [f"--{k}", str(v)]


        print("→ Running trial:", params)
        subprocess.run(cmd, check=True)

        # read summary and pull out mean_sharpe & mean_calmar
        df = pd.read_csv(summary_tmp.name, names=["metric", "value"])
        if df.empty:
            print(f"⚠️ Warning: No results for params {params}")
            continue
        if "mean_sharpe" not in df.metric.values or "mean_calmar" not in df.metric.values:
            print(f"⚠️ Warning: Missing metrics for params {params}")
            continue
        m_sharpe = df.loc[df.metric=="mean_sharpe","value"].astype(float).item()
        m_calmar = df.loc[df.metric=="mean_calmar","value"].astype(float).item()
        
        composite = m_sharpe * m_calmar

        results.append({
            **params,
            "mean_sharpe":  m_sharpe,
            "mean_calmar":  m_calmar,
            "composite":    composite
        })

    # save full grid results
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_file, index=False)
    print(f"✨ Saved tuning results to {args.out_file}")

if __name__ == "__main__":
    main()
