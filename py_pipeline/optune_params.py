#!/usr/bin/env python3
"""
Optuna-based tuning for model hyperparameters.

This script mirrors the data loading / feature building pipeline from run_models.py,
but performs an Optuna search over model parameters (learning_rate, max_depth, etc.).
Feature, model, and lab-specific parameters stay fixed (taken from lab_config + feats_config).
"""

import gc
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List

import optuna
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from py_pipeline.models.boostings_multihead import BoostingMultihead
from py_pipeline.utils.data_loads import (
    build_dataset,
    get_bodyparts,
    get_whitelist,
    load_meta_df,
    load_train_ann,
    get_lab_window_specs,
    load_lab_config,
    load_feats_config,
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXP_PARAMS = {
    "data_dir": "data",
    "model_name": "pyboost_multihead",
    "log_num": "optuna_pyboost_map_1",
    "config_id": "1",
    "labs": ["DeliriousFly"],
    "feats_config": "configs/feats_config_xgb_103.yaml",
}

DEFAULT_FEATURE_PARAMS = {
    "invariants": True,
    "single_mouse": True,
    "pair_dist": True,
    "coords": False,
    "add_temporal": True,
    "norm2cm": True,
    "norm_by": "frame_body_len",
    "fill_na": "none",
    "bp_set": "raw",
}

OPTUNA_CFG = {
    "n_trials": 24,
    "sampler_seed": 1003,
    "target_metric": "cv_valid_metric",
    "metric_direction": "maximize",
}

def suggest_booster_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
      'lr': trial.suggest_categorical('lr', [0.01, 0.03]),
      "lambda_l2": 1,
      "gd_steps": 1,
      "subsample": trial.suggest_categorical('subsample', [0.8, 0.9]), 
      "colsample": trial.suggest_categorical('colsample', [0.8, 0.9]), 
      "min_data_in_leaf": 10, 
      "use_hess": True,
      "max_bin": 256, 
      "max_depth": trial.suggest_categorical('max_depth', [7, 8, 9]), 
      "debug": True
    }
    return params

def objective(
    trial: optuna.Trial,
    train_dt: pd.DataFrame,
    whitelist: pd.DataFrame,
    feature_params: Dict[str, Any],
    model_name: str,
    lab_name: str,
    log_path: Path,
) -> float:
    fit_time = 0.0
    try:
        feature_params_trial = feature_params.copy()

        booster_params = suggest_booster_params(trial)
        model = BoostingMultihead(
            booster_type=model_name,
            params=booster_params,
            settings={"print_every": 0},
            save_dir=None,
            whitelist=whitelist,
            feature_params=feature_params_trial,
        )

        fit_start = time.time()
        fitted = model.fit(train_dt)
        fit_time = time.time() - fit_start
        trial.set_user_attr("fit_time", fit_time)

        summary = fitted["summary"]
        metric_value = summary[OPTUNA_CFG["target_metric"]]
        
        trial.set_user_attr("metric_value", metric_value)
        status = "ok"
    except Exception as e:
        print(f"Error: {e}")
        raise

    row = {
        "entry_type": "trial",
        "lab_name": lab_name,
        "trial_number": trial.number,
        "score": metric_value,
        "fit_time": fit_time,
        **trial.params,
    }
    row_df = pd.DataFrame([row])
    header = not log_path.exists()
    row_df.to_csv(log_path, mode="a", header=header, index=False)
    print(
        f"\nAdded trial {trial.number} to log: "
        f"{OPTUNA_CFG['target_metric']}={metric_value:.4f} | "
        f"fit_time={fit_time:.1f}s | params={trial.params}"
    )
    gc.collect()
    return metric_value


def tune_lab(
    lab_name: str,
    model_name: str,
    train_dt: pd.DataFrame,
    whitelist: pd.DataFrame,
    feature_params: Dict[str, Any],
    log_path: Path,
) -> optuna.trial.FrozenTrial:
    print(f"\n{'='*60}")
    print(f"Tuning booster params for lab: {lab_name}")
    print(f"{'='*60}")
    start_time = time.time()

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_CFG["sampler_seed"])
    study = optuna.create_study(direction=OPTUNA_CFG["metric_direction"], sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            train_dt,
            whitelist,
            feature_params,
            model_name,
            lab_name,
            log_path,
        ),
        n_trials=OPTUNA_CFG["n_trials"],
        show_progress_bar=False,
    )

    elapsed = time.time() - start_time
    print(f"\nBest score for {lab_name}: {study.best_value:.6f}")
    print("Best booster params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Tuning time for {lab_name}: {elapsed:.1f}s\n")
    return study.best_trial


def main():
    # Paths
    project_root = Path(__file__).parent.parent

    data_dir   = project_root / EXP_PARAMS["data_dir"]
    tracks_dir = data_dir / "train_tracking"
    
    lab_config_path   = project_root / "configs/lab_config.yaml"
    feats_config_path = project_root / EXP_PARAMS["feats_config"]
    
    save_dir = project_root / "train_logs"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"tuning_log_{EXP_PARAMS['log_num']}.csv"
    print(f"Log path: {log_path}")
        
    # Load data, get whitelist and compute action durations
    train_meta = load_meta_df(data_dir / "train.csv")
    train_ann = load_train_ann(data_dir, train_meta)
    whitelist_all = get_whitelist(train_meta)
    lab_config = load_lab_config(lab_config_path)
    feats_config = load_feats_config(feats_config_path)
    
    # Get train labs
    all_labs: List[str] = sorted(train_ann["lab_id"].unique().tolist(), reverse=True)
    
    if EXP_PARAMS["labs"]:
        train_labs = [lab for lab in all_labs if lab in EXP_PARAMS["labs"]]
    else:
        excluded_labs = set(lab_config.get("labs", {}).get("excluded", []))
        train_labs = [lab for lab in all_labs if lab not in excluded_labs]
    
    print(f"Train labs: {train_labs}")

    # Get model/features config
    model_name = EXP_PARAMS["model_name"]
    model_cfg: Dict[str, Any] = lab_config.get(model_name, {}) or {}

    lag_secs = model_cfg.get("lag_secs", {}) or {}
    sample_neg = model_cfg.get("sample_neg_percent", {}) or {}
    sample_unl = model_cfg.get("sample_unl_percent", {}) or {}

    for lab_name in train_labs:
        train_labels = train_ann[train_ann["lab_id"] == lab_name].copy()
        whitelist = whitelist_all[whitelist_all["lab_id"] == lab_name].copy()
        
        # Build feature parameters for the lab
        feature_params = dict(DEFAULT_FEATURE_PARAMS)
        feature_params["lag_sec"] = lag_secs.get(lab_name, 0.1)
        feature_params["sample_neg_percent"] = sample_neg.get(lab_name, 10)
        feature_params["sample_unl_percent"] = sample_unl.get(lab_name, 0)

        feature_params["bodyparts"] = get_bodyparts(train_meta, lab_name)

        window_specs = get_lab_window_specs(lab_name, feats_config)
        feature_params["window_cols"] = window_specs["window_cols"]
        feature_params["tmp_fnames_by_action"] = window_specs["tmp_fnames_by_action"]

        # Build dataset
        train_dt = build_dataset(
            labels_dt=train_labels,
            dt_meta=train_meta,
            whitelist=whitelist,
            tracks_dir=tracks_dir,
            lab_name=lab_name,
            feature_params=feature_params,
        )
        train_dt = train_dt.fillna(0)

        best_trial = tune_lab(
            lab_name=lab_name,
            model_name=model_name,
            train_dt=train_dt,
            whitelist=whitelist,
            feature_params=feature_params,
            log_path=log_path,
        )
        row_df = pd.DataFrame(
            [
                {
                    "entry_type": "best",
                    "lab_name": lab_name,
                    "trial_number": best_trial.number,
                    "score": best_trial.value,
                    **best_trial.params,
                }
            ]
        )
        header = not log_path.exists()
        row_df.to_csv(log_path, mode="a", header=header, index=False)
        print(f"Logged best config for {lab_name} to {log_path}")

        params_config_path = (
            project_root / "configs" /
            f"params_config_{EXP_PARAMS['model_name'].replace('_multihead','')}_{EXP_PARAMS['config_id']}.yaml"
        )

        if params_config_path.exists():
            with params_config_path.open("r") as f:
                params_cfg = yaml.safe_load(f) or {}
        else:
            params_cfg = {}

        if model_name not in params_cfg:
            params_cfg[model_name] = {}

        params_cfg[model_name][lab_name] = best_trial.params

        with params_config_path.open("w") as f:
            yaml.safe_dump(params_cfg, f, sort_keys=False)
        print(f"Logged best params for {lab_name} to {params_config_path}")

    print("Optuna tuning finished.")


if __name__ == "__main__":
    main()

