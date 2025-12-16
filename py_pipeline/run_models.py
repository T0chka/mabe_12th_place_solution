#!/usr/bin/env python3
"""
Run models training pipeline

Trains models for each lab, saves OOF predictions, calibration, thresholds, and logs.

Lab and model-specific parameters are loaded from lab_config.yaml, feats_config*.yaml, params_config*.yaml,
which can be shared between R and Python pipelines.

Usage:
    Set EXP_PARAMS, MODEL_SETTINGS, and DEFAULT_FEATURE_PARAMS and run: python py_pipeline/run_models.py
"""

import gc
import sys
import time
import shutil
import pickle
import pandas as pd

from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from py_pipeline.models.boostings_multihead import BoostingMultihead
from py_pipeline.models.tabm_multiclass import TabMMulticlass
from py_pipeline.utils.data_loads import (
    build_dataset,
    compute_action_duration,
    get_bodyparts,
    get_whitelist,
    load_meta_df,
    load_train_ann,
    get_lab_window_specs,
    load_lab_config,
    load_feats_config,
    load_params_config,
    load_pseudo_df
)
from py_pipeline.utils.training import save_run_log
from py_pipeline.utils.scoring import get_scores

EXP_PARAMS = {
    "data_dir": "data",
    "submit_num": 155,

    # set one of: "lgbm_multihead", "cat_multihead", "xgb_multihead", "pyboost_multihead", "tabm_multiclass"
    "model_name": "tabm_multiclass", 

    # set list of labs to train on (e.g. ["CautiousGiraffe"] or None to train in all labs):
    "labs": None,
    
    # set path to params config (e.g. "configs/params_config_xgb.yaml" or None to use default params):
    "params_config": None,
    
    # set path to feats config (e.g. "configs/feats_config_xgb_103.yaml" or None to use base features only):
    "feats_config": "configs/feats_config_tabm_bp_raw_by_act.yaml",
    
    # set path to pseudo labels (e.g. "data_prepared/pseudo_labels.csv" or None to use no pseudo labels):
    "pseudo_path": None, 
}

MODEL_SETTINGS: Dict[str, Dict[str, Any]] = {
    "xgb_multihead": {
        "print_every": 0,
        "use_weights": False,
        "use_calibration": True,
        "k_folds": 5,
        "seed": 777,
    },
    "lgbm_multihead": {
        "print_every": 0,
        "use_weights": False,
        "use_calibration": True,
        "seed": 3,
    },
    "cat_multihead": {
        "print_every": 0,
        "use_weights": False,
        "use_calibration": True,
        "seed": 41,
    },
    "pyboost_multihead": {
        "print_every": 0,
        "use_weights": False,
        "use_calibration": True,
        "seed": 333,
    },
    "tabm_multiclass": {
        "task_mode": "multilabel_ovr", # "multilabel_ovr" or "multiclass"
        "seed": 1,
    },
}

DEFAULT_FEATURE_PARAMS = {
    "invariants": True,
    "single_mouse": True,
    "pair_dist": True,
    "coords": False,    
    "add_temporal": True,
    "augment": False,            # use True for TABM only
    "norm2cm": True,
    "norm_by": "frame_body_len", # "med_body_len", "frame_body_len", or "none"
    "fill_na": "none",           # "none" or "zeros"
    "bp_set": "raw"              # "raw" or "avg"
}

def main():
    """Main training loop."""
    # Paths
    project_root = Path(__file__).parent.parent

    data_dir   = project_root / EXP_PARAMS["data_dir"]
    tracks_dir = data_dir / "train_tracking"
    
    lab_config_path   = project_root / "configs/lab_config.yaml"
    feats_config_key = EXP_PARAMS.get("feats_config")
    params_config_key = EXP_PARAMS.get("params_config")
    pseudo_path = EXP_PARAMS.get("pseudo_path")
    pseudo_path = project_root / pseudo_path if pseudo_path else None

    save_dir = project_root / "submits_data" / f"submit{EXP_PARAMS['submit_num']}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "training_log.csv"
    print(f"[INFO] Output directory: {save_dir}")
        
    # Lab and features config
    lab_config = load_lab_config(lab_config_path)
    if feats_config_key:
        feats_config_path = project_root / feats_config_key
        feats_config = load_feats_config(feats_config_path)
    else:
        print("[INFO] feats_config not provided.")
        feats_config = None

    # Params config
    if params_config_key:
        params_config_path = project_root / params_config_key
        params_config = load_params_config(params_config_path).get(EXP_PARAMS["model_name"], {})
    else:
        print("[INFO] params_config not provided; using default params.")
        params_config = {}

    # Load data, get whitelist and compute action durations
    train_meta = load_meta_df(data_dir / "train.csv")
    train_ann = load_train_ann(data_dir, train_meta)
    whitelist_all = get_whitelist(train_meta)
    act_duration = compute_action_duration(train_ann)
    pseudo_df = load_pseudo_df(pseudo_path)

    # Get train labs
    all_labs: List[str] = sorted(train_ann["lab_id"].unique().tolist(), reverse=True)
    
    if EXP_PARAMS["labs"]:
        train_labs = [lab for lab in all_labs if lab in EXP_PARAMS["labs"]]
    else:
        excluded_labs = set(lab_config.get("labs", {}).get("excluded", []))
        train_labs = [lab for lab in all_labs if lab not in excluded_labs]
    
    # Get model/lab configs
    model_name = EXP_PARAMS["model_name"]
    model_settings: Dict[str, Any] = MODEL_SETTINGS.get(model_name, {"print_every": 0})
    model_cfg: Dict[str, Any] = lab_config.get(model_name, {}) or {}
    lag_secs = model_cfg.get("lag_secs", {}) or {}
    sample_neg = model_cfg.get("sample_neg_percent", {}) or {}
    sample_unl = model_cfg.get("sample_unl_percent", {}) or {}

    # Training loop
    start_time = time.time()
    for lab_name in train_labs:
        print(f"\n{'='*60}")
        print(f"LAB: {lab_name}")
        print(f"{'='*60}\n")
        
        train_labels = train_ann[train_ann["lab_id"] == lab_name].copy()
        whitelist = whitelist_all[whitelist_all["lab_id"] == lab_name].copy()
        
        # Build feature parameters for the lab
        feature_params = dict(DEFAULT_FEATURE_PARAMS)
        feature_params["lag_sec"] = lag_secs.get(lab_name, 0.1)
        feature_params["sample_neg_percent"] = sample_neg.get(lab_name, 10)
        feature_params["sample_unl_percent"] = sample_unl.get(lab_name, 0)

        feature_params["bodyparts"] = get_bodyparts(train_meta, lab_name)

        if feats_config and feature_params["add_temporal"]:
            window_specs = get_lab_window_specs(lab_name, feats_config)
            feature_params["window_cols"] = window_specs["window_cols"]
            feature_params["tmp_fnames_by_action"] = window_specs["tmp_fnames_by_action"]

        # Extract model params for the lab
        params = params_config.get(lab_name, {})

        print(
            f"Actions: {whitelist['action'].nunique()}, "
            f"Videos: {whitelist['video_id'].nunique()}, "
            f"lag_sec: {feature_params['lag_sec']}"
        )
        
        # Build dataset
        train_dt = build_dataset(
            labels_dt=train_labels,
            dt_meta=train_meta,
            whitelist=whitelist,
            tracks_dir=tracks_dir,
            lab_name=lab_name,
            feature_params=feature_params,
            verbose=False,
        )
        train_dt = train_dt.fillna(0)
        print("\nNumber of rows in train_dt:", len(train_dt))
        
        # Setup model directory (remove if already exists)
        lab_dir = save_dir / "labs" / lab_name
        if lab_dir.exists():
            shutil.rmtree(lab_dir)
        lab_dir.mkdir(parents=True)
        
        # Train model
        if model_name == "tabm_multiclass":
            model = TabMMulticlass(
                settings=model_settings,
                params=params,
                save_dir=lab_dir / "models",
                whitelist=whitelist,
                feature_params=feature_params,
            )
        else:
            model = BoostingMultihead(
                booster_type=model_name,
                settings=model_settings,
                params=params,
                save_dir=lab_dir / "models",
                whitelist=whitelist,
                feature_params=feature_params,
            )

        fit_start = time.time()
        fitted = model.fit(train_dt, pseudo_labels=pseudo_df)
        fit_time = time.time() - fit_start
        print(f"\nTraining time: {fit_time:.1f}s")
        
        print(f"\n{'='*60}")
        print(f"LAB: {lab_name} DONE")
        print(f"{'='*60}\n")
        
        # Get scores
        res = get_scores(
            fitted=fitted,
            train_labels=train_labels,
            whitelist=whitelist,
            act_duration=act_duration,
            verbose=True,
        )
        z_by_lab = res["z_by_lab"]
        scores = res["scores"]
        print("\nScores:")
        print(pd.DataFrame([scores]))
        print()
        
        # Save OOF predictions for this lab
        oof = fitted["oof"].copy()
        oof["lab_id"] = lab_name
        oof.to_parquet(lab_dir / "oof_dt.parquet", index=False)
        
        # Save z_by_lab for this lab
        z_by_lab.to_csv(lab_dir / "z_by_lab_by_act.csv", index=False)
        
        # Save training log
        lens = [len(cols) for cols in fitted["feature_cols"].values()]
        n_features = f"{min(lens)}-{max(lens)}"
        save_feature_params = {
            k: v for k, v in feature_params.items() 
            if k != "tmp_fnames_by_action"
        }
        
        save_run_log(
            lab_name=lab_name,
            params={**save_feature_params, **fitted["summary"], "fit_time_sec": round(fit_time, 1)},
            n_features=n_features,
            scores=scores,
            path=log_path,
        )
        
        # Save metadata
        (lab_dir / "config").mkdir(exist_ok=True)
        
        # Save config (excluding large objects)
        meta = {
            k: v for k, v in fitted.items()
            if k not in ["models", "oof", "calibs"]
        }
        meta["feature_params"] = feature_params
        
        with open(lab_dir / "config" / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        with open(lab_dir / "config" / "calibs.pkl", "wb") as f:
            pickle.dump(fitted["calibs"], f)
        
        gc.collect()
    
    # Save action duration (common for all labs)
    act_duration.to_csv(save_dir / "act_duration.csv", index=False)
    
    # Save source code
    code_dir = save_dir / "code"
    code_dir.mkdir(exist_ok=True)
    
    shutil.copy2(Path(__file__), code_dir / "run_models.py")
    project_root = Path(__file__).parent.parent
    pkg_src = project_root / "py_pipeline"
    pkg_dst = code_dir / "py_pipeline"

    shutil.copytree(
        pkg_src,
        pkg_dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()

