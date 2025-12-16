#!/usr/bin/env python3
"""
Grid tuning script for models.
Saves only one tuning_log.csv file.

Usage: 
    Set EXP_PARAMS, MODEL_SETTINGS, and DEFAULT_FEATURE_PARAMS;
    set WINDOW_SETS (feature sets to tune) and/or LAG_SEC_GRID and/or SAMPLE_NEG_GRID and/or SAMPLE_UNL_GRID;
    and run: python py_pipeline/tune_models.py
"""
import gc
import sys
import time
import pandas as pd

from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

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
)
from py_pipeline.utils.training import save_run_log
from py_pipeline.utils.scoring import get_scores

SMALL_LABS = ["CautiousGiraffe", "DeliriousFly", "JovialSwallow", "ReflectiveManatee", "SparklingTapir"]
BIG_LABS = [
    "ElegantMink", "GroovyShrew", "InvincibleJellyfish",
    "LyricalHare", "NiftyGoldfinch", "PleasantMeerkat",
    "TranquilPanther", "UppityFerret", "BoisterousParrot", "AdaptableSnail"
]
LABS_WITH_SELF = [
    "UppityFerret", "TranquilPanther", "NiftyGoldfinch", "LyricalHare",
    "InvincibleJellyfish", "GroovyShrew", "AdaptableSnail",
]
LABS_WITH_PAIR = [
        "CautiousGiraffe", "DeliriousFly", "ElegantMink", "JovialSwallow",
        "PleasantMeerkat", "ReflectiveManatee", "SparklingTapir", "BoisterousParrot"
    ]
LABS_WITH_NA = [
    "TranquilPanther", "PleasantMeerkat", "JovialSwallow", "GroovyShrew",
    "ElegantMink", "CautiousGiraffe", "AdaptableSnail"
]
LABS_WITH_MABEUNL = [
    "CautiousGiraffe", "ElegantMink", "GroovyShrew", "InvincibleJellyfish", "LyricalHare",
    "PleasantMeerkat", "SparklingTapir", "TranquilPanther", "UppityFerret", "AdaptableSnail"
]

EXP_PARAMS = {
    "data_dir": "data",
    "log_num": "tabm_window_cols",
    "model_name": "tabm_multiclass",
    "labs": None,
    "feats_config": None,
    "early_stop": {
        "enabled": False,
        "patience": 2,
    },
    "params_config": None,
}

MODEL_SETTINGS: Dict[str, Dict[str, Any]] = {
    "xgb_multihead": {
        "print_every": 0,
        "use_weights": False,
        "use_calibration": True,
        "seed": 777,
    },
    "lgbm_multihead": {
        "print_every": 0,
        "use_weights": True,
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
    "norm2cm": True,
    "norm_by": "frame_body_len",
    "fill_na": "none",
    "bp_set": "raw"
}

WINDOW_SETS: List[List[str]] = [
    ["ag_speed_total", "head_approach", "m1m2+tail_base+tail_base", "turn_toward"],
    ["align_agent_los", "head_approach", "ag_forward_consistency", "m1m2+nose+tail_base"],
    ["r_head", "m1m2+nose+nose", "ag_curvature", "head_approach", "ag_a_forward"],
    ["ag_v_forward", "turn_toward", "m1m2+nose+tail_base", "head_approach", "ag_speed_total"],
    ["ag_curvature", "head_approach", "m1m2+tail_base+tail_base", "turn_toward", "ag_v_forward", "ag_forward_consistency"],
    ["head_approach", "m1m2+nose+tail_base", "ag_turn_smoothness", "turn_toward", "m1m2+tail_base+tail_base"],
    ["head_approach", "ag_v_lateral", "m1m2+nose+nose", "turn_toward", "ag_turn_smoothness"],
    ["ag_v_lateral", "turn_toward", "head_approach", "m1m2+nose+nose", "ag_curvature", "ag_forward_consistency"],
    ["r_head", "align_agent_los", "m1m2+nose+tail_base", "ag_a_forward", "m1m2+nose+nose"],
    ["head_approach", "ag_v_forward", "r_head", "m1m2+nose+tail_base", "m1m2+nose+nose", "ag_v_lateral"]
]
LAG_SEC_GRID: List[float] = []
SAMPLE_NEG_GRID: List[float] = []
SAMPLE_UNL_GRID: List[float] = []

def build_window_param_grid() -> List[Dict[str, Any]]:
    return [
        {
            "tune": "window",
            "config_name": f"window_{idx}",
            "window_cols": cols,
        }
        for idx, cols in enumerate(WINDOW_SETS, start=1)
    ]


def build_lag_param_grid() -> List[Dict[str, Any]]:
    return [
        {"tune": "lag", "config_name": f"lag_{lag}", "lag_sec": lag}
        for lag in LAG_SEC_GRID
    ]


def build_sample_param_grid() -> List[Dict[str, Any]]:
    if not SAMPLE_NEG_GRID:
        return [
            {
                "tune": "sample",
                "config_name": f"sample_{idx}",
                "sample_neg_percent": None,
                "sample_unl_percent": unl,
            }
            for idx, unl in enumerate(SAMPLE_UNL_GRID, start=1)
        ]
    else:
        combos = [
            (neg, unl)
            for neg in SAMPLE_NEG_GRID
            for unl in SAMPLE_UNL_GRID
        ]
        return [
            {
                "tune": "sample",
                "config_name": f"sample_{idx}",
                "sample_neg_percent": neg,
                "sample_unl_percent": unl,
            }
            for idx, (neg, unl) in enumerate(combos, start=1)
        ]

def build_tune_feats_config(
    lab_name: str,
    actions: List[str],
    window_cols_vec: List[str],
) -> Dict[str, Any]:
    lab_entry: Dict[str, Any] = {
        action: {"window_cols": window_cols_vec} for action in actions
    }
    feats_config: Dict[str, Any] = {lab_name: lab_entry}
    return feats_config

def main():
    """Main tuning loop."""
    # Paths
    project_root = Path(__file__).parent.parent

    data_dir   = project_root / EXP_PARAMS["data_dir"]
    tracks_dir = data_dir / "train_tracking"
    
    lab_config_path   = project_root / "configs/lab_config.yaml"
    feats_config_key = EXP_PARAMS.get("feats_config")
    params_config_key = EXP_PARAMS.get("params_config")

    save_dir = project_root / "train_logs"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"tuning_log_{EXP_PARAMS['log_num']}.csv"
    print(f"Log path: {log_path}")

    # Params config
    if params_config_key:
        params_config_path = project_root / params_config_key
        params_config = load_params_config(params_config_path).get(EXP_PARAMS["model_name"], {})
    else:
        print("[INFO] params_config not provided; using default params.")
        params_config = {}
        
    # Lab and features config
    lab_config = load_lab_config(lab_config_path)
    if feats_config_key:
        feats_config_path = project_root / feats_config_key
        feats_config = load_feats_config(feats_config_path)
    else:
        print("[INFO] feats_config not provided.")
        feats_config = None

    # Load data, get whitelist and compute action durations
    train_meta = load_meta_df(data_dir / "train.csv")
    train_ann = load_train_ann(data_dir, train_meta)
    whitelist_all = get_whitelist(train_meta)
    act_duration = compute_action_duration(train_ann)
    
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

    param_grid = (
        build_window_param_grid()
        + build_lag_param_grid()
        + build_sample_param_grid()
    )
    if not param_grid:
        param_grid = [{"tune": "baseline", "config_name": "default"}]

    early_cfg = EXP_PARAMS.get("early_stop", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 2))
    best_scores = defaultdict(lambda: float("-inf"))
    best_configs = {}
    no_improve_counts = defaultdict(int)
    stopped_labs = set()

    # Training loop
    start_time = time.time()

    for cfg_index, cfg in enumerate(param_grid, start=1):
        tune_type = cfg["tune"]
        cfg_name = cfg.get("config_name", f"cfg_{cfg_index}")
        window_cols_vec: Optional[List[str]] = cfg.get("window_cols")
        print("\n" + "=" * 80)
        print(f"LABS: {train_labs}")
        print(f"TUNING CONFIG {cfg_index} [{cfg_name}]")
        print("=" * 80 + "\n")

        for lab_name in train_labs:
            if early_enabled:
                if lab_name in stopped_labs:
                    print(
                        f"Skipping {lab_name} for config {cfg_name} "
                        f"(early stop triggered)."
                    )
                    continue

            train_labels = train_ann[train_ann["lab_id"] == lab_name].copy()
            whitelist = whitelist_all[whitelist_all["lab_id"] == lab_name].copy()
            
            # Build feature parameters for the lab
            feature_params = dict(DEFAULT_FEATURE_PARAMS)
            feature_params["lag_sec"] = lag_secs.get(lab_name, 0.1)
            feature_params["sample_neg_percent"] = sample_neg.get(lab_name, 10)
            feature_params["sample_unl_percent"] = sample_unl.get(lab_name, 0)

            feature_params["bodyparts"] = get_bodyparts(train_meta, lab_name)
            
            # build params for the lab
            params = params_config.get(lab_name, {})

            if tune_type == "window":
                if not window_cols_vec:
                    raise ValueError("window_cols must be provided for window tuning")
                actions = sorted(train_labels["action"].unique().tolist())
                feats_config = build_tune_feats_config(
                    lab_name=lab_name,
                    actions=actions,
                    window_cols_vec=window_cols_vec,
                )
            elif tune_type == "lag":
                feature_params["lag_sec"] = cfg["lag_sec"]
            elif tune_type == "sample":
                features_params_overrides = {
                    k: v for k, v in cfg.items()
                    if k not in {"tune", "config_name", "window_cols"} and v is not None
                }
                for key, value in features_params_overrides.items():
                    feature_params[key] = value
            elif tune_type == "baseline":
                pass
            else:
                raise ValueError(f"Unknown tune type: {tune_type}")
            
            if feats_config and feature_params.get("add_temporal", False):
                window_specs = get_lab_window_specs(lab_name, feats_config)
                feature_params["window_cols"] = window_specs["window_cols"]
                feature_params["tmp_fnames_by_action"] = window_specs["tmp_fnames_by_action"]

            log_params = {
                "lag_sec": feature_params["lag_sec"],
                "sample_neg_percent": feature_params["sample_neg_percent"],
                "sample_unl_percent": feature_params["sample_unl_percent"],
                "window_cols": feature_params.get("window_cols"),
            }
            print(
                f"LAB: {lab_name}, "
                f"Actions: {train_labels['action'].nunique()}, "
                f"Videos: {train_labels['video_id'].nunique()} "
                f"| CONFIG #{cfg_index} [{cfg_name}]"
            )
            print("Current params:", ", ".join(f"{k}={v}" for k, v in log_params.items()))
            print()
            
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
            
            # Train model
            if model_name == "tabm_multiclass":
                model = TabMMulticlass(
                    settings=model_settings,
                    params=params,
                    save_dir=None,
                    whitelist=whitelist,
                    feature_params=feature_params,
                )
            else:
                model = BoostingMultihead(
                    booster_type=model_name,
                    settings=model_settings,
                    params=params,
                    save_dir=None,
                    whitelist=whitelist,
                    feature_params=feature_params,
                )

            fit_start = time.time()
            fitted = model.fit(train_dt) #, pseudo_labels=None
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
                verbose=False,
            )
            scores = res["scores"]
            print("\nScores:")
            print(pd.DataFrame([scores]))
            print()
            
            # Save training log
            lens = [len(cols) for cols in fitted["feature_cols"].values()]
            n_features = f"{min(lens)}-{max(lens)}"
            save_feature_params = {
                k: v for k, v in feature_params.items() 
                if k != "tmp_fnames_by_action"
            }
            save_feature_params["tune"] = tune_type
            save_feature_params["config_name"] = cfg_name
            
            save_run_log(
                lab_name=lab_name,
                params={**save_feature_params, **fitted["summary"], "fit_time_sec": round(fit_time, 1)},
                n_features=n_features,
                scores=scores,
                path=log_path,
            )

            macro_score = scores.get("f_beta_postproc")
            if early_enabled:
                if macro_score > best_scores[lab_name]:
                    best_scores[lab_name] = macro_score
                    best_configs[lab_name] = {
                        "config": cfg_name,
                        "tune": tune_type,
                        "score": macro_score,
                        **log_params,
                    }
                    no_improve_counts[lab_name] = 0
                else:
                    no_improve_counts[lab_name] += 1
                    if no_improve_counts[lab_name] >= early_patience:
                        stopped_labs.add(lab_name)
                        print(
                            f"Early stop for {lab_name} "
                            f"(patience={early_patience})."
                        )
            
            del model, fitted, train_dt, train_labels, whitelist
            gc.collect()

    if best_configs:
        print("\nBest configs by lab:")
        for lab, info in best_configs.items():
            print(
                f"- {lab}: score={info['score']:.4f}, "
                f"config={info['config']} (tune={info['tune']}), "
                f"params={{lag_sec={info['lag_sec']}, "
                f"sample_neg_percent={info['sample_neg_percent']}, "
                f"sample_unl_percent={info['sample_unl_percent']}}}"
            )
    else:
        print("\nNo configurations improved over the baseline.")
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()

