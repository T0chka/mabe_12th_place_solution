"""
Threshold fitting utilities.

The functions below fit optimal logit thresholds z per (lab_id, action) pair or per lab_id .
"""

import pandas as pd
import numpy as np
import time
import multiprocessing as mp

from typing import Any, Optional
from concurrent.futures import ProcessPoolExecutor

from .postprocessing import (
    filter_preds_by_thr,
    get_intervals,
    postprocess_preds,
)
from .scoring import mouse_fbeta_by_action


def tune_z(
    pred: pd.DataFrame,
    sol: pd.DataFrame,
    z_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Tune logit threshold z to maximize F-beta
    for a single (lab_id, action) pair.
    """
    lab_vals = pred["lab_id"].unique()
    act_vals = pred["action"].unique()
    
    lab = lab_vals[0]
    act = act_vals[0]

    scores_arr = pred["score"].to_numpy(dtype=np.float32, copy=False)
    
    z_grid = np.asarray(z_grid, dtype=np.float32)

    zmin = float(scores_arr.min())
    zmax = float(scores_arr.max())

    mask = (z_grid >= zmin) & (z_grid <= zmax)
    zg = z_grid[mask]

    if zg.size == 0:
        zg = np.array([zmin], dtype=np.float32)

    best_z: float = np.nan
    best_s: float = -np.inf

    for z in zg:
        sub = filter_preds_by_thr(pred.copy(), logit_thr=float(z))
        sub = get_intervals(sub)

        if sub.empty:
            s = 0.0
        else:
            score_dict = mouse_fbeta_by_action(sol, sub)
            if "by_action" not in score_dict:
                raise ValueError("mouse_fbeta_by_action must return 'by_action'")
            by_action = score_dict["by_action"]
            if by_action.empty:
                raise ValueError("by_action DataFrame is empty")
            action_scores = by_action[by_action["action"] == act]
            if action_scores.empty:
                raise ValueError(f"Action {act} not found in scores")
            s = float(action_scores["f_beta"].iloc[0])

        if np.isfinite(s) and s > best_s:
            best_s = s
            best_z = float(z)

    if np.isnan(best_z):
        raise ValueError("No valid threshold found")

    result = pd.DataFrame(
        {
            "lab_id": [lab],
            "action": [act],
            "z_thr": np.array([best_z], dtype=np.float32),
            "score": np.array([best_s], dtype=np.float32),
        }
    )
    return result


def fit_z(
    preds_dt: pd.DataFrame,
    solution: pd.DataFrame,
    roll_frames: Optional[Any],
    z_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Fit optimal logit thresholds z per (lab_id, action) pair or per lab_id .
    """
    if preds_dt.empty:
        raise ValueError("fit_z: preds_dt is empty")
    if solution.empty:
        raise ValueError("fit_z: solution is empty")

    preds_dt = preds_dt.copy()
    preds_dt["score"] = preds_dt["score"].astype("float32")
    z_grid = np.asarray(z_grid, dtype=np.float32)

    preds_dt = postprocess_preds(preds_dt, roll_frames=roll_frames)

    pairs_available = preds_dt[["lab_id", "action"]].drop_duplicates()
    pair_sizes = (
        preds_dt.groupby(["lab_id", "action"])
        .size()
        .reset_index(name="N")
    )
    tasks = pairs_available.merge(
        pair_sizes, on=["lab_id", "action"], how="left"
    )
    tasks = tasks.sort_values(
        ["N", "lab_id", "action"], ascending=[False, True, True]
    )

    pred_by_lab = {
        str(lab): group for lab, group in preds_dt.groupby("lab_id")
    }
    sol_by_lab = {
        str(lab): group for lab, group in solution.groupby("lab_id")
    }

    pred_list = []
    sol_list = []
    for i in range(len(tasks)):
        row = tasks.iloc[i]
        lab = str(row["lab_id"])
        act = row["action"]

        if lab not in pred_by_lab or lab not in sol_by_lab:
            continue

        pred_lab = pred_by_lab[lab]
        sol_lab = sol_by_lab[lab]

        pred_sub = pred_lab[pred_lab["action"] == act].copy()
        if pred_sub.empty:
            continue

        pred_list.append(pred_sub)
        sol_list.append(sol_lab)

    if not pred_list:
        raise ValueError("fit_z: no (lab_id, action) pairs to tune")

    z_list = [z_grid] * len(pred_list)

    if len(pred_list) == 1:
        res = tune_z(pred_list[0], sol_list[0], z_list[0])
        return res.reset_index(drop=True)

    print("More than one (lab_id, action) pair, tuning z in parallel")
    max_workers = min(mp.cpu_count(), len(pred_list))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(tune_z, pred_list, sol_list, z_list))

    return pd.concat(results, ignore_index=True)


def tune_z_per_lab(
    pred: pd.DataFrame,
    sol: pd.DataFrame,
    z_grid: np.ndarray,
) -> pd.DataFrame:
    lab = pred["lab_id"].iloc[0]

    scores_arr = pred["score"].to_numpy(dtype=np.float32, copy=False)
    z_grid = np.asarray(z_grid, dtype=np.float32)

    zmin = float(scores_arr.min())
    zmax = float(scores_arr.max())

    mask = (z_grid >= zmin) & (z_grid <= zmax)
    zg = z_grid[mask]

    if zg.size == 0:
        zg = np.array([zmin], dtype=np.float32)

    print("Tuning for lab_id:", lab, "with z range:", min(zg), "-", max(zg))
    best_z = float(zg[0])
    best_s = -np.inf

    for z in zg:
        sub = filter_preds_by_thr(pred.copy(), logit_thr=float(z))
        sub = get_intervals(sub)

        if sub.empty:
            s = 0.0
        else:
            score_dict = mouse_fbeta_by_action(sol, sub)
            s = score_dict["macro_f_beta"]

        if np.isfinite(s) and s > best_s:
            best_s = s
            best_z = float(z)

    result = pd.DataFrame(
        {
            "lab_id": [lab],
            "z_thr": np.array([best_z], dtype=np.float32),
            "score": np.array([best_s], dtype=np.float32),
        }
    )
    return result

def fit_z_per_lab(
    preds_dt: pd.DataFrame,
    solution: pd.DataFrame,
    roll_frames: Optional[Any],
    z_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Fit one threshold z_thr per lab_id.
    Returns: lab_id, z_thr, score.
    """
    preds_dt = preds_dt.copy()
    preds_dt["score"] = preds_dt["score"].astype("float32")
    z_grid = np.asarray(z_grid, dtype=np.float32)

    preds_dt = postprocess_preds(preds_dt, roll_frames=roll_frames)

    sol_by_lab = dict(tuple(solution.groupby("lab_id")))
    tasks_pred = []
    tasks_sol = []

    for lab, pred_lab in preds_dt.groupby("lab_id"):
        tasks_pred.append(pred_lab)
        tasks_sol.append(sol_by_lab[lab])

    z_list = [z_grid] * len(tasks_pred)

    if len(tasks_pred) == 1:
        res = tune_z_per_lab(tasks_pred[0], tasks_sol[0], z_grid)
        return res.reset_index(drop=True)

    print("More than one lab_id, tuning z in parallel")
    max_workers = min(mp.cpu_count(), len(tasks_pred))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(tune_z_per_lab, tasks_pred, tasks_sol, z_list)
        )

    return pd.concat(results, ignore_index=True)