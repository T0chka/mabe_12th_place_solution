"""
Scoring utilities.

The functions below mirror the logic used for official evaluation, but:
- are a bit faster than the official scoring script;
- include comprehensive optional validation of input data;
- output more information (per action scores with TP, FP, FN, and Fβ + by lab scores).
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .postprocessing import filter_preds_by_thr, get_intervals

__all__ = [
    "HostVisibleError",
    "validate_for_scoring",
    "single_lab_stats",
    "mouse_fbeta_by_action",
]

REQUIRED_SUBMISSION_COLS = [
    "video_id",
    "agent_id",
    "target_id",
    "action",
    "start_frame",
    "stop_frame",
]
REQUIRED_SOLUTION_COLS = REQUIRED_SUBMISSION_COLS + ["behaviors_labeled", "lab_id"]
KEY_COLS = ["video_id", "agent_id", "target_id", "action"]


class HostVisibleError(RuntimeError):
    """Error exposed to Kaggle host when validation fails."""

def validate_for_scoring(solution: pd.DataFrame, submission: pd.DataFrame) -> None:
    """Validate solution/submission tables for metric computation."""
    if solution is None or solution.empty:
        raise HostVisibleError("Empty solution is not allowed for scoring")
    if submission is None or submission.empty:
        raise HostVisibleError("Empty submission is not allowed for scoring")

    missing_sub = [c for c in REQUIRED_SUBMISSION_COLS if c not in submission.columns]
    if missing_sub:
        raise HostVisibleError(
            f"Submission is missing columns: {', '.join(missing_sub)}"
        )

    missing_sol = [c for c in REQUIRED_SOLUTION_COLS if c not in solution.columns]
    if missing_sol:
        raise HostVisibleError(
            f"Solution is missing columns: {', '.join(missing_sol)}"
        )

    if submission[REQUIRED_SUBMISSION_COLS].isna().any().any():
        raise HostVisibleError("Submission contains NA in required columns")
    if solution[REQUIRED_SOLUTION_COLS].isna().any().any():
        raise HostVisibleError("Solution contains NA in required columns")

    if (submission["start_frame"] >= submission["stop_frame"]).any():
        raise HostVisibleError("Submission has intervals with start_frame >= stop_frame")
    if (solution["start_frame"] >= solution["stop_frame"]).any():
        raise HostVisibleError("Solution has intervals with start_frame >= stop_frame")

    sol_video_ids = set(solution["video_id"].unique())
    if not set(submission["video_id"].unique()).issubset(sol_video_ids):
        raise HostVisibleError("Submission contains unknown video_id")

    bl_counts = solution.groupby("video_id", sort=False)["behaviors_labeled"].nunique(dropna=False)
    if (bl_counts > 1).any():
        raise HostVisibleError("Inconsistent behaviors_labeled within the same video_id")

    if _has_overlap(submission):
        raise HostVisibleError("Multiple predictions for the same frame in submission")
    if _has_overlap(solution):
        raise HostVisibleError("Multiple predictions for the same frame in solution")

def _has_overlap(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    sort_cols = ["video_id", "agent_id", "target_id", "start_frame", "stop_frame"]
    df_sorted = df.sort_values(sort_cols, kind="mergesort")
    for _, group in df_sorted.groupby(["video_id", "agent_id", "target_id"], sort=False):
        if len(group) <= 1:
            continue
        starts = group["start_frame"].to_numpy(np.int64, copy=False)
        stops = group["stop_frame"].to_numpy(np.int64, copy=False)
        if np.any(starts[1:] < np.maximum.accumulate(stops[:-1])):
            return True
    return False

def _active_labels_by_video(lab_solution: pd.DataFrame) -> Dict[str, set]:
    active: Dict[str, set] = {}
    first_rows = (
        lab_solution[["video_id", "behaviors_labeled"]]
        .drop_duplicates("video_id")
        .itertuples(index=False)
    )
    for video_id, raw in first_rows:
        if isinstance(raw, str) and raw.strip():
            active[video_id] = set(json.loads(raw))
        else:
            active[video_id] = set()
    return active


def _sum_overlap(a: np.ndarray, b: np.ndarray) -> int:
    """Compute total intersection length between two sorted interval arrays."""
    total = 0
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        start = max(a[i, 0], b[j, 0])
        end = min(a[i, 1], b[j, 1])
        if start < end:
            total += int(end - start)
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return total


def single_lab_stats(
    lab_solution: pd.DataFrame,
    lab_submission: pd.DataFrame,
    beta: float = 1.0,
) -> pd.DataFrame:
    """Compute per-action TP/FP/FN/Fβ for a single lab using interval arithmetic."""
    if lab_solution.empty:
        return pd.DataFrame(columns=["action", "tp", "fp", "fn", "f_beta"])

    active = _active_labels_by_video(lab_solution)

    if not lab_submission.empty:
        video_ids = lab_submission["video_id"].to_numpy()
        agent_ids = lab_submission["agent_id"].to_numpy()
        target_ids = lab_submission["target_id"].to_numpy()
        actions = lab_submission["action"].to_numpy()

        keep_mask = np.zeros(len(lab_submission), dtype=bool)
        for i in range(len(lab_submission)):
            labels = active.get(video_ids[i])
            if not labels:
                continue
            tri = f"{agent_ids[i]},{target_ids[i]},{actions[i]}"
            if tri in labels:
                keep_mask[i] = True

        lab_submission = lab_submission.loc[keep_mask]

    label_df = lab_solution.loc[
        :, KEY_COLS + ["start_frame", "stop_frame"]
    ].sort_values(KEY_COLS + ["start_frame"], kind="mergesort")

    label_groups: Dict[Tuple, np.ndarray] = {}
    for key, group in label_df.groupby(KEY_COLS, sort=False):
        label_groups[key] = group[["start_frame", "stop_frame"]].to_numpy(
            dtype=np.int64, copy=False
        )

    pred_groups: Dict[Tuple, np.ndarray] = {}
    if not lab_submission.empty:
        pred_df = lab_submission.loc[
            :, KEY_COLS + ["start_frame", "stop_frame"]
        ].sort_values(KEY_COLS + ["start_frame"], kind="mergesort")

        for key, group in pred_df.groupby(KEY_COLS, sort=False):
            pred_groups[key] = group[["start_frame", "stop_frame"]].to_numpy(
                dtype=np.int64, copy=False
            )

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)

    for key, label_arr in label_groups.items():
        action = key[3]
        pred_arr = pred_groups.get(key)

        label_len = int((label_arr[:, 1] - label_arr[:, 0]).sum())
        if pred_arr is not None:
            pred_len = int((pred_arr[:, 1] - pred_arr[:, 0]).sum())
            tp = _sum_overlap(label_arr, pred_arr)
        else:
            pred_len = 0
            tp = 0

        fn = label_len - tp
        fp = pred_len - tp

        tps[action] += tp
        fns[action] += fn
        fps[action] += fp

    for key, pred_arr in pred_groups.items():
        if key in label_groups:
            continue
        action = key[3]
        pred_len = int((pred_arr[:, 1] - pred_arr[:, 0]).sum())
        fps[action] += pred_len

    distinct_actions = set(lab_solution["action"].unique())
    if not distinct_actions:
        return pd.DataFrame(columns=["action", "tp", "fp", "fn", "f_beta"])

    beta2 = float(beta) * float(beta)
    rows: List[Dict[str, Any]] = []

    for action in sorted(distinct_actions):
        tp = tps.get(action, 0)
        fn = fns.get(action, 0)
        fp = fps.get(action, 0)

        if tp + fn + fp == 0:
            f_beta = 0.0
        else:
            denom = (1.0 + beta2) * tp + beta2 * fn + fp
            f_beta = (1.0 + beta2) * tp / denom if denom else 0.0

        rows.append(
            {
                "action": action,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f_beta": f_beta,
            }
        )

    return pd.DataFrame(rows)


def mouse_fbeta_by_action(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    beta: float = 1.0,
    validate: bool = True,
):
    """Compute per-lab and per-action Fβ with macro aggregation by lab."""
    if validate:
        validate_for_scoring(solution, submission)

    vids = set(solution["video_id"].unique())
    submission = submission[submission["video_id"].isin(vids)]

    labs = solution["lab_id"].unique()
    all_stats: List[pd.DataFrame] = []
    all_stats_append = all_stats.append

    lab_vids_map: Dict[Any, set] = {}
    for lab in labs:
        lab_mask = solution["lab_id"] == lab
        lab_vids_map[lab] = set(solution.loc[lab_mask, "video_id"].unique())

    for lab in labs:
        lab_mask = solution["lab_id"] == lab
        lab_solution = solution.loc[lab_mask]
        lab_vids = lab_vids_map[lab]
        lab_submission = submission[submission["video_id"].isin(lab_vids)]

        stats = single_lab_stats(lab_solution, lab_submission, beta=beta)
        if not stats.empty:
            stats["lab_id"] = lab
        else:
            stats = pd.DataFrame(
                {"action": [], "tp": [], "fp": [], "fn": [], "f_beta": [], "lab_id": []}
            )
        all_stats_append(stats)

    if all_stats:
        all_stats_df = pd.concat(all_stats, ignore_index=True)
    else:
        all_stats_df = pd.DataFrame(
            columns=["action", "tp", "fp", "fn", "f_beta", "lab_id"]
        )

    if len(all_stats_df):
        macro_by_lab = all_stats_df.groupby("lab_id", as_index=False)["f_beta"].mean()
        macro_by_action = all_stats_df.groupby("action", as_index=False)["f_beta"].mean()
        macro_f_beta = float(macro_by_lab["f_beta"].mean())
    else:
        macro_by_lab = pd.DataFrame(columns=["lab_id", "f_beta"])
        macro_by_action = pd.DataFrame(columns=["action", "f_beta"])
        macro_f_beta = 0.0

    return {
        "all_stats": all_stats_df,
        "by_action": macro_by_action,
        "by_lab": macro_by_lab,
        "macro_f_beta": macro_f_beta,
    }


def get_scores(
    fitted: Dict[str, Any],
    train_labels: pd.DataFrame,
    whitelist: pd.DataFrame,
    act_duration: pd.DataFrame,
    verbose: bool = False,
    validate: bool = False,
) -> Dict[str, Any]:
    """
    Compute scores from fitted model.
    
    Args:
        fitted: Fitted model results with 'oof' DataFrame
        train_labels: Training labels DataFrame
        whitelist: Whitelist DataFrame
        act_duration: Action duration DataFrame for smoothing
        verbose: Print verbose output
    
    Returns:
        Dict with 'scores' and 'z_by_lab':
            'scores': Dict with 'f_beta_raw' and 'f_beta_postproc'
            'z_by_lab': DataFrame with thresholds by lab and action
    """
    from py_pipeline.utils.data_loads import build_solution, prepare_submit
    
    train_solution = build_solution(train_labels, whitelist)
    
    # Raw OOF scores
    pred_oof = fitted["oof"].copy()
    pred_oof = filter_preds_by_thr(pred_oof, logit_thr=None)
    submit_oof = get_intervals(pred_oof)
    oof_scores_raw = mouse_fbeta_by_action(train_solution, submit_oof, validate=validate)
    oof_f_beta_raw = oof_scores_raw["macro_f_beta"]
    
    # Postprocessed OOF scores using prepare_submit
    lab_name = train_labels["lab_id"].iloc[0]
    pred_oof = fitted["oof"].copy()
    pred_oof["lab_id"] = lab_name
    
    res = prepare_submit(
        preds_dt=pred_oof,
        solution=train_solution,
        whitelist=whitelist,
        roll_frames=act_duration,
        z_grid=np.arange(-11, 20.1, 0.1)
    )
    
    oof_scores_postproc = mouse_fbeta_by_action(train_solution, res["submit"], validate=validate)
    oof_f_beta_postproc = oof_scores_postproc["macro_f_beta"]
    
    if verbose:
        print("\n======= OOF scores =======")
        print(f"\nOOF shape: {fitted['oof'].shape}")
        print(f"OOF actions: {fitted['oof']['action'].value_counts().to_dict()}")
        
        print("\nRaw scores (no thresholds):")
        print(oof_scores_raw["all_stats"])
        print(f"macro_f_beta_raw: {oof_scores_raw['macro_f_beta']:.6f}")
        
        print("\nPostprocessed scores (with thresholds):")
        print(oof_scores_postproc["all_stats"])
        print(f"macro_f_beta_postproc: {oof_scores_postproc['macro_f_beta']:.6f}")
        
        if res["z_by_lab"] is not None:
            print("\nThresholds:")
            print(res["z_by_lab"])
        print()
    
    # Extract action scores
    act_scores = {}
    if "by_action" in oof_scores_postproc and not oof_scores_postproc["by_action"].empty:
        for _, row in oof_scores_postproc["by_action"].iterrows():
            act_scores[row["action"]] = row["f_beta"]
    
    all_scores = {}
    
    all_scores.update({
        "f_beta_raw": oof_f_beta_raw,
        "f_beta_postproc": oof_f_beta_postproc,
    })
    all_scores.update(act_scores)
    
    return {"scores": all_scores, "z_by_lab": res["z_by_lab"]}
