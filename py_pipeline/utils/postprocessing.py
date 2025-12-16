"""
Postprocessing functions.
"""
from typing import Any, Optional

import numpy as np
import pandas as pd

def postprocess_preds(
    pred_dt: pd.DataFrame,
    roll_frames: Optional[Any] = None
) -> pd.DataFrame:
    """
    Post-process predictions with rolling mean smoothing.
    """
    if roll_frames is None:
        return pred_dt

    if isinstance(roll_frames, (int, float)):
        pred_dt = pred_dt.copy()
        pred_dt["roll_frames"] = int(roll_frames)
    elif isinstance(roll_frames, pd.DataFrame):
        pred_dt = pred_dt.merge(
            roll_frames,
            on=["lab_id", "action"],
            how="left"
        )
    else:
        raise ValueError("roll_frames must be int/float or DataFrame")

    # Convert to odd number of frames
    pred_dt["roll_frames"] = (pred_dt["roll_frames"] // 2) * 2 + 1

    pred_dt = pred_dt.sort_values(
        ["video_id", "agent_id", "target_id", "action", "video_frame"]
    )

    group_cols = ["video_id", "agent_id", "target_id", "action"]

    roll_vals = pred_dt["roll_frames"].dropna().unique()

    for w in roll_vals:
        subset = pred_dt.loc[pred_dt["roll_frames"] == w]

        g = subset.groupby(group_cols)["score"]

        sm = g.rolling(
            window=int(w),
            center=True,
            min_periods=int(w)
        ).mean()

        sm.index = sm.index.get_level_values(-1)
        sm = sm.astype("float32")

        valid = sm.notna()
        if not valid.any():
            continue

        idx_valid = sm.index[valid]
        pred_dt.loc[idx_valid, "score"] = sm[valid]

    pred_dt = pred_dt.drop(columns=["roll_frames"])

    return pred_dt


def filter_preds_by_thr(
    pred_dt: pd.DataFrame,
    logit_thr: Optional[Any] = None
) -> pd.DataFrame:
    """
    Filter predictions by threshold and keep only max score per frame.
    logit_thr can be a single threshold value or a DataFrame with thresholds
    """
    if logit_thr is not None:
        if isinstance(logit_thr, (int, float)):
            pred_dt = pred_dt.assign(z_thr=logit_thr)
        elif isinstance(logit_thr, pd.DataFrame):
            on_cols = list(set(pred_dt.columns) & set(logit_thr.columns))
            pred_dt = pred_dt.merge(
                logit_thr, on=on_cols, how="left", suffixes=("", "_thr")
            )

        pred_dt = pred_dt[pred_dt["score"] > pred_dt["z_thr"]]
        pred_dt = pred_dt.drop(columns=["z_thr"], errors="ignore")

    max_scores = pred_dt.groupby(
        ["video_id", "agent_id", "target_id", "video_frame"]
    )["score"].transform("max")

    pred_dt = pred_dt[pred_dt["score"] == max_scores]
    pred_dt = pred_dt.drop_duplicates(
        subset=["video_id", "agent_id", "target_id", "video_frame"],
        keep="first"
    )

    return pred_dt


def get_intervals(pred_dt: pd.DataFrame) -> pd.DataFrame:
    """
    Get intervals from per frame predictions.
    """
    pred_dt = pred_dt.sort_values(
        ["video_id", "agent_id", "target_id", "video_frame"]
    )

    vid = pred_dt["video_id"]
    ag = pred_dt["agent_id"]
    tg = pred_dt["target_id"]
    act = pred_dt["action"]
    frame = pred_dt["video_frame"]

    new_group = vid.ne(vid.shift()) | ag.ne(ag.shift()) | tg.ne(tg.shift())
    new_action = act.ne(act.shift())
    frame_gap = frame.ne(frame.shift() + 1)

    run_id = (new_group | new_action | frame_gap).cumsum()
    pred_dt = pred_dt.assign(run_id=run_id)

    intervals = (
        pred_dt.groupby(
            ["video_id", "agent_id", "target_id", "run_id"], sort=False
        )
        .agg(
            action=("action", "first"),
            start_frame=("video_frame", "first"),
            stop_frame=("video_frame", "last"),
        )
        .reset_index()
    )

    intervals["stop_frame"] = intervals["stop_frame"] + 1

    intervals = intervals[
        (intervals["stop_frame"] - intervals["start_frame"]) >= 2
    ]

    intervals = intervals.drop(columns=["run_id"])

    intervals["agent_id"] = "mouse" + intervals["agent_id"].astype(str)
    intervals["target_id"] = "mouse" + intervals["target_id"].astype(str)

    intervals = intervals[
        ["video_id", "agent_id", "target_id",
         "action", "start_frame", "stop_frame"]
    ]

    return intervals

