"""
Feature computation functions - v.2.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd


def _prepare_group_index(dt: pd.DataFrame, by_cols: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Build positional index and per-row group offsets for lag computations.
    """
    idx = np.arange(len(dt), dtype=np.int64)
    cumcount = dt.groupby(by_cols, sort=False, observed=True).cumcount()
    return idx, cumcount.to_numpy(dtype=np.int64)


def _lag_index(idx: np.ndarray, steps: np.ndarray, cumcount: np.ndarray) -> np.ndarray:
    """
    Compute row indices for lagged references using per-row step sizes.
    """
    steps = steps.astype(np.int64, copy=False)
    valid = (steps > 0) & (cumcount >= steps)
    lag_idx = idx - steps
    lag_idx[~valid] = -1
    return lag_idx


def _lagged_delta(
    values: np.ndarray,
    lag_idx: np.ndarray,
    mask: Optional[np.ndarray] = None,
    angle: bool = False
) -> np.ndarray:
    """
    Difference between current value and its lag reference.
    """
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = lag_idx >= 0
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return result
    prev_idx = lag_idx[valid]
    delta = values[valid] - values[prev_idx]
    if angle:
        finite = np.isfinite(values[valid]) & np.isfinite(values[prev_idx])
        if not np.any(finite):
            return result
        angle_valid_idx = np.where(valid)[0][finite]
        wrapped = np.arctan2(
            np.sin(delta[finite]),
            np.cos(delta[finite])
        )
        result[angle_valid_idx] = wrapped
        return result
    result[valid] = delta
    return result


def _lagged_rate(
    values: np.ndarray,
    lag_idx: np.ndarray,
    scale: np.ndarray,
    mask: Optional[np.ndarray] = None,
    angle: bool = False
) -> np.ndarray:
    """
    Scaled lagged delta, typically used for velocities and accelerations.
    """
    delta = _lagged_delta(values, lag_idx, mask=mask, angle=angle)
    return delta * scale

def normalize_xy(dt: pd.DataFrame, dt_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize coordinates from pixels to centimeters.
    Divides all coordinate columns by pix_per_cm_approx for each video_id.
    """
    dt = dt.copy()
    
    # Merge pix_per_cm_approx
    dt = dt.merge(
        dt_meta[["video_id", "pix_per_cm_approx"]],
        on="video_id",
        how="left"
    )
    
    # Find coordinate columns
    x_cols = [c for c in dt.columns if re.search(r"(^|_)x($|_)", c)]
    y_cols = [c for c in dt.columns if re.search(r"(^|_)y($|_)", c)]
    
    # Normalize coordinates
    for col in x_cols + y_cols:
        dt[col] = dt[col] / dt["pix_per_cm_approx"]
    
    dt = dt.drop(columns=["pix_per_cm_approx"])
    return dt


def bidir_fill_na(dt: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values bidirectionally (forward then backward).

    """
    dt = dt.sort_values(["video_id", "agent_id", "target_id", "video_frame"])
    
    num_cols = dt.select_dtypes(include=[np.number, bool]).columns
    
    for col in num_cols:
        dt[col] = (
            dt.groupby(["video_id", "agent_id", "target_id"])[col]
            .ffill()
            .bfill()
            .fillna(0)
        )
    
    return dt


def compute_center_and_length(dt: pd.DataFrame, role: str) -> None:
    """
    Compute body center, length, and orientation unit vector.
    Modifies dt in-place.
    """
    assert role in ["ag", "tg"]
    prefix_x = f"{role}_x_"
    prefix_y = f"{role}_y_"
    
    upper_parts = [
        "nose", "head", "ear_left", "ear_right", "neck",
        "headpiece_bottombackleft", "headpiece_bottombackright",
        "headpiece_bottomfrontleft", "headpiece_bottomfrontright",
        "headpiece_topbackleft", "headpiece_topbackright",
        "headpiece_topfrontleft", "headpiece_topfrontright"
    ]
    lower_parts = ["tail_base", "hip_left", "hip_right"]
    other_parts = ["body_center", "lateral_left", "lateral_right", "spine_1", "spine_2"]
    parts_all = upper_parts + lower_parts + other_parts
    
    cols_x_all = [c for c in dt.columns if c in [prefix_x + p for p in parts_all]]
    cols_y_all = [c for c in dt.columns if c in [prefix_y + p for p in parts_all]]
    cols_x_upper = [c for c in dt.columns if c in [prefix_x + p for p in upper_parts]]
    cols_y_upper = [c for c in dt.columns if c in [prefix_y + p for p in upper_parts]]
    cols_x_lower = [c for c in dt.columns if c in [prefix_x + p for p in lower_parts]]
    cols_y_lower = [c for c in dt.columns if c in [prefix_y + p for p in lower_parts]]
    
    dt[f"{role}_x_center"] = dt[cols_x_all].mean(axis=1)
    dt[f"{role}_y_center"] = dt[cols_y_all].mean(axis=1)
    dt[f"{role}_x_upper"] = dt[cols_x_upper].mean(axis=1)
    dt[f"{role}_y_upper"] = dt[cols_y_upper].mean(axis=1)
    dt[f"{role}_x_lower"] = dt[cols_x_lower].mean(axis=1)
    dt[f"{role}_y_lower"] = dt[cols_y_lower].mean(axis=1)
    
    dx = dt[f"{role}_x_upper"] - dt[f"{role}_x_lower"]
    dy = dt[f"{role}_y_upper"] - dt[f"{role}_y_lower"]
    body_len_col = f"{role}_body_len"
    dt[body_len_col] = np.sqrt(dx**2 + dy**2)
    min_len = np.array(1e-6, dtype=dt[body_len_col].dtype)
    dt.loc[dt[body_len_col] < min_len, body_len_col] = min_len
    
    valid = dt[f"{role}_body_len"].notna()
    dt.loc[valid, f"{role}_unit_x"] = dx[valid] / dt.loc[valid, f"{role}_body_len"]
    dt.loc[valid, f"{role}_unit_y"] = dy[valid] / dt.loc[valid, f"{role}_body_len"]


def add_lag_structure(
    dt: pd.DataFrame,
    dt_meta: pd.DataFrame,
    lag_sec: float,
    by_cols: list
) -> pd.DataFrame:
    """
    Add lag structure: kf, ok1, ok2, ok3 masks for temporal features.
    Returns modified DataFrame.
    """
    dt = dt.merge(
        dt_meta[["video_id", "frames_per_second"]],
        on="video_id",
        how="left"
    )
    dt.rename(columns={"frames_per_second": "fps"}, inplace=True)
    
    dt = dt.sort_values(by_cols + ["video_frame"]).reset_index(drop=True)
    
    dt["kf"] = dt.groupby(by_cols)["fps"].transform(
        lambda x: np.maximum(1, np.round(x.iloc[0] * lag_sec).astype(int))
    )
    
    for col in by_cols:
        if col not in dt.columns:
            raise ValueError(f"Column {col} not found in dt")
    
    idx, cumcount = _prepare_group_index(dt, by_cols)
    kf_vals = dt["kf"].to_numpy(dtype=np.int64)
    lag1_idx = _lag_index(idx, kf_vals, cumcount)
    
    video_vals = dt["video_frame"].to_numpy(dtype=np.float64)
    frame_gap = np.full(len(dt), np.nan, dtype=np.float64)
    valid_lag1 = lag1_idx >= 0
    prev_idx = lag1_idx[valid_lag1]
    frame_gap[valid_lag1] = video_vals[valid_lag1] - video_vals[prev_idx]
    
    ok1 = np.zeros(len(dt), dtype=bool)
    ok1[valid_lag1] = frame_gap[valid_lag1] == kf_vals[valid_lag1]
    
    ok1_shift = np.zeros(len(dt), dtype=bool)
    ok1_shift[valid_lag1] = ok1[prev_idx]
    ok2 = ok1 & ok1_shift
    
    lag2_idx = _lag_index(idx, 2 * kf_vals, cumcount)
    valid_lag2 = lag2_idx >= 0
    ok1_shift2 = np.zeros(len(dt), dtype=bool)
    ok1_shift2[valid_lag2] = ok1[lag2_idx[valid_lag2]]
    ok3 = ok2 & ok1_shift2
    
    dt["ok1"] = ok1
    dt["ok2"] = ok2
    dt["ok3"] = ok3
    return dt


def compute_invariants(
    dt: pd.DataFrame,
    dt_meta: pd.DataFrame,
    args: dict
) -> pd.DataFrame:
    """
    Compute invariant features in agent-centric frame.
    """
    norm_by = args["norm_by"]
    lag_sec = args["lag_sec"]
    by_cols = ["aug", "video_id", "agent_id", "target_id"]
    
    dt = dt.copy()
    
    compute_center_and_length(dt, "ag")
    compute_center_and_length(dt, "tg")
    
    # Reference body length for normalization
    if norm_by == "med_body_len":
        dt["ref_body_len"] = dt.groupby(by_cols)["ag_body_len"].transform("median")
        dt.loc[dt["ref_body_len"] < 1e-9, "ref_body_len"] = 1e-9
    elif norm_by == "frame_body_len":
        dt["ref_body_len"] = dt["ag_body_len"]
    else:
        dt["ref_body_len"] = 1.0
    
    # Set origin as agent body center
    dt["origin_x"] = dt["ag_x_center"]
    dt["origin_y"] = dt["ag_y_center"]
    
    # Relative target position
    dt["dx_t"] = dt["tg_x_center"] - dt["origin_x"]
    dt["dy_t"] = dt["tg_y_center"] - dt["origin_y"]
    dt["dx_h"] = dt["tg_x_upper"] - dt["origin_x"]
    dt["dy_h"] = dt["tg_y_upper"] - dt["origin_y"]
    
    # Rotate to agent-centric frame
    dt["rel_x_t"] = dt["dx_t"] * dt["ag_unit_x"] + dt["dy_t"] * dt["ag_unit_y"]
    dt["rel_y_t"] = -dt["dx_t"] * dt["ag_unit_y"] + dt["dy_t"] * dt["ag_unit_x"]
    dt["rel_x_h"] = dt["dx_h"] * dt["ag_unit_x"] + dt["dy_h"] * dt["ag_unit_y"]
    dt["rel_y_h"] = -dt["dx_h"] * dt["ag_unit_y"] + dt["dy_h"] * dt["ag_unit_x"]
    
    # Orientation angles
    dt["angle_ag"] = np.arctan2(dt["ag_unit_y"], dt["ag_unit_x"])
    dt["angle_tg"] = np.arctan2(dt["tg_unit_y"], dt["tg_unit_x"])
    
    # Build output with static invariants
    key_cols = ["aug", "video_id", "agent_id", "target_id", "video_frame"]
    if "action" in dt.columns:
        key_cols.append("action")
    
    out = dt[key_cols].copy()
    out["r_center"] = np.sqrt(dt["rel_x_t"]**2 + dt["rel_y_t"]**2) / dt["ref_body_len"]
    out["r_head"] = np.sqrt(dt["rel_x_h"]**2 + dt["rel_y_h"]**2) / dt["ref_body_len"]
    out["theta_center"] = np.arctan2(dt["rel_y_t"], dt["rel_x_t"])
    out["theta_head"] = np.arctan2(dt["rel_y_h"], dt["rel_x_h"])
    out["cos_dpsi"] = np.cos(dt["angle_tg"] - dt["angle_ag"])
    out["sin_dpsi"] = np.sin(dt["angle_tg"] - dt["angle_ag"])
    out["long"] = dt["rel_x_t"] / dt["ref_body_len"]
    out["lat"] = dt["rel_y_t"] / dt["ref_body_len"]
    out["origin_x"] = dt["origin_x"]
    out["origin_y"] = dt["origin_y"]
    out["unit_x_ag"] = dt["ag_unit_x"]
    out["unit_y_ag"] = dt["ag_unit_y"]
    out["ref_body_len"] = dt["ref_body_len"]
    
    # Add lag structure
    out = add_lag_structure(out, dt_meta, lag_sec, by_cols)
    
    idx, cumcount = _prepare_group_index(out, by_cols)
    kf_vals = out["kf"].to_numpy(dtype=np.int64)
    fps_vals = out["fps"].to_numpy(dtype=np.float64)
    lag1_idx = _lag_index(idx, kf_vals, cumcount)
    scale = fps_vals / np.maximum(kf_vals, 1)
    ok1 = out["ok1"].to_numpy(dtype=bool)
    ok2 = out["ok2"].to_numpy(dtype=bool)
    
    r_center_vals = out["r_center"].to_numpy(dtype=np.float64)
    r_head_vals = out["r_head"].to_numpy(dtype=np.float64)
    theta_vals = out["theta_center"].to_numpy(dtype=np.float64)
    long_vals = out["long"].to_numpy(dtype=np.float64)
    lat_vals = out["lat"].to_numpy(dtype=np.float64)
    origin_x = out["origin_x"].to_numpy(dtype=np.float64)
    origin_y = out["origin_y"].to_numpy(dtype=np.float64)
    unit_x = out["unit_x_ag"].to_numpy(dtype=np.float64)
    unit_y = out["unit_y_ag"].to_numpy(dtype=np.float64)
    ref_len = out["ref_body_len"].to_numpy(dtype=np.float64)
    
    dr_center = _lagged_rate(r_center_vals, lag1_idx, scale, mask=ok1)
    d2r_center = _lagged_rate(dr_center, lag1_idx, scale, mask=ok2)
    dtheta_center = _lagged_rate(theta_vals, lag1_idx, scale, mask=ok1, angle=True)
    d2dtheta_center = _lagged_rate(dtheta_center, lag1_idx, scale, mask=ok2)
    v_long = _lagged_rate(long_vals, lag1_idx, scale, mask=ok1)
    v_lat = _lagged_rate(lat_vals, lag1_idx, scale, mask=ok1)
    
    scale_body = scale / np.maximum(ref_len, 1e-9)
    delta_x = _lagged_delta(origin_x, lag1_idx, mask=ok1)
    delta_y = _lagged_delta(origin_y, lag1_idx, mask=ok1)
    v_forward_a = (delta_x * unit_x + delta_y * unit_y) * scale_body
    v_lateral_a = ((-delta_x * unit_y) + (delta_y * unit_x)) * scale_body
    a_long = _lagged_rate(v_long, lag1_idx, scale, mask=ok2)
    a_lat = _lagged_rate(v_lat, lag1_idx, scale, mask=ok2)
    head_approach = _lagged_rate(r_head_vals, lag1_idx, scale, mask=ok1)
    
    out["dr_center"] = dr_center
    out["d2r_center"] = d2r_center
    out["dtheta_center"] = dtheta_center
    out["d2dtheta_center"] = d2dtheta_center
    out["v_long"] = v_long
    out["v_lat"] = v_lat
    out["v_forward_a"] = v_forward_a
    out["v_lateral_a"] = v_lateral_a
    out["a_long"] = a_long
    out["a_lat"] = a_lat
    out["head_approach"] = head_approach
    
    # Derived features (computed on all rows, NaN propagates naturally)
    out["los"] = np.sqrt(np.maximum(out["long"]**2 + out["lat"]**2, 1e-12))
    out["u_x"] = out["long"] / out["los"]
    out["u_y"] = out["lat"] / out["los"]
    out["v_a"] = np.sqrt(np.maximum(out["v_forward_a"]**2 + out["v_lateral_a"]**2, 1e-12))
    out["v_rel"] = np.sqrt(np.maximum(out["v_long"]**2 + out["v_lat"]**2, 1e-12))
    out["align_agent_los"] = (out["v_forward_a"] * out["u_x"] + out["v_lateral_a"] * out["u_y"]) / out["v_a"]
    out["align_target_los"] = (out["v_long"] * out["u_x"] + out["v_lat"] * out["u_y"]) / out["v_rel"]
    out["ttc_inv"] = np.maximum(0, -out["dr_center"]) / np.maximum(out["r_center"], 1e-6)
    out["log_ttc_inv"] = np.log10(out["ttc_inv"] + 1e-12)
    
    # turn_toward uses dtheta_center which is already masked by ok1
    out["turn_toward"] = -np.sign(out["theta_center"]) * out["dtheta_center"]
    
    # Trig features (theta_center is static, always valid)
    out["cos_theta"] = np.cos(out["theta_center"])
    out["sin_theta"] = np.sin(out["theta_center"])
    
    # approach_dir requires ok1 and dr_center (which is masked by ok1)
    out["approach_dir"] = ((out["ok1"]) & (out["dr_center"] < 0) & (out["cos_theta"] > 0)).astype(int)
    
    tau_window = (2 * kf_vals).astype(int)
    frac_tau = np.full(len(out), np.nan, dtype=np.float64)
    for window in np.unique(tau_window):
        mask = tau_window == window
        subset = out.loc[mask, by_cols + ["approach_dir"]]
        if subset.empty:
            continue
        if window <= 1:
            frac_tau[mask] = subset["approach_dir"].to_numpy(dtype=np.float64)
            continue
        rolled = subset.groupby(
            by_cols,
            sort=False,
            observed=True
        )["approach_dir"].transform(
            lambda x, win=window: x.rolling(
                window=win,
                min_periods=1,
                center=False
            ).mean()
        )
        frac_tau[mask] = rolled.to_numpy(dtype=np.float64)
    out["frac_approach_tau"] = frac_tau
    
    is_contact = (out["r_center"] < 1.0).astype(int)
    contact_frame = out["video_frame"].where(is_contact == 1, np.nan)
    contact_meta = out[by_cols].copy()
    contact_meta["contact_frame"] = contact_frame
    last_contact = contact_meta.groupby(
        by_cols,
        sort=False,
        observed=True
    )["contact_frame"].ffill()
    out["t_since_contact"] = (
        (out["video_frame"] - last_contact) / out["fps"]
    )
    
    out["motion_parallel"] = (
        (out["v_long"] * out["v_forward_a"] + out["v_lat"] * out["v_lateral_a"]) /
        (np.sqrt(out["v_long"]**2 + out["v_lat"]**2) * 
         np.sqrt(out["v_forward_a"]**2 + out["v_lateral_a"]**2) + 1e-6)
    )
    
    # Select final columns
    final_cols = [
        "aug", "video_id", "agent_id", "target_id", "video_frame",
        "r_center", "r_head", "theta_center", "theta_head",
        "cos_dpsi", "sin_dpsi", "long", "lat", "v_long", "v_lat",
        "dr_center", "d2r_center", "dtheta_center", "d2dtheta_center",
        "align_agent_los", "align_target_los", "motion_parallel",
        "frac_approach_tau", "t_since_contact",
        "a_long", "a_lat", "turn_toward", "head_approach",
        "cos_theta", "sin_theta"
    ]
    if "action" in out.columns:
        final_cols.insert(4, "action")
    
    return out[final_cols]


def compute_single_mouse(
    dt: pd.DataFrame,
    dt_meta: pd.DataFrame,
    args: dict
) -> pd.DataFrame:
    """
    Compute single-mouse kinematic features.
    """
    norm_by = args["norm_by"]
    lag_sec = args["lag_sec"]
    
    dt = dt.sort_values(["aug", "video_id", "agent_id", "video_frame"])
    
    # Extract agent and target data
    ag_cols = [c for c in dt.columns if c.startswith("ag_x_") or c.startswith("ag_y_")]
    tg_cols = [c for c in dt.columns if c.startswith("tg_x_") or c.startswith("tg_y_")]
    
    base_ag = dt[["aug", "video_id", "agent_id", "video_frame"] + ag_cols].drop_duplicates(
        subset=["aug", "video_id", "agent_id", "video_frame"]
    ).copy()
    base_ag.rename(columns={"agent_id": "mouse_id"}, inplace=True)
    
    base_tg = dt[["aug", "video_id", "target_id", "video_frame"] + tg_cols].drop_duplicates(
        subset=["aug", "video_id", "target_id", "video_frame"]
    ).copy()
    base_tg.rename(columns={"target_id": "mouse_id"}, inplace=True)
    base_tg.columns = [c.replace("tg_", "ag_") if c.startswith("tg_") else c for c in base_tg.columns]
    
    base_dt = pd.concat([base_ag, base_tg], ignore_index=True).drop_duplicates(
        subset=["aug", "video_id", "mouse_id", "video_frame"]
    )
    
    by_cols = ["aug", "video_id", "mouse_id"]
    base_dt = base_dt.sort_values(by_cols + ["video_frame"])
    
    base_dt = add_lag_structure(base_dt, dt_meta, lag_sec, by_cols)
    compute_center_and_length(base_dt, "ag")
    
    # Set normalization reference
    if norm_by == "med_body_len":
        base_dt["ref_body_len"] = base_dt.groupby(by_cols)["ag_body_len"].transform("median")
        base_dt.loc[base_dt["ref_body_len"] < 1e-9, "ref_body_len"] = 1e-9
    elif norm_by == "frame_body_len":
        base_dt["ref_body_len"] = base_dt["ag_body_len"]
    else:
        base_dt["ref_body_len"] = 1.0
    
    # Ear gap
    if "ag_x_ear_left" in base_dt.columns and "ag_x_ear_right" in base_dt.columns:
        base_dt["ear_gap"] = (
            np.sqrt(
                (base_dt["ag_x_ear_left"] - base_dt["ag_x_ear_right"])**2 +
                (base_dt["ag_y_ear_left"] - base_dt["ag_y_ear_right"])**2
            ) / base_dt["ref_body_len"]
        )
    else:
        base_dt["ear_gap"] = np.nan
    
    idx, cumcount = _prepare_group_index(base_dt, by_cols)
    kf_vals = base_dt["kf"].to_numpy(dtype=np.int64)
    fps_vals = base_dt["fps"].to_numpy(dtype=np.float64)
    lag1_idx = _lag_index(idx, kf_vals, cumcount)
    scale = fps_vals / np.maximum(kf_vals, 1)
    ok1 = base_dt["ok1"].to_numpy(dtype=bool)
    ok2 = base_dt["ok2"].to_numpy(dtype=bool)
    
    ag_x_center = base_dt["ag_x_center"].to_numpy(dtype=np.float64)
    ag_y_center = base_dt["ag_y_center"].to_numpy(dtype=np.float64)
    unit_x = base_dt["ag_unit_x"].to_numpy(dtype=np.float64)
    unit_y = base_dt["ag_unit_y"].to_numpy(dtype=np.float64)
    ref_len = base_dt["ref_body_len"].to_numpy(dtype=np.float64)
    
    dx = _lagged_delta(ag_x_center, lag1_idx, mask=ok1)
    dy = _lagged_delta(ag_y_center, lag1_idx, mask=ok1)
    
    scale_body = scale / np.maximum(ref_len, 1e-9)
    v_forward = (dx * unit_x + dy * unit_y) * scale_body
    v_lateral = ((-dx * unit_y) + (dy * unit_x)) * scale_body
    speed_total = np.sqrt(v_forward**2 + v_lateral**2)
    
    a_forward = _lagged_rate(v_forward, lag1_idx, scale, mask=ok2)
    a_lateral = _lagged_rate(v_lateral, lag1_idx, scale, mask=ok2)
    
    ddx = _lagged_delta(dx, lag1_idx, mask=ok2)
    ddy = _lagged_delta(dy, lag1_idx, mask=ok2)
    denom = np.maximum((dx**2 + dy**2)**1.5, 1e-6)
    curvature = np.where(
        (~np.isnan(ddx)) & (~np.isnan(ddy)),
        np.abs(dx * ddy - dy * ddx) / denom * ref_len,
        np.nan
    )
    
    psi = np.arctan2(unit_y, unit_x)
    turn_agent = _lagged_rate(psi, lag1_idx, scale, mask=ok1, angle=True)
    
    w3 = 2 * ((kf_vals * 3) // 2) + 1
    ok_w3 = np.zeros(len(base_dt), dtype=bool)
    v_forward_med = np.full(len(base_dt), np.nan, dtype=np.float64)
    v_forward_sdev = np.full(len(base_dt), np.nan, dtype=np.float64)
    forward_consistency = np.full(len(base_dt), np.nan, dtype=np.float64)
    lat_ratio_med = np.full(len(base_dt), np.nan, dtype=np.float64)
    speed_med = np.full(len(base_dt), np.nan, dtype=np.float64)
    turn_smoothness = np.full(len(base_dt), np.nan, dtype=np.float64)
    
    lat_ratio = np.abs(v_lateral) / np.maximum(np.abs(v_forward), 1e-6)
    idx_range = base_dt.index
    series_ok1 = pd.Series(ok1.astype(int), index=idx_range)
    series_v_forward = pd.Series(v_forward, index=idx_range)
    series_v_lateral = pd.Series(v_lateral, index=idx_range)
    series_speed = pd.Series(speed_total, index=idx_range)
    series_turn = pd.Series(turn_agent, index=idx_range)
    series_lat_ratio = pd.Series(lat_ratio, index=idx_range)
    
    def _roll_mean(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, center=True, min_periods=1).mean()
    
    def _roll_sdev(series: pd.Series, window: int) -> pd.Series:
        mean_vals = _roll_mean(series, window)
        mad = (series - mean_vals).abs().rolling(
            window=window,
            center=True,
            min_periods=1
        ).mean()
        return 1.253314 * mad
    
    unique_w = np.unique(w3)
    for window in unique_w:
        mask = w3 == window
        if not np.any(mask):
            continue
        keys = base_dt.loc[mask, by_cols].copy()
        subset_idx = idx_range[mask]
        
        ok_block = keys.copy()
        ok_block["ok1"] = series_ok1.loc[subset_idx].to_numpy()
        ok_sum = ok_block.groupby(
            by_cols,
            sort=False,
            observed=True
        )["ok1"].transform(
            lambda x, win=window: x.rolling(
                window=win,
                center=True,
                min_periods=1
            ).sum()
        )
        ok_w3[mask] = ok_sum.eq(window).to_numpy(dtype=bool)
        
        fwd_block = keys.copy()
        fwd_block["value"] = series_v_forward.loc[subset_idx].to_numpy()
        med_vals = fwd_block.groupby(
            by_cols,
            sort=False,
            observed=True
        )["value"].transform(
            lambda x, win=window: _roll_mean(x, win)
        )
        sdev_vals = fwd_block.groupby(
            by_cols,
            sort=False,
            observed=True
        )["value"].transform(
            lambda x, win=window: _roll_sdev(x, win)
        )
        v_forward_med[mask] = med_vals.to_numpy(dtype=np.float64)
        v_forward_sdev[mask] = sdev_vals.to_numpy(dtype=np.float64)
        
        lat_block = keys.copy()
        lat_block["value"] = series_lat_ratio.loc[subset_idx].to_numpy()
        lat_vals = lat_block.groupby(
            by_cols,
            sort=False,
            observed=True
        )["value"].transform(
            lambda x, win=window: _roll_mean(x, win)
        )
        lat_ratio_med[mask] = lat_vals.to_numpy(dtype=np.float64)
        
        speed_block = keys.copy()
        speed_block["value"] = series_speed.loc[subset_idx].to_numpy()
        speed_vals = speed_block.groupby(
            by_cols,
            sort=False,
            observed=True
        )["value"].transform(
            lambda x, win=window: _roll_mean(x, win)
        )
        speed_med[mask] = speed_vals.to_numpy(dtype=np.float64)
        
        turn_block = keys.copy()
        turn_block["value"] = series_turn.loc[subset_idx].to_numpy()
        turn_vals = turn_block.groupby(
            by_cols,
            sort=False,
            observed=True
        )["value"].transform(
            lambda x, win=window: _roll_sdev(x, win)
        )
        turn_smoothness[mask] = turn_vals.to_numpy(dtype=np.float64)
    
    forward_consistency = v_forward_med / (v_forward_sdev + 1e-6)
    invalid = ~ok_w3
    for arr in [
        v_forward_med,
        v_forward_sdev,
        forward_consistency,
        lat_ratio_med,
        speed_med,
        turn_smoothness
    ]:
        arr[invalid] = np.nan
    
    base_dt["v_forward"] = v_forward
    base_dt["v_lateral"] = v_lateral
    base_dt["a_forward"] = a_forward
    base_dt["a_lateral"] = a_lateral
    base_dt["speed_total"] = speed_total
    base_dt["turn_agent"] = turn_agent
    base_dt["curvature"] = curvature
    base_dt["turn_smoothness"] = turn_smoothness
    base_dt["forward_consistency"] = forward_consistency
    base_dt["lat_ratio_med"] = lat_ratio_med
    base_dt["speed_med"] = speed_med
    
    # Select output columns
    feats = base_dt[[
        "aug", "video_id", "mouse_id", "video_frame",
        "v_forward", "v_lateral", "a_forward", "a_lateral", "speed_total",
        "turn_agent", "curvature", "turn_smoothness",
        "forward_consistency", "lat_ratio_med", "speed_med", "ear_gap"
    ]].copy()
    feats["mouse_body_len"] = base_dt["ag_body_len"]
    
    # Merge back to original dt structure
    feature_cols = [
        "v_forward", "v_lateral", "a_forward", "a_lateral", "speed_total",
        "turn_agent", "curvature", "turn_smoothness",
        "forward_consistency", "lat_ratio_med", "speed_med", "mouse_body_len", "ear_gap"
    ]
    
    feats_ag = feats.copy()
    feats_ag.columns = ["ag_" + c if c in feature_cols else c for c in feats_ag.columns]
    feats_ag.rename(columns={"mouse_id": "agent_id"}, inplace=True)
    
    feats_tg = feats.copy()
    feats_tg.columns = ["tg_" + c if c in feature_cols else c for c in feats_tg.columns]
    feats_tg.rename(columns={"mouse_id": "target_id"}, inplace=True)
    
    id_cols = [c for c in ["aug", "video_id", "agent_id", "target_id", "video_frame", "action"] if c in dt.columns]
    out = dt[id_cols].copy()
    
    ag_cols_out = ["ag_" + c for c in feature_cols]
    out = out.merge(
        feats_ag[["aug", "video_id", "agent_id", "video_frame"] + ag_cols_out],
        on=["aug", "video_id", "agent_id", "video_frame"],
        how="left"
    )
    
    tg_cols_out = ["tg_" + c for c in feature_cols]
    out = out.merge(
        feats_tg[["aug", "video_id", "target_id", "video_frame"] + tg_cols_out],
        on=["aug", "video_id", "target_id", "video_frame"],
        how="left"
    )
    
    return out


def make_pair_dist_features(
    dt: pd.DataFrame,
    dt_meta: pd.DataFrame,
    args: dict
) -> pd.DataFrame:
    """
    Compute pairwise distance features between bodyparts.
    
    R: make_pair_dist_features (lines 561-654)
    """
    bodyparts = args["bodyparts"]
    bp_set = args.get("bp_set", "raw")
    
    dt = dt.copy()
    
    if bp_set != "raw":
        compute_center_and_length(dt, "ag")
        compute_center_and_length(dt, "tg")
        
        upper_parts = [
            "nose", "head", "ear_left", "ear_right", "neck",
            "headpiece_bottombackleft", "headpiece_bottombackright",
            "headpiece_bottomfrontleft", "headpiece_bottomfrontright",
            "headpiece_topbackleft", "headpiece_topbackright",
            "headpiece_topfrontleft", "headpiece_topfrontright"
        ]
        lower_parts = ["tail_base", "hip_left", "hip_right"]
        
        bodyparts_to_use = ["center", "lower", "upper"]
    else:
        bodyparts_to_use = bodyparts
    
    def has_xy(prefix, part):
        return f"{prefix}x_{part}" in dt.columns and f"{prefix}y_{part}" in dt.columns
    
    ag_parts = [p for p in bodyparts_to_use if has_xy("ag_", p)]
    tg_parts = [p for p in bodyparts_to_use if has_xy("tg_", p)]
    
    feat_dict = {}
    
    # Agent-agent pairs
    if len(ag_parts) >= 2:
        for i in range(len(ag_parts) - 1):
            p1 = ag_parts[i]
            x1 = dt[f"ag_x_{p1}"]
            y1 = dt[f"ag_y_{p1}"]
            for j in range(i + 1, len(ag_parts)):
                p2 = ag_parts[j]
                dx = x1 - dt[f"ag_x_{p2}"]
                dy = y1 - dt[f"ag_y_{p2}"]
                feat_dict[f"m1m1+{p1}+{p2}"] = dx**2 + dy**2
    
    # Target-target pairs
    if len(tg_parts) >= 2:
        for i in range(len(tg_parts) - 1):
            p1 = tg_parts[i]
            x1 = dt[f"tg_x_{p1}"]
            y1 = dt[f"tg_y_{p1}"]
            for j in range(i + 1, len(tg_parts)):
                p2 = tg_parts[j]
                dx = x1 - dt[f"tg_x_{p2}"]
                dy = y1 - dt[f"tg_y_{p2}"]
                feat_dict[f"m2m2+{p1}+{p2}"] = dx**2 + dy**2
    
    # Agent-target pairs
    if len(ag_parts) > 0 and len(tg_parts) > 0:
        for p1 in ag_parts:
            x1 = dt[f"ag_x_{p1}"]
            y1 = dt[f"ag_y_{p1}"]
            for p2 in tg_parts:
                dx = x1 - dt[f"tg_x_{p2}"]
                dy = y1 - dt[f"tg_y_{p2}"]
                feat_dict[f"m1m2+{p1}+{p2}"] = dx**2 + dy**2
    
    if not feat_dict:
        return pd.DataFrame()
    
    return pd.DataFrame(feat_dict)


def add_lag_window_stats(
    dt: pd.DataFrame,
    value_col: str,
    by_cols: list,
    num_windows: int = 3
) -> None:
    """
    Add lagged window statistics for temporal features.
    Modifies dt in-place.
    """
    col_names = [f"{value_col}_lagw{w}_mean" for w in range(1, num_windows + 1)]
    for col in col_names:
        if col not in dt.columns:
            dt[col] = np.nan
    
    kf_vals = dt["kf"].to_numpy(dtype=np.int64)
    unique_kf = np.unique(kf_vals)
    
    for kf in unique_kf:
        mask = kf_vals == kf
        subset = dt.loc[mask, by_cols + [value_col, "ok1", "ok2", "ok3"]].copy()
        if subset.empty:
            continue
        
        grouped = subset.groupby(by_cols, sort=False, observed=True)
        base_mean = grouped[value_col].transform(
            lambda x, win=kf: x.rolling(window=win, min_periods=1).mean()
        )
        subset[f"{value_col}_lagw1_mean"] = base_mean
        
        grouped_base = subset.groupby(by_cols, sort=False, observed=True)
        for w in range(2, num_windows + 1):
            col_name = f"{value_col}_lagw{w}_mean"
            subset[col_name] = grouped_base[f"{value_col}_lagw1_mean"].shift((w - 1) * kf)
        
        subset[f"{value_col}_lagw1_mean"] = np.where(
            subset["ok1"],
            subset[f"{value_col}_lagw1_mean"],
            np.nan
        )
        if num_windows >= 2:
            col_name = f"{value_col}_lagw2_mean"
            if col_name in subset.columns:
                subset[col_name] = np.where(subset["ok2"], subset[col_name], np.nan)
        if num_windows >= 3:
            col_name = f"{value_col}_lagw3_mean"
            if col_name in subset.columns:
                subset[col_name] = np.where(subset["ok3"], subset[col_name], np.nan)
        
        for col in col_names:
            if col in subset.columns:
                dt.loc[mask, col] = subset[col].to_numpy()


def add_temporal_feats(
    dt: pd.DataFrame,
    dt_meta: pd.DataFrame,
    args: dict
) -> None:
    """
    Add temporal (lagged window) features.
    Modifies dt in-place.
    """
    lag_sec = args["lag_sec"]
    window_cols = [c for c in args.get("window_cols", []) if c in dt.columns]
    by_cols = ["aug", "video_id", "agent_id", "target_id"]
    key_cols = by_cols + ["video_frame"]
    if "action" in dt.columns:
        key_cols.append("action")
    num_windows = args.get("num_windows", 3)
    
    if not window_cols:
        return
    
    work_cols = list(dict.fromkeys(key_cols + window_cols))
    temp = dt[work_cols].copy()
    temp = add_lag_structure(temp, dt_meta, lag_sec, by_cols)
    
    for col in window_cols:
        add_lag_window_stats(temp, col, by_cols, num_windows=num_windows)
    
    lag_cols = [
        f"{col}_lagw{win}_mean"
        for col in window_cols
        for win in range(1, num_windows + 1)
    ]
    lag_cols = [c for c in lag_cols if c in temp.columns]
    if not lag_cols:
        return
    
    merge_src = temp[key_cols + lag_cols].copy()
    augmented = dt[key_cols].merge(merge_src, on=key_cols, how="left", sort=False)
    
    for col in lag_cols:
        dt[col] = augmented[col].to_numpy()
