"""
Data loading functions and preparation functions.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import yaml
import re
import math

from py_pipeline.utils.thresholds import fit_z, fit_z_per_lab
from py_pipeline.utils.postprocessing import (
    filter_preds_by_thr,
    get_intervals,
    postprocess_preds,
)

def load_meta_df(file_path: Path) -> pd.DataFrame:
    """
    Load metadata from train.csv/test.csv.
    """
    meta_df = pd.read_csv(file_path)

    meta_df["lab_id"] = meta_df["lab_id"].astype(str)
    meta_df["video_id"] = meta_df["video_id"].astype(str)
    meta_df["frames_per_second"] = meta_df["frames_per_second"].astype("float32")
    meta_df["pix_per_cm_approx"] = meta_df["pix_per_cm_approx"].astype("float32")
    meta_df["body_parts_tracked"] = meta_df["body_parts_tracked"].astype(str)
    meta_df["behaviors_labeled"] = (
            meta_df["behaviors_labeled"]
            .fillna("[]")
            .astype(str)
        )

    return meta_df

def load_train_ann(
    data_dir: Path,
    train_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Load training annotations from parquet files.
    """
    ann_dir = Path(data_dir) / "train_annotation"
    ann_files = list(ann_dir.rglob("*.parquet"))
    if not ann_files:
        raise FileNotFoundError(f"No parquet files found in {ann_dir}")
    
    frames = []
    for file_path in ann_files:
        df = pd.read_parquet(file_path)
        df["video_id"] = file_path.stem
        frames.append(df)
    
    train_ann = pd.concat(frames, ignore_index=True, sort=False)

    train_ann["video_id"] = train_ann["video_id"].astype(str)
    train_ann = train_ann.merge(
        train_meta[["video_id", "lab_id"]],
        on="video_id",
        how="left",
    )
    train_ann["agent_id"] = train_ann["agent_id"].astype("int32")
    train_ann["target_id"] = train_ann["target_id"].astype("int32")
    train_ann["start_frame"] = train_ann["start_frame"].astype("int32")
    train_ann["stop_frame"] = train_ann["stop_frame"].astype("int32")

    train_ann = train_ann[train_ann["action"] != "ejaculate"].copy()
    
    return train_ann


def compute_action_duration(train_ann: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-lab, per-action median duration in frames.
    """
    act_duration = train_ann.copy()
    act_duration["roll_frames"] = act_duration["stop_frame"] - act_duration["start_frame"]

    act_duration = (
        act_duration.groupby(["lab_id", "action"], as_index=False)["roll_frames"]
        .median()
    )
    act_duration["roll_frames"] = (
        act_duration["roll_frames"].round().astype("int32")
    )
    return act_duration

def get_whitelist(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build global whitelist of (lab_id, video_id, agent_id, target_id, action)
    from the behaviors_labeled column in train/test meta df.
    """
    whitelist = []    
    for _, row in meta_df.iterrows():
        lab_id   = row['lab_id']
        video_id = row['video_id']
        raw      = row['behaviors_labeled']
        
        if not isinstance(raw, str) or not raw.strip():
            continue
        
        triads = json.loads(raw)
        triads = sorted({t.replace("'", "") for t in triads})
        
        for triad in triads:
            parts = [p.strip().lower() for p in triad.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Invalid triad format '{triad}' video_id={video_id}")

            agent_str, target_str, action = parts

            if target_str == "self":
                target_str = agent_str

            agent_id  = int(agent_str.replace("mouse", ""))
            target_id = int(target_str.replace("mouse", ""))

            whitelist.append({
                "lab_id":    str(lab_id),
                "video_id":  str(video_id),
                "agent_id":  np.int32(agent_id),
                "target_id": np.int32(target_id),
                "action":    str(action),
            })

    return pd.DataFrame(whitelist)


def get_bodyparts(train_meta: pd.DataFrame, lab_name: str) -> list[str]:
    """
    Return bodyparts that are tracked in every video for the given lab.
    """
    lab_meta = train_meta[train_meta["lab_id"] == lab_name]
    
    shared_parts = None
    for bp_str in lab_meta["body_parts_tracked"]:
        try:
            parts = json.loads(bp_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid body_parts_tracked JSON for lab {lab_name}: {bp_str}"
            ) from exc

        current = set(parts)

        if shared_parts is None:
            shared_parts = current
        else:
            shared_parts &= current

    return sorted(shared_parts)

def get_tracking(
    tracks_dir: Path,
    lab_name: str,
    bodyparts: list[str],
    frames: Optional[pd.DataFrame] = None,
    videos: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load tracking data for a single lab and a given set of bodyparts.
    Optionally restrict to specific frames or videos.
    When frames df is given, the set of videos is taken from frames["video_id"].
    """
    tracks_dir = Path(tracks_dir)
    lab_dir    = tracks_dir / lab_name
    if not lab_dir.is_dir():
        raise ValueError(f"Lab directory not found: {lab_dir}")
    
    parquet_files = list(lab_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {lab_dir}")
    
    file_map = pd.DataFrame(
        {
            "path": [str(path).replace("\\", "/") for path in parquet_files],
            "video_id": [path.stem for path in parquet_files],
        }
    )
    
    if videos is not None:
        videos_set = {str(v) for v in videos}
        file_map = file_map[file_map["video_id"].isin(videos_set)]
    
    if file_map.empty:
        raise ValueError("No matching parquet files for videos")
    
    tracks_list: list[pd.DataFrame] = []
    for _, row in file_map.iterrows():
        df = pd.read_parquet(row["path"])
        df["video_id"] = str(row["video_id"])
        tracks_list.append(df)
    
    tracks = pd.concat(tracks_list, ignore_index=True)
    tracks = tracks[tracks["bodypart"].isin(bodyparts)]
    
    if frames is not None and not frames.empty:
        frames_local = frames[["video_id", "mouse_id", "video_frame"]].copy()
        frames_local["video_id"] = frames_local["video_id"].astype(str)
        tracks = tracks.merge(
            frames_local,
            on  = ["video_id", "mouse_id", "video_frame"],
            how = "inner",
        )
    
    if tracks.empty:
        raise ValueError("No rows for requested lab/bodyparts/frames")
    
    tracks["video_frame"] = tracks["video_frame"].astype("int32")
    tracks["mouse_id"] = tracks["mouse_id"].astype("int32")
    tracks["x"] = tracks["x"].astype("float32")
    tracks["y"] = tracks["y"].astype("float32")

    return tracks

def merge_labels_tracks(
    labels: pd.DataFrame,
    tracks: pd.DataFrame,
    whitelist: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge labels with tracking data:
    Creates dataset with all frame-agent-target combinations from
    whitelist, adds coordinates for agent and target.
    """
    # Pivot tracks to wide format (one row per frame-mouse)
    pivot = tracks.pivot_table(
        index=["video_id", "mouse_id", "video_frame"],
        columns="bodypart",
        values=[c for c in ["x", "y"] if c in tracks.columns],
        aggfunc="first"
    )
    pivot.columns = [
        f"{coord}_{bp}" for coord, bp in pivot.columns.to_flat_index()
    ]
    tracks_wide = pivot.reset_index()
    
    # Get all frames and pairs
    frames = tracks_wide[["video_id", "video_frame"]].drop_duplicates()
    pairs = whitelist[["video_id", "agent_id", "target_id"]].drop_duplicates()
    
    # Create all combinations (all frames only for whitelisted pairs)
    all_combos = pairs.merge(frames, on="video_id", how="left")
    
    # Add labels
    if len(labels) > 0:
        all_combos = all_combos.merge(
            labels,
            on=["video_id", "video_frame", "agent_id", "target_id"],
            how="left"
        )
    else:
        all_combos["action"] = None
    
    all_combos["action"] = all_combos["action"].fillna("Nothing")
    
    coord_cols = [
        c for c in tracks_wide.columns
        if c not in ["video_id", "mouse_id", "video_frame"]
    ]
    
    # Add agent coordinates
    agent_rename = {"mouse_id": "agent_id"}
    agent_rename.update({c: f"ag_{c}" for c in coord_cols})
    tracks_wide_ag = tracks_wide.rename(columns=agent_rename)

    out = all_combos.merge(
        tracks_wide_ag,
        on=["video_id", "video_frame", "agent_id"],
        how="left"
    )

    # Add target coordinates
    target_rename = {"mouse_id": "target_id"}
    target_rename.update({c: f"tg_{c}" for c in coord_cols})
    tracks_wide_tg = tracks_wide.rename(columns=target_rename)

    out = out.merge(
        tracks_wide_tg,
        on=["video_id", "video_frame", "target_id"],
        how="left"
    )

    # Reorder columns
    key_cols = ["video_id", "agent_id", "target_id", "action", "video_frame"]
    other_cols = [c for c in out.columns if c not in key_cols]
    out = out[key_cols + other_cols]
    
    return out

def expand_ann_to_frames(train_ann: pd.DataFrame) -> pd.DataFrame:
    """
    Expand annotated intervals into per-frame labels.
    """
    # intervals lengths [start_frame, stop_frame)
    start_f = train_ann["start_frame"].to_numpy(dtype="int32")
    stop_f  = train_ann["stop_frame"].to_numpy(dtype="int32")
    
    lengths = (stop_f - start_f)

    frames = np.concatenate([
        np.arange(s, e, dtype="int32")
        for s, e in zip(start_f, stop_f)
    ])

    expanded = pd.DataFrame({
        "video_id": np.repeat(train_ann["video_id"].to_numpy(), lengths),
        "agent_id": np.repeat(train_ann["agent_id"].to_numpy(), lengths),
        "target_id": np.repeat(train_ann["target_id"].to_numpy(), lengths),
        "video_frame": frames,
        "action": np.repeat(train_ann["action"].to_numpy(), lengths),
    })
    return expanded

def build_dataset(
    labels_dt: pd.DataFrame,
    dt_meta: pd.DataFrame,
    whitelist: pd.DataFrame,
    tracks_dir: Path,
    lab_name: str,
    feature_params: Dict[str, Any],
    frames: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build dataset with features from labels, tracking, and metadata.
    """
    from py_pipeline.utils.features import (
        compute_invariants,
        compute_single_mouse,
        make_pair_dist_features,
        normalize_xy,
        add_temporal_feats,
    )
    
    bodyparts = feature_params.get("bodyparts", [])
    fill_na = feature_params.get("fill_na", "none")
    use_augment = bool(feature_params.get("augment", False))
    
    # Get video IDs
    if labels_dt is not None and not labels_dt.empty:
        video_ids = labels_dt["video_id"].unique().tolist()
    else:
        video_ids = whitelist["video_id"].unique().tolist()
    
    # Get tracking data
    tracks = get_tracking(
        tracks_dir=tracks_dir,
        lab_name=lab_name,
        bodyparts=bodyparts,
        frames=frames,
        videos=video_ids,
    )
    
    # Normalize to cm if requested
    if feature_params.get("norm2cm", True):
        tracks = normalize_xy(tracks, dt_meta)
    
    # Get annotations by frame
    if labels_dt is not None and not labels_dt.empty:
        labels_expanded = expand_ann_to_frames(labels_dt)
    else:
        labels_expanded = pd.DataFrame(
            columns=["video_id", "agent_id", "target_id", "video_frame", "action"]
        )
    
    # Build dt with coords for each ag/tg pair
    dt = merge_labels_tracks(labels_expanded, tracks, whitelist)
    
    # Fill NA based on method
    if fill_na == "none":
        pass
    elif fill_na == "zeros":
        dt = dt.fillna(0)
    else:
        raise ValueError(f"Unknown fill_na method: {fill_na}")

    if use_augment:
        dt = add_augmented_blocks(dt, dt_meta, feature_params, verbose=verbose)
    else:
        dt["aug"] = False

    # Build features
    key_cols = ["video_id", "agent_id", "target_id", "video_frame", "action", "aug"]
    coord_cols = [c for c in dt.columns if re.search(r"(^|_)x($|_)|(^|_)y($|_)", c)]

    # single mouse features
    if feature_params.get("single_mouse", False):
        args = {
            "lag_sec": feature_params.get("lag_sec", 0.1),
            "norm_by": feature_params.get("norm_by", "med_body_len"),
        }
        dt_mouse = compute_single_mouse(dt, dt_meta, args)
        new_cols = [c for c in dt_mouse.columns if c not in key_cols]
        dt = dt.merge(
            dt_mouse[key_cols + new_cols],
            on=key_cols,
            how="left",
        )
        if verbose:
            print(f"\nADDED: {len(new_cols)} features: {', '.join(new_cols)}")

    # invariants
    if feature_params.get("invariants", False):
        args = {
            "lag_sec": feature_params.get("lag_sec", 0.1),
            "norm_by": feature_params.get("norm_by", "med_body_len"),
        }
        dt_inv = compute_invariants(dt, dt_meta, args)
        new_cols = [c for c in dt_inv.columns if c not in key_cols]
        dt = dt.merge(
            dt_inv[key_cols + new_cols],
            on=key_cols,
            how="left",
        )
        if verbose:
            print(f"\nADDED: {len(new_cols)} features: {', '.join(new_cols)}")

    # pair distances
    if feature_params.get("pair_dist", False):
        args = {
            "bodyparts": bodyparts,
            "bp_set": feature_params.get("bp_set", "raw"),
        }
        dt_dist = make_pair_dist_features(dt, dt_meta, args)
        if not dt_dist.empty:
            dt = pd.concat(
                [dt.reset_index(drop=True), dt_dist.reset_index(drop=True)],
                axis=1,
            )
            if verbose:
                print(
                f"\nADDED: {len(dt_dist.columns)} features: "
                f"{', '.join(dt_dist.columns)}"
            )

    # temporal features
    if feature_params.get("add_temporal", False):
        before_cols = set(dt.columns)
        args = {
            "lag_sec": feature_params.get("lag_sec", 0.1),
            "window_cols": feature_params.get("window_cols", []),
        }
        add_temporal_feats(dt, dt_meta, args)
        new_cols = sorted(c for c in dt.columns if c not in before_cols)
        if verbose:
            print(
            f"\nADDED: {len(new_cols)} temporal features: {', '.join(new_cols)}"
        )

    # remove raw coords
    if not feature_params.get("coords", False):
        dt = dt.drop(columns=coord_cols, errors="ignore")

    if use_augment:
        is_aug = dt["aug"]
        is_labeled = dt["action"] != "Nothing"
        dt = dt[(~is_aug) | is_labeled].reset_index(drop=True)

    float_cols = dt.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        dt[float_cols] = dt[float_cols].astype("float32")

    return dt

def add_augmented_blocks(
    dt: pd.DataFrame,
    dt_meta: pd.DataFrame,
    feature_params: Dict[str, Any],
    verbose: bool = False,
) -> pd.DataFrame:
    dt = dt.copy()
    dt["aug"] = False

    lag_sec = float(feature_params.get("lag_sec", 0.1))
    
    fps_map = (
        dt_meta[["video_id", "frames_per_second"]]
        .drop_duplicates()
        .set_index("video_id")["frames_per_second"]
    )

    augment_parts: List[pd.DataFrame] = []

    for (video_id, agent_id, target_id), dt_pair in dt.groupby(
        ["video_id", "agent_id", "target_id"],
        sort=False,
    ):
        actions_pair = dt_pair["action"].to_numpy()
        labeled_local = actions_pair != "Nothing"
        
        fps_val = float(fps_map.loc[video_id])
        frame_radius = max(1, int(round(lag_sec * fps_val)))

        kernel = np.ones(2 * frame_radius + 1, dtype=np.int32)
        dilated = np.convolve(
            labeled_local.astype(np.int32),
            kernel,
            mode="same",
        )
        block_mask = dilated > 0
        
        dt_block = dt_pair.loc[block_mask].copy()
        dt_block = augment_pair_coords(dt_block, feature_params)
        dt_block["aug"] = True
        augment_parts.append(dt_block)

    dt_aug = pd.concat(augment_parts, ignore_index=True)
    if verbose:
        print(f"\nAUGMENT: {len(dt_aug)} rows, " f"lag_sec={lag_sec}")

    dt_full = pd.concat([dt, dt_aug], ignore_index=True)
    return dt_full

def augment_pair_coords(
    dt_block: pd.DataFrame,
    feature_params: Dict[str, Any],
) -> pd.DataFrame:
    max_deg = 35.0
    flip_prob = 0.8
    scale_min = 0.9
    scale_max = 1.1
    noise_std = 0.02

    dt_block = dt_block.copy()

    rng = np.random.default_rng()
    angle_deg = rng.uniform(-max_deg, max_deg)
    angle_rad = angle_deg * math.pi / 180.0

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    do_flip = rng.random() < flip_prob
    flip_sign = -1.0 if do_flip else 1.0

    scale = rng.uniform(scale_min, scale_max)

    coord_pairs: List[Tuple[str, str]] = []
    for col_x in dt_block.columns:
        if col_x.startswith("ag_x_"):
            col_y = "ag_y_" + col_x[len("ag_x_"):]
        elif col_x.startswith("tg_x_"):
            col_y = "tg_y_" + col_x[len("tg_x_"):]
        else:
            continue
        if col_y in dt_block.columns:
            coord_pairs.append((col_x, col_y))

    for col_x, col_y in coord_pairs:
        x_vals = dt_block[col_x].to_numpy(dtype=np.float32, copy=False)
        y_vals = dt_block[col_y].to_numpy(dtype=np.float32, copy=False)

        x_new = cos_a * x_vals - sin_a * y_vals
        y_new = sin_a * x_vals + cos_a * y_vals

        x_new = flip_sign * x_new

        x_new = scale * x_new
        y_new = scale * y_new

        if noise_std > 0.0:
            x_new = x_new + rng.normal(0.0, noise_std, size=x_new.shape).astype(np.float32)
            y_new = y_new + rng.normal(0.0, noise_std, size=y_new.shape).astype(np.float32)

        dt_block[col_x] = x_new
        dt_block[col_y] = y_new

    return dt_block

def load_lab_config(config_path: Path) -> Dict[str, Any]:
    """
    Load lab-specific configuration from YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config

def load_params_config(config_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load parameters configuration from YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("params_config must be a mapping of labs.")
    return data

def load_feats_config(config_path: Path) -> Dict[str, Any]:
    """
    Load features configuration from YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"feats_config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("feats_config must be a mapping of labs.")
    return data

def get_lab_window_specs(
    lab_name: str,
    feats_config: Dict[str, Any],
    num_windows: int = 3,
) -> Dict[str, Any]:
    """
    Extract per-lab window columns and derived feature names.
    """
    if lab_name not in feats_config:
        raise ValueError(f"Lab {lab_name} missing in feats_config.")
    lab_entry = feats_config[lab_name] or {}
    if not isinstance(lab_entry, dict):
        raise ValueError(f"Invalid config for lab {lab_name}.")
    
    window_cols: set[str] = set()
    tmp_fnames_by_action: Dict[str, List[str]] = {}
    
    for action_name, cfg in lab_entry.items():
        if cfg is None:
            tmp_fnames_by_action[action_name] = []
            continue
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Invalid action config for {lab_name}:{action_name}"
            )
        action_cols = cfg.get("window_cols", [])
        if not action_cols:
            tmp_fnames_by_action[action_name] = []
            continue
        window_cols.update(action_cols)
        feat_names = {
            f"{col}_lagw{num}_mean"
            for col in action_cols
            for num in range(1, num_windows + 1)
        }
        tmp_fnames_by_action[action_name] = sorted(feat_names)
    
    return {
        "window_cols": sorted(window_cols),
        "tmp_fnames_by_action": tmp_fnames_by_action,
    }

def load_pseudo_df(pseudo_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if pseudo_path is None:
        print("[INFO] pseudo_path not provided.")
        return None
    
    pseudo_df = pd.read_csv(
        pseudo_path,
        dtype={
            "video_id": "string",
            "agent_id": "int32",
            "target_id": "int32",
            "video_frame": "int32",
            "action": "string",
        },
    )

    if pseudo_df.empty:
        print(f"[INFO] pseudo_df is empty at {pseudo_path}")
    else:
        print(f"[INFO] pseudo_df loaded from {pseudo_path}")

    return pseudo_df

def prepare_submit(
    preds_dt: pd.DataFrame,
    whitelist: pd.DataFrame,
    roll_frames: pd.DataFrame,
    solution: Optional[pd.DataFrame] = None,
    z_grid: Optional[np.ndarray] = None,
    z_by_lab: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Prepare submission intervals from frame-level predictions:
      1. Filter predictions by whitelist (plus 'unlabeled'/'Nothing').
      2. If z_by_lab is None and z_grid is given, fit thresholds.
        - per (lab_id, action) in binary setup
        - one per lab_id when 'Nothing' is present (multiclass).
      3. Apply temporal smoothing (postprocess_preds).
      4. Apply threshold filtering, then argmax (filter_preds_by_thr).
      5. Convert remaining positives to intervals (get_intervals).

    Returns: {"submit": intervals, "z_by_lab": thresholds_or_None}.
    """
    # filter predictions to whitelist
    wl_keys = whitelist[
        ["video_id", "agent_id", "target_id", "action"]
    ].drop_duplicates()

    preds_dt = preds_dt.merge(
        wl_keys.assign(in_wl=True),
        on=["video_id", "agent_id", "target_id", "action"],
        how="left",
    )

    preds_dt = preds_dt[
        (preds_dt["in_wl"] == True)
        | (preds_dt["action"].isin(["unlabeled", "Nothing"]))
    ].copy()

    preds_dt = preds_dt.drop(columns=["in_wl"], errors="ignore")

    # fit thresholds
    if z_by_lab is None and z_grid is not None:
        has_nothing = "Nothing" in preds_dt["action"].unique()

        if has_nothing:
            print("Fitting thresholds per lab_id")
            z_by_lab = fit_z_per_lab(
                preds_dt=preds_dt,
                solution=solution,
                roll_frames=roll_frames,
                z_grid=z_grid,
            )
        else:
            print("Fitting thresholds per (lab_id, action) pair")
            z_by_lab = fit_z(
                preds_dt=preds_dt,
                solution=solution,
                roll_frames=roll_frames,
                z_grid=z_grid,
            )
        z_by_lab = z_by_lab.drop(columns=["score"], errors="ignore")

    # smooth and filter predictions based on thresholds, then argmax
    preds_dt = postprocess_preds(preds_dt, roll_frames=roll_frames)
    preds_dt = filter_preds_by_thr(preds_dt, logit_thr=z_by_lab)

    # get intervals
    submit = get_intervals(preds_dt)

    return {"submit": submit, "z_by_lab": z_by_lab}


def build_solution(
    labels: pd.DataFrame,
    whitelist: pd.DataFrame
) -> pd.DataFrame:
    """
    Build solution DataFrame from labels df and whitelist:
    For each video_id constructs a JSON-encoded list of labeled
    (agent,target,action) triads in the column `behaviors_labeled`
    and converts agent_id/target_id to 'mouseX' format.
    """
    solution = labels.copy()

    wl = whitelist[["video_id", "agent_id", "target_id", "action"]].copy()
    wl["behaviors_labeled"] = (
        "mouse" + wl["agent_id"].astype(str) + "," +
        "mouse" + wl["target_id"].astype(str) + "," +
        wl["action"].astype(str)
    )

    wl_grouped = (
        wl.groupby("video_id", as_index=False)["behaviors_labeled"]
        .agg(lambda x: json.dumps(sorted(set(x))))
    )

    solution = solution.merge(wl_grouped, on="video_id", how="left")

    # Ensure valid JSON string even if video_id is absent in whitelist
    solution["behaviors_labeled"] = solution["behaviors_labeled"].fillna("[]")

    solution["agent_id"] = "mouse" + solution["agent_id"].astype(str)
    solution["target_id"] = "mouse" + solution["target_id"].astype(str)

    cols = [
        "lab_id",
        "video_id",
        "agent_id",
        "target_id",
        "action",
        "start_frame",
        "stop_frame",
        "behaviors_labeled",
    ]
    return solution[cols].copy()