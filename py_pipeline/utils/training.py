"""
Training utilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import math
import numpy as np
import pandas as pd

def make_folds(
    pos_all: pd.DataFrame,
    vids_all: List[str],
    k_folds: int,
    seed: int,
    num_models: Optional[int] = None,
) -> List[Dict[str, List[str]]]:
    """
    Split vids_all into validation folds with at least one positive video
    per fold. If num_models is set, return that many val-sets
    generated from repeated k-fold partitions with different seeds.
    """
    pos_vids = set(pos_all["video_id"].astype(str).unique().tolist())
    vids_all = sorted([str(v) for v in vids_all])

    # clamp k_folds by number of positive videos and total videos
    k_folds = max(1, min(k_folds, len(pos_vids), len(vids_all)))
    
    def build_splits(rng: np.random.RandomState) -> List[List[str]]:
        # Create fold indices: rep(1:k_folds, length.out=n_vids)
        folds_idx = np.tile(np.arange(k_folds), (len(vids_all) // k_folds) + 1)[:len(vids_all)]
        
        # Shuffle videos and fold indices independently
        perm_vids = rng.permutation(len(vids_all))
        perm_idx = rng.permutation(len(vids_all))
        vids = np.array(vids_all)[perm_vids]
        folds_idx = folds_idx[perm_idx]
        
        # Split by indices
        folds = [[] for _ in range(k_folds)]
        for vid, idx in zip(vids, folds_idx):
            folds[idx].append(vid)

        # ensure each fold has at least one positive video
        for i in range(k_folds):
            if any(v in pos_vids for v in folds[i]):
                continue

            # Find all donors with positive videos
            donors_with_pos = [
                j for j in range(k_folds)
                if sum(v in pos_vids for v in folds[j]) > 0
            ]
            if not donors_with_pos:
                continue
            
            # Randomly select donor among those with most positive videos
            max_pos_count = max(
                sum(v in pos_vids for v in folds[j]) for j in donors_with_pos
            )
            best_donors = [
                j for j in donors_with_pos
                if sum(v in pos_vids for v in folds[j]) == max_pos_count
            ]
            donor_idx = best_donors[rng.randint(len(best_donors))]
            
            candidates = [v for v in folds[donor_idx] if v in pos_vids]
            # Randomly select candidate to move
            move_vid = candidates[rng.randint(len(candidates))]
            folds[i].append(move_vid)
            folds[donor_idx].remove(move_vid)

        # build output format: train/val pairs
        splits = []
        for i in range(k_folds):
            val = folds[i]
            train = [v for v in vids_all if v not in val]
            splits.append({"train": train, "val": val})
        return splits

    rng_master = np.random.RandomState(seed)

    if num_models is None:
        # classical k-fold: return list of val_vids by folds
        folds = build_splits(rng_master)
        return folds

    # bagging from multiple k-fold partitions
    n_models = int(num_models)
    n_reps = math.ceil(n_models / k_folds)

    all_splits = []
    for _ in range(n_reps):
        rng_local = np.random.RandomState(rng_master.randint(0, 2**31 - 1))
        all_splits.extend(build_splits(rng_local))

    return all_splits[:n_models]

def make_fold_sets(
    pos_all: pd.DataFrame,
    neg_all: pd.DataFrame,
    train_vids: list[str],
    val_vids: list[str],
    sample_neg_percent: Optional[float],
    sample_unl_percent: Optional[float],
    whitelist: pd.DataFrame,
    seed: int,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    a = pos_all["action"].iloc[0]

    pos_vids = pos_all["video_id"].values
    neg_vids = neg_all["video_id"].values
    neg_actions = neg_all["action"].values
    neg_agents = neg_all["agent_id"].values
    neg_targets = neg_all["target_id"].values

    train_vids_set = set(train_vids)
    val_vids_set = set(val_vids)

    pos_train_mask = np.fromiter((v in train_vids_set for v in pos_vids), dtype=bool)
    pos_val_mask   = np.fromiter((v in val_vids_set   for v in pos_vids), dtype=bool)
    neg_train_mask = np.fromiter((v in train_vids_set for v in neg_vids), dtype=bool)
    neg_val_mask   = np.fromiter((v in val_vids_set   for v in neg_vids), dtype=bool)

    idx_pos_train = pos_all.index[pos_train_mask].to_numpy()
    idx_pos_val = pos_all.index[pos_val_mask].to_numpy()
    idx_neg_train = neg_all.index[neg_train_mask].to_numpy()
    idx_neg_val = neg_all.index[neg_val_mask].to_numpy()

    wl_a = whitelist[whitelist["action"] == a]
    combos_with_a = set(
        zip(
            wl_a["video_id"].to_numpy(),
            wl_a["agent_id"].to_numpy(),
            wl_a["target_id"].to_numpy(),
        )
    )

    neg_train_vids = neg_vids[neg_train_mask]
    neg_train_actions = neg_actions[neg_train_mask]
    neg_train_agents = neg_agents[neg_train_mask]
    neg_train_targets = neg_targets[neg_train_mask]

    mask_pair_has_a = np.fromiter(
        (
            (v, ag, tg) in combos_with_a
            for v, ag, tg in zip(
                neg_train_vids,
                neg_train_agents,
                neg_train_targets,
            )
        ),
        dtype=bool,
    )

    mask_real = (
        (neg_train_actions != "Nothing")
        | ((neg_train_actions == "Nothing") & mask_pair_has_a)
    )
    mask_maybe = (neg_train_actions == "Nothing") & (~mask_pair_has_a)

    idx_real = idx_neg_train[mask_real]
    idx_maybe = idx_neg_train[mask_maybe]

    unlabeled_mask = neg_train_actions == "Nothing"
    n_unlabeled_total = int(unlabeled_mask.sum())

    mask_real_unlabeled = unlabeled_mask & mask_real
    n_real_unlabeled = int(mask_real_unlabeled.sum())
    n_maybe_unlabeled = int(idx_maybe.size)

    sampled_real = sample_by_video(neg_all.loc[idx_real], sample_neg_percent, seed).index.to_numpy()
    sampled_maybe = sample_by_video(neg_all.loc[idx_maybe], sample_unl_percent, seed).index.to_numpy()

    idx_neg_train_final = np.concatenate([sampled_real, sampled_maybe])

    if verbose:
        n_sampled_real = int(sampled_real.size)
        n_sampled_maybe = int(sampled_maybe.size)

        total_neg_train = n_sampled_real + n_sampled_maybe

        if n_unlabeled_total > 0:
            frac_real_unlabeled = n_real_unlabeled / n_unlabeled_total
            frac_maybe_unlabeled = n_maybe_unlabeled / n_unlabeled_total
        else:
            frac_real_unlabeled = 0.0
            frac_maybe_unlabeled = 0.0

        if total_neg_train > 0:
            frac_tr_real = n_sampled_real / total_neg_train
            frac_tr_maybe = n_sampled_maybe / total_neg_train
        else:
            frac_tr_real = 0.0
            frac_tr_maybe = 0.0

        print(
            f"[neg stats]: Unlabeled_total={n_unlabeled_total}, "
            f"real_unlabeled={n_real_unlabeled} "
            f"({frac_real_unlabeled*100:.2f}%), "
            f"maybe_unlabeled={n_maybe_unlabeled} "
            f"({frac_maybe_unlabeled*100:.2f}%)"
        )
        print(
            f"[neg stats]: train_neg={total_neg_train}, "
            f"from_real={n_sampled_real} "
            f"({frac_tr_real*100:.2f}%), "
            f"from_maybe={n_sampled_maybe} "
            f"({frac_tr_maybe*100:.2f}%)"
        )

    train_idx = np.concatenate([idx_pos_train, idx_neg_train_final])
    valid_idx = np.concatenate([idx_pos_val, idx_neg_val])

    return {"train_idx": train_idx, "valid_idx": valid_idx}

def print_data_counts(
    train_labels: np.ndarray,
    valid_labels: np.ndarray,
) -> None:
    """
    Prints information about the number of data in the train and valid.
    """
    n_pos_tr = int(train_labels.sum())
    n_neg_tr = len(train_labels) - n_pos_tr
    n_pos_va = int(valid_labels.sum())
    n_neg_va = len(valid_labels) - n_pos_va
    
    ratio_tr = n_neg_tr / n_pos_tr if n_pos_tr > 0 else float('inf')
    ratio_va = n_neg_va / n_pos_va if n_pos_va > 0 else float('inf')
    
    ratio_tr_str = f"{ratio_tr:.1f}" if ratio_tr != float('inf') else "inf"
    ratio_va_str = f"{ratio_va:.1f}" if ratio_va != float('inf') else "inf"
    
    print(
        f"train neg/pos={n_neg_tr}/{n_pos_tr} (ratio={ratio_tr_str}) | "
        f"valid neg/pos={n_neg_va}/{n_pos_va} (ratio={ratio_va_str}) | ",
        end="",
        flush=True
    )

def sample_by_video(
    dt: pd.DataFrame,
    sample_percent: Optional[float],
    seed: int
) -> pd.DataFrame:
    if sample_percent is None or sample_percent >= 100:
        return dt.copy()

    dt = dt.sample(frac=1.0, random_state=seed)

    sizes = dt["video_id"].value_counts()
    n_take = (sizes * sample_percent / 100.0).astype(int)

    n_take = n_take.clip(upper=sizes)
    n_take.name = "n_take"

    dt = dt.merge(n_take, left_on="video_id", right_index=True, how="left")

    dt["_row_idx"] = dt.groupby("video_id").cumcount()

    sampled = dt[dt["_row_idx"] < dt["n_take"]].drop(
        columns=["_row_idx", "n_take"]
    )

    return sampled

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, ",".join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)


def save_run_log(
    lab_name: str,
    params: Dict[str, Any],
    n_features: int,
    scores: Dict[str, float],
    path: Path,
) -> None:
    """
    Append one training run summary to CSV log at `path`.
    Creates a new file if it does not exist.
    """
    scores_rounded = {k: round(v, 4) for k, v in scores.items()}

    run_summary = {
        "lab_name": lab_name,
        "n_features": n_features,
        **scores_rounded,
        **params,
    }

    run_summary_flat = flatten_dict(run_summary)
    run_df = pd.DataFrame([run_summary_flat])

    if path.exists():
        prev = pd.read_csv(path)
        all_runs = pd.concat([prev, run_df], ignore_index=True)
    else:
        all_runs = run_df

    path.parent.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(path, index=False)