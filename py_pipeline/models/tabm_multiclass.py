"""
TabM model for the MABe Challenge.

This module ports the training logic from `exp_code/MABe_train_tabm.ipynb` into
the unified Python pipeline. For each laboratory, it trains a single TabM
network using GroupKFold splits over `video_id` and produces out-of-fold (OOF)
predictions compatible with `py_pipeline.utils.scoring.get_scores`.

Two output modes are supported:
- `multiclass`: a single softmax distribution over the full action vocabulary
  (including unlabeled frames), exported as per-class logits.
- `multilabel_ovr`: one-vs-rest sigmoid probabilities for the non-"Nothing"
  actions, exported as per-action logits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import math
import numpy as np
import pandas as pd
import scipy.special
import sklearn.preprocessing
import torch
import tabm
import rtdl_num_embeddings

from sklearn.model_selection import GroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader

device = torch.device("cuda")

class TabMMulticlass:
    """
    TabM multi-class model with two modes:
    - multiclass: softmax probabilities
    - multilabel_ovr: sigmoid probabilities
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        save_dir: Optional[Path] = None,
        whitelist: Optional[pd.DataFrame] = None,
        feature_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        
        default_params: Dict[str, Any] = {
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "patience": 5,
            "n_epochs": 1000,
            "scaler": "StandardScaler",
            "embeddings": "PiecewiseLinearEmbeddings",
            "n_bins": 48,
            "d_embedding": 16,
            "arch_type": "tabm",  # "tabm" or "tabm-mini"
            "n_blocks": 2,
            "d_block": 256,
            "dropout": 0.2,
            "k": 32,
            "eval_batch_size": 4096,
        }
        default_settings: Dict[str, Any] = {
            "task_mode": "multiclass",
            "n_folds": 5,
            "seed": 0,
        }

        self.params: Dict[str, Any] = {**default_params, **(params or {}),}

        self.settings: Dict[str, Any] = {
            **default_settings, **(settings or {}),
        }

        self.task_mode: str = str(self.settings.get("task_mode", "multiclass"))

        self.seed: int = int(self.settings.get("seed", 0))
        self.save_dir: Optional[Path] = Path(save_dir) if save_dir else None
        self.feature_params: Dict[str, Any] = feature_params or {}
        self.whitelist: Optional[pd.DataFrame] = (
            whitelist.copy() if whitelist is not None else None
        )

        # Will be set after fit()
        self.feature_cols: Dict[str, List[str]] = {}
        self.calibs: Dict[str, Any] = {}
        self.summary: Dict[str, Any] = {}
        self.model_paths: Dict[str, List[Optional[str]]] = {}
        self.classes_: Optional[np.ndarray] = None

    def _format_meta_str(self) -> str:
        """Format model and training parameters for logging."""
        feat_meta = {
            k: v
            for k, v in self.feature_params.items()
            if k not in ["tmp_fnames_by_action", "bodyparts"]
        }
        meta: Dict[str, Any] = {
            "model_name": "tabm_multiclass",
            **feat_meta, **self.params, **self.settings,
        }
        items = [f"{k}={v}" for k, v in meta.items()]
        return ", ".join(items)

    def _sample_by_video(self, df: pd.DataFrame) -> np.ndarray:
        """
        Subsample rows with action == "Nothing" (unlabeled) while keeping all others.
        Uses sample_unl_percent from feature_params (e.g. 30 means 30%).
        Returns indices (relative to df) that should be kept.
        """
        sample_unl_percent = self.feature_params.get("sample_unl_percent")
        if sample_unl_percent is None or sample_unl_percent >= 100:
            return df.index.to_numpy(dtype=np.int64)
        
        rate = float(sample_unl_percent) / 100.0

        unlabeled_mask = (df["action"] == "Nothing").to_numpy()
        unlabeled_df = df.loc[unlabeled_mask]
        labeled_rows_idx = df.index[~unlabeled_mask].to_numpy(dtype=np.int64)

        def _sample_group(group: pd.DataFrame) -> pd.DataFrame:
            n_rows = len(group)
            n_samples = max(1, int(n_rows * rate))
            return group.sample(n=n_samples, random_state=self.seed)

        sampled_unlabeled = (
            unlabeled_df
            .groupby(["video_id", "agent_id", "target_id"], group_keys=False)[unlabeled_df.columns]
            .apply(_sample_group)
        )

        sampled_idx = sampled_unlabeled.index.to_numpy(dtype=np.int64)
        keep_idx = np.concatenate([labeled_rows_idx, sampled_idx])
        keep_idx.sort()

        return keep_idx

    def _print_fold_stats(
        self,
        fold: int,
        n_folds: int,
        train_dt: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        n_features: int,
    ) -> None:
        """
        Print fold statistics: train/valid unl/any ratios and feature count.
        """
        train_labels = train_dt.loc[train_idx, "action"]
        valid_labels = train_dt.loc[val_idx, "action"]

        tr_unl = int((train_labels == "Nothing").sum())
        va_unl = int((valid_labels == "Nothing").sum())

        tr_any = int(train_labels.size - tr_unl)
        va_any = int(valid_labels.size - va_unl)

        ratio_tr = tr_unl / tr_any if tr_any > 0 else float("inf")
        ratio_va = va_unl / va_any if va_any > 0 else float("inf")

        msg = (
            f"\n[fold {fold + 1}/{n_folds}] "
            f"train unl/any={tr_unl}/{tr_any} "
            f"(ratio={ratio_tr:.1f}) | "
            f"valid unl/any={va_unl}/{va_any} "
            f"(ratio={ratio_va:.1f}) | "
            f"n_features={n_features}\n"
        )
        print(msg, flush=True)

    def _make_fold_sets(
        self,
        train_dt: pd.DataFrame,
        fold_train_idx: np.ndarray,
        fold_val_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare train and validation indices for a single fold.
        """
        fold_train_df = train_dt.iloc[fold_train_idx].reset_index()
        fold_train_df = fold_train_df.rename(columns={"index": "orig_idx"})
        
        keep_idx = self._sample_by_video(fold_train_df)
        keep_train_idx = fold_train_df.loc[keep_idx, "orig_idx"].to_numpy(dtype=np.int64)

        return keep_train_idx, fold_val_idx
    
    def _build_preprocessor(
        self,
        X_train: np.ndarray,
        fold: int,
    ) -> sklearn.preprocessing.FunctionTransformer:
        scaler_name = str(self.params.get("scaler", "QuantileTransformer"))

        if scaler_name == "StandardScaler":
            scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        elif scaler_name == "QuantileTransformer":
            noise = (
                np.random.default_rng(self.seed + fold)
                .normal(0.0, 1e-5, X_train.shape)
                .astype(X_train.dtype)
            )
            scaler = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
                output_distribution="normal",
                subsample=10**9,
            ).fit(X_train + noise)
        else:
            raise ValueError(f"Unknown scaler: {scaler_name}")

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                scaler,
                self.save_dir / f"tabm_scaler_fold{fold}.pkl",
            )

        return scaler

    def _build_num_embeddings(
        self,
        X_train: np.ndarray,
        fold: int,
    ) -> Optional[nn.Module]:
        emb_type = str(self.params.get("embeddings", "PiecewiseLinearEmbeddings"))
        n_features = X_train.shape[1]

        if emb_type == "LinearReLUEmbeddings":
            return rtdl_num_embeddings.LinearReLUEmbeddings(n_features)
        if emb_type == "PeriodicEmbeddings":
            return rtdl_num_embeddings.PeriodicEmbeddings(n_features, lite=False)
        if emb_type == "PiecewiseLinearEmbeddings":
            emb_bins = rtdl_num_embeddings.compute_bins(
                torch.from_numpy(X_train),
                n_bins=int(self.params.get("n_bins", 48)),
            )
            if self.save_dir is not None:
                torch.save(
                    emb_bins,
                    self.save_dir / f"tabm_emb_bins_fold{fold}.pt",
                )
            return rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                emb_bins,
                d_embedding=int(self.params.get("d_embedding", 16)),
                activation=False,
                version="B",
            )
        if emb_type in ("none", "None", "", "null"):
            return None

        raise ValueError(f"Unknown embeddings type: {emb_type}")

    def _make_model(
        self,
        n_features: int,
        n_classes: int,
        num_embeddings: Optional[nn.Module],
    ) -> nn.Module:
        return tabm.TabM.make(
            n_num_features=n_features,
            cat_cardinalities=None,
            d_out=n_classes,
            num_embeddings=num_embeddings,
            arch_type=str(self.params.get("arch_type", "tabm")),
            n_blocks=int(self.params.get("n_blocks", 2)),
            d_block=int(self.params.get("d_block", 256)),
            dropout=float(self.params.get("dropout", 0.2)),
            k=int(self.params.get("k", 32)),
        ).to(device)

    def _loss_fn(
        self,
        model: nn.Module,
        logits: Tensor,
        y_true: Tensor,
    ) -> Tensor:
        """
        Compute loss from logits and true labels:
        - multiclass: cross-entropy loss
        - multilabel_ovr: binary cross-entropy loss
        
        logits: (N, k, n_classes) tensor of logits.
        y_true: array of true labels:
        - multiclass: (N,) array of class indices
        - multilabel_ovr: (N, n_classes) array of binary labels
        """
        k_heads = getattr(model.backbone, "k", 1)
        y_pred = logits.flatten(0, 1)                         # (batch*k, n_classes)
        y_true = y_true.repeat_interleave(k_heads, dim=0)     # (batch*k, ...) 

        # cross-entropy loss: (batch*k, n_classes) vs (batch*k,)
        if self.task_mode == "multiclass":
            return nn.functional.cross_entropy(y_pred, y_true)
        
        # binary cross-entropy loss: (batch*k, n_outputs) vs (batch*k, n_outputs)
        elif self.task_mode == "multilabel_ovr":
            return nn.functional.binary_cross_entropy_with_logits(y_pred, y_true) 

        raise ValueError(f"Unknown task_mode: {self.task_mode}")
        
    def _forward_probs(
        self,
        model: nn.Module,
        X_tensor: torch.Tensor,
    ) -> np.ndarray:
        """
        Forward pass through model and return shape (N_val, n_classes):
        - multiclass: softmax probabilities
        - multilabel_ovr: sigmoid probabilities
        X_tensor: (N_rows, n_features) tensor of input features.
        """
        model.eval()
        batch_size = int(self.params.get("eval_batch_size", 4096))
        
        probs_list: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, X_tensor.shape[0], batch_size):
                stop = min(start + batch_size, X_tensor.shape[0])
                xb = X_tensor[start:stop].to(device, non_blocking=True) # (batch_size, n_features)
                logits = model(xb, None).float().detach().cpu().numpy() # (batch_size, k, n_classes)
                
                if self.task_mode == "multiclass":
                    probs_heads = scipy.special.softmax(logits, axis=-1)
                else:
                    probs_heads = scipy.special.expit(logits)
                
                # average over k heads
                probs_mean = probs_heads.mean(axis=1)                    # (batch_size, n_classes)
                probs_list.append(probs_mean.astype("float32"))
        
        return np.concatenate(probs_list, axis=0) # (N, n_classes)

    def _compute_logloss(
        self,
        y_probs: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """
        Compute log loss from probabilities and true labels.
        y_probs: (N, n_classes) array of probabilities.
        y_val: array of true labels:
        - multiclass: (N,) array of class indices
        - multilabel_ovr: (N, n_classes) array of binary labels
        """
        if self.task_mode == "multiclass":
            p_true = np.clip(y_probs[np.arange(y_val.size), y_val], 1e-6, 1 - 1e-6)
            return float(-np.mean(np.log(p_true)))

        p = np.clip(y_probs, 1e-6, 1 - 1e-6)
        y_true = y_val.astype(np.float32)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def _train_single_fold(
        self,
        fold: int,
        train_dt: pd.DataFrame,
        feature_cols: List[str],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> Tuple[float, np.ndarray, Optional[str]]:
        if self.task_mode == "multiclass":
            n_classes = len(self.classes_)
        else:
            n_classes = len(self.acts)

        X_train = train_dt.loc[train_idx, feature_cols].to_numpy(
            dtype="float32",
            copy=False,
        )
        y_train = train_dt.loc[train_idx, "action_id"].to_numpy()
        
        X_val = train_dt.loc[val_idx, feature_cols].to_numpy(
            dtype="float32",
            copy=False,
        )
        y_val = train_dt.loc[val_idx, "action_id"].to_numpy()

        if self.task_mode == "multilabel_ovr":
            n_outputs = len(self.acts)
            y_train_bin = np.zeros((len(y_train), n_outputs), dtype=np.float32)
            y_val_bin = np.zeros((len(y_val), n_outputs), dtype=np.float32)
            
            for j, act_id in enumerate(self.act_indices):
                y_train_bin[:, j] = (y_train == act_id).astype(np.float32)
                y_val_bin[:, j] = (y_val == act_id).astype(np.float32)
            y_train = y_train_bin
            y_val = y_val_bin

        scaler = self._build_preprocessor(X_train, fold)
        X_train = scaler.transform(X_train).astype("float32", copy=False)
        X_val = scaler.transform(X_val).astype("float32", copy=False)

        num_embeddings = self._build_num_embeddings(X_train, fold)
        model = self._make_model(X_train.shape[1], n_classes, num_embeddings)

        batch_size = int(self.params.get("batch_size", 256))
        lr = float(self.params.get("lr", 1e-3))
        weight_decay = float(self.params.get("weight_decay", 5e-4))
        n_epochs = int(self.params.get("n_epochs", 1000))
        patience = int(self.params.get("patience", 10))

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        grad_scaler = torch.amp.GradScaler()

        X_train_tensor = torch.from_numpy(X_train)
        X_val_tensor = torch.from_numpy(X_val)

        if self.task_mode == "multiclass":
            y_train_tensor = torch.from_numpy(y_train).long() # dtype=int64
        else:
            y_train_tensor = torch.from_numpy(y_train).float() # dtype=float32

        train_dataset = torch.utils.data.TensorDataset(
            X_train_tensor,
            y_train_tensor,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        best_logloss = math.inf
        best_state: Optional[Dict[str, Any]] = None
        remaining_patience = patience

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(xb, None).float()  # (batch, k, n_classes)
                    loss = self._loss_fn(model, logits, yb)
                
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss += float(loss.detach().cpu())
                n_batches += 1

            mean_loss = epoch_loss / max(1, n_batches)

            val_probs = self._forward_probs(model, X_val_tensor)
            val_logloss = self._compute_logloss(val_probs, y_val)

            if val_logloss < best_logloss:
                best_logloss = val_logloss
                best_state = {
                    "model": {
                        k: v.detach().cpu()
                        for k, v in model.state_dict().items()
                    }
                }
                remaining_patience = patience
            else:
                remaining_patience -= 1

            print(
                f"Epoch={epoch} "
                f"train_loss={mean_loss:.4f} "
                f"val_logloss={val_logloss:.4f} "
                f"best_val_logloss={best_logloss:.4f}",
                flush=True,
            )

            if remaining_patience < 0:
                break

        if best_state is not None:
            model.load_state_dict(best_state["model"])

        val_probs = self._forward_probs(model, X_val_tensor)

        model_path: Optional[str] = None
        if self.save_dir is not None:
            models_dir = self.save_dir / "tabm"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path_obj = models_dir / f"tabm_fold{fold}.pt"
            torch.save(model.state_dict(), model_path_obj)
            model_path = str(model_path_obj)

        return best_logloss, val_probs, model_path

    def filter_by_whitelist(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter frame-level predictions by whitelist + 'Nothing' (unlabeled).
        Expects columns: - video_id, agent_id, target_id, action, score.
        """
        if self.whitelist is None:
            return df

        wl_keys = self.whitelist[
            ["video_id", "agent_id", "target_id", "action"]
        ].drop_duplicates()

        df = df.merge(
            wl_keys.assign(in_wl=True),
            on=["video_id", "agent_id", "target_id", "action"],
            how="left",
        )

        df = df[
            (df["in_wl"] == True) | (df["action"].isin(["unlabeled", "Nothing"]))
        ].copy()

        df = df.drop(columns=["in_wl"], errors="ignore")

        return df
    
    def fit(self, train_dt: pd.DataFrame, pseudo_labels: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train TabM model on frame-level dataset.
        pseudo_labels are not implemented.
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed + 1)
        torch.cuda.manual_seed_all(self.seed + 2)

        train_dt = train_dt.copy().reset_index(drop=True)

        aug_mask = train_dt["aug"].astype(bool)
        aug_idx = np.flatnonzero(aug_mask.to_numpy())
        base_mask = ~aug_mask
        base_idx = np.flatnonzero(base_mask.to_numpy())

        key_cols = ["aug", "video_id", "agent_id", "target_id", "video_frame", "action"]
        feature_cols = [c for c in train_dt.columns if c not in key_cols]

        if train_dt[feature_cols].isna().any().any():
            raise ValueError("NA values found in feature columns for TabM.")

        self.feature_cols = {"__all__": feature_cols}

        classes = np.sort(train_dt["action"].unique())
        self.classes_ = classes

        label_to_id = {label: idx for idx, label in enumerate(classes)}

        self.acts = classes[classes != "Nothing"].tolist()
        self.act_indices = [label_to_id[a] for a in self.acts]

        train_dt["action_id"] = train_dt["action"].map(label_to_id).astype(np.int64)

        groups = train_dt.loc[base_mask, "video_id"].to_numpy()
        n_folds = int(self.settings.get("n_folds", 5))
        
        n_groups = np.unique(groups).size
        n_splits = min(n_folds, n_groups)

        gkf = GroupKFold(n_splits=n_splits)
        
        if self.task_mode == "multiclass":
            class_names = [str(c) for c in self.classes_]
        else:
            class_names = self.acts

        oof_probs = np.full(
            (len(train_dt), len(class_names)),
            np.nan,
            dtype="float32",
        )

        fold_scores: List[float] = []
        model_paths: List[Optional[str]] = []
        
        print(self._format_meta_str())
        print()

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(X=base_idx, groups=groups),
        ):
            train_idx, val_idx = self._make_fold_sets(
                train_dt=train_dt,
                fold_train_idx=base_idx[train_idx],
                fold_val_idx=base_idx[val_idx],
            )
            if aug_idx.size > 0:
                val_videos = train_dt.loc[val_idx, "video_id"].to_numpy()
                aug_videos = train_dt.loc[aug_idx, "video_id"].to_numpy()
                aug_in_train_mask = ~np.isin(aug_videos, val_videos)
                aug_train_idx = aug_idx[aug_in_train_mask]

                train_idx = np.concatenate([train_idx, aug_train_idx])
                train_idx = np.unique(train_idx)
            
            self._print_fold_stats(
                fold=fold,
                n_folds=n_splits,
                train_dt=train_dt,
                train_idx=train_idx,
                val_idx=val_idx,
                n_features=len(feature_cols),
            )

            best_logloss, val_probs, model_path = self._train_single_fold(
                fold=fold,
                train_dt=train_dt,
                feature_cols=feature_cols,
                train_idx=train_idx,
                val_idx=val_idx
            )
            fold_scores.append(best_logloss)
            model_paths.append(model_path)

            # Store probabilities
            oof_probs[val_idx, :] = val_probs

        # Convert probabilities to logits (matching boostings pipeline)
        oof_probs = np.clip(oof_probs, 1e-6, 1.0 - 1e-6)
        oof_scores = np.log(oof_probs / (1.0 - oof_probs))

        oof = train_dt[["video_id", "agent_id", "target_id", "video_frame",]].copy()
        for j, act in enumerate(class_names):
            oof[act] = oof_scores[:, j]

        if self.task_mode == "multiclass":
            value_vars = class_names
        else:
            value_vars = self.acts 

        oof = oof.melt(
            id_vars=["video_id", "agent_id", "target_id", "video_frame"],
            value_vars=value_vars,
            var_name="action",
            value_name="score",
        )
        oof = oof[oof["score"].notna()].reset_index(drop=True)
        oof = self.filter_by_whitelist(oof)

        self.calibs = {}
        self.model_paths = {"tabm": model_paths}
        
        self.summary = {
            "model_name": "tabm_multiclass",
            **self.params,
            **self.settings,
            "cv_logloss_mean": float(np.mean(fold_scores)) if fold_scores else 0.0,
        }

        result = {
            "actions": self.acts,
            "feature_cols": self.feature_cols,
            "val_by_action": {},
            "calibs": self.calibs,
            "summary": self.summary,
            "model_paths": self.model_paths,
            "oof": oof,
        }
        return result

    def predict(self, new_dt: pd.DataFrame) -> pd.DataFrame:
        if self.save_dir is None:
            raise ValueError("TabMMulticlass.predict: save_dir is None, nothing to load.")
        
        id_cols = ["video_id", "agent_id", "target_id", "video_frame"]
        feature_cols = self.feature_cols.get("__all__")
        
        preds = new_dt[id_cols].copy()

        features_array = new_dt[feature_cols].to_numpy(
            dtype="float32",
            copy=False,
        )
        if self.task_mode == "multiclass":
            class_names = [str(c) for c in self.classes_]
        else:
            class_names = self.acts

        n_classes = len(class_names)
        emb_type = str(self.params.get("embeddings", "PiecewiseLinearEmbeddings"))

        probs_folds: List[np.ndarray] = []
        tabm_paths = self.model_paths.get("tabm", [])

        for fold, model_path in enumerate(tabm_paths):
            model_path_obj = Path(model_path)
            
            scaler_path = self.save_dir / f"tabm_scaler_fold{fold}.pkl"
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(features_array).astype(
                "float32",
                copy=False,
            )

            if emb_type == "LinearReLUEmbeddings":
                num_embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(
                    X_scaled.shape[1]
                )
            elif emb_type == "PeriodicEmbeddings":
                num_embeddings = rtdl_num_embeddings.PeriodicEmbeddings(
                    X_scaled.shape[1],
                    lite=False,
                )
            elif emb_type == "PiecewiseLinearEmbeddings":
                emb_bins_path = self.save_dir / f"tabm_emb_bins_fold{fold}.pt"
                if not emb_bins_path.exists():
                    raise FileNotFoundError(
                        "TabMMulticlass.predict: emb_bins file not found: "
                        f"{emb_bins_path}"
                    )
                emb_bins = torch.load(emb_bins_path, map_location="cpu")
                num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                    emb_bins,
                    d_embedding=int(self.params.get("d_embedding", 16)),
                    activation=False,
                    version="B",
                )
            elif emb_type in ("none", "None", "", "null"):
                num_embeddings = None
            else:
                raise ValueError(
                    f"TabMMulticlass.predict: unknown embeddings type '{emb_type}'."
                )

            model = self._make_model(
                X_scaled.shape[1],
                n_classes,
                num_embeddings,
            )
            state_dict = torch.load(model_path_obj, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

            X_tensor = torch.from_numpy(X_scaled)
            probs_fold = self._forward_probs(model, X_tensor)
            probs_folds.append(probs_fold.astype("float32"))

            del model, scaler, X_scaled, X_tensor, probs_fold
            torch.cuda.empty_cache()

        probs_mean = np.mean(probs_folds, axis=0).astype("float32")
        probs_mean = np.clip(probs_mean, 1e-6, 1.0 - 1e-6)
        scores = np.log(probs_mean / (1.0 - probs_mean))

        for j, name in enumerate(class_names):
            preds[name] = scores[:, j].astype("float32")

        return preds
