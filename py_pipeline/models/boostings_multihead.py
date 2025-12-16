"""
Unified boosting multihead models for the MABe Challenge.

This module implements a common training and inference pipeline for
binary one-vs-rest action classifiers based on gradient boosting.
It supports LightGBM, XGBoost, CatBoost, and PyBoost under a shared
interface.

For each laboratory, a separate model is trained per action using
cross-validation over videos. Models operate on frame-level features
and produce out-of-fold (OOF) raw scores or calibrated logits that are
compatible with `py_pipeline.utils.scoring.get_scores`.
"""

import re
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from py_boost import GradientBoosting
from sklearn.metrics import average_precision_score

from py_pipeline.utils.calibration import fit_platt, apply_platt
from py_pipeline.utils.training import (
    make_folds,
    make_fold_sets,
    print_data_counts,
)


class BoostingMultihead:
    """
    Unified boosting multihead model for behavior classification.
    
    Supports LightGBM, XGBoost, CatBoost, and PyBoost with shared training pipelines.
    Trains separate models for each action using cross-validation.
    """
    
    def __init__(
        self,
        booster_type: str = "lgbm_multihead",
        params: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
        save_dir: Optional[Path] = None,
        whitelist: pd.DataFrame = None,
        feature_params: Dict[str, Any] = None
    ):
        """
        Initialize model with parameters.
        
        Args:
            booster_type: "lgbm_multihead", "xgb_multihead", "cat_multihead", or "pyboost_multihead"
            params: Dict with booster parameters
            settings: Dict with training settings (including seed, use_weights)
            save_dir: Optional directory for saving models
            whitelist: DataFrame with whitelist (required)
            feature_params: Dict with feature parameters including
                          tmp_fnames_by_action and sample_neg_percent, sample_unl_percent
        """
        self.booster_type = booster_type
        
        default_settings = {
            "nrounds": 2000,
            "early_stop": 50,
            "print_every": 50,
            "use_weights": False,
            "pseudo_pos_weight": 0,
            "use_calibration": True,
            "k_folds": 5,
            "num_models": None,
            "seed": 1,
        }
        
        if booster_type == "lgbm_multihead":
            default_params = {
                "objective": "binary",
                "metric": "average_precision",
                "learning_rate": 0.03,
                "num_leaves": 63,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "max_depth": 20,
                "min_data_in_leaf": 32,
                "device": "cpu",
                "num_threads": -1,
                "lambda_l1": 0,
                "lambda_l2": 0,
                "min_gain_to_split": 0.001,
                "force_col_wise": True,
                "tree_learner": "data"
            }
        elif booster_type == "xgb_multihead":
            default_params = {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "learning_rate": 0.05,
                "max_depth": 8,
                "subsample": 0.8,
                "colsample_bynode": 0.7,
                "min_child_weight": 5,
                "max_delta_step": 3,
                "tree_method": "hist",
                "device": "cuda",
                "nthread": -1,
            }
        elif booster_type == "cat_multihead":
            default_params = {
                "loss_function": "Logloss",
                "learning_rate": 0.05,
                "depth": 6,
                "subsample": 0.8,
                "l2_leaf_reg": 3,
                "bootstrap_type": "Bernoulli",
                "task_type": "GPU",
                "thread_count": -1,
            }
        elif booster_type == "pyboost_multihead":
            default_params = {
                "loss": "crossentropy",
                "ntrees": 2000,
                "lr": 0.03,
                "subsample": 0.8,
                "colsample": 0.8,
                "max_depth": 6,
                "min_data_in_leaf": 10,
                "lambda_l2": 1.0,
                "gd_steps": 1,
                "use_hess": True,
                "max_bin": 256,
                "es": 50,
                "verbose": -1,
            }
        else:
            raise ValueError(f"Invalid booster type: {booster_type}")
        
        # Store default and override parameters
        self.default_params = default_params
        self.override_params = params or {}
        self.settings = {**default_settings, **(settings or {})}
        self.seed = int(self.settings.get("seed"))
        self.save_dir = save_dir
        self.feature_params = feature_params
        self.whitelist = whitelist.copy()
        
        # Will be set after fit()
        self.params_by_action: Dict[str, Dict[str, Any]] = {}
        self.model_paths: Dict[str, List[Optional[str]]] = {}
        self.feature_cols: Dict[str, List[str]] = {}
        self.actions: List[str] = []
        self.val_by_action: Dict[str, List] = {}
        self.calibs: Dict[str, Any] = {}
        self.summary: Optional[Dict[str, Any]] = None
    
    def _format_meta_str(self, booster_type: str, action: Optional[str] = None) -> str:
        """
        Format model, features and training parameters for logging.
        """
        feature_params_print = {
            k: v for k, v in self.feature_params.items()
            if k != "tmp_fnames_by_action"
        }

        booster_params = dict(self.params_by_action[action])

        meta = {
            "model_name": f"{booster_type}",
            **feature_params_print, **booster_params, **self.settings,
        }

        meta_str = ", ".join(
            f"{k}={v}" if k != "objective" else ""
            for k, v in meta.items()
            if k != "objective"
        )
        return meta_str

    def _build_params_by_action(self, acts: List[str]) -> (Dict[str, Dict[str, Any]], bool):
        """
        Build params_by_action dictionary and return if params are per-action or global.
        """
        override_keys = set(self.override_params)
        per_action = bool(override_keys) and override_keys.issubset(acts)

        if per_action:
            params_by_action = {
                a: {**self.default_params, **self.override_params.get(a, {})}
                for a in acts
            }
        else:
            params_by_action = {
                a: dict({**self.default_params, **self.override_params}  )
                for a in acts
            }
        return params_by_action, per_action
    
    def _prepare_action_data(
        self,
        action: str,
        train_dt: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Prepare action-specific data mask and features.
        """
        pos_all = train_dt[train_dt["action"] == action]
        if len(pos_all) == 0:
            raise ValueError(f"No positives for action {action}")
        
        # Filter optional features by action
        opt_feature_cols = [
            c for c in feature_cols
            if re.search(r"_lagw\d+_", c)
        ]
        
        a_opt_feature_cols = opt_feature_cols
        tmp_fnames_by_action = self.feature_params.get("tmp_fnames_by_action", {})
        if action in tmp_fnames_by_action:
            allowed_opt = set(tmp_fnames_by_action[action])
            a_opt_feature_cols = [
                c for c in opt_feature_cols
                if c in allowed_opt
            ]
        
        filtered_feature_cols = [
            c for c in feature_cols
            if c not in opt_feature_cols
        ] + a_opt_feature_cols
        
        # if self-action - subset features and rows where agent_id == target_id
        train_self_action = (pos_all["agent_id"] == pos_all["target_id"]).all()
        
        if train_self_action:
            a_feature_cols = [
                c for c in filtered_feature_cols
                if "ag_" in c or "m1m1" in c
            ]
            act_mask = train_dt["agent_id"].to_numpy() == train_dt["target_id"].to_numpy()
        else:
            a_feature_cols = filtered_feature_cols
            act_mask = train_dt["agent_id"].to_numpy() != train_dt["target_id"].to_numpy()
        
        return {
            "act_mask": act_mask,
            "a_feature_cols": a_feature_cols
        }
    
    def _compute_weights(
        self,
        dt_all: pd.DataFrame,
        action: str,
        train_idx: np.ndarray
    ) -> np.ndarray:
        sort_cols = ["video_id", "agent_id", "target_id", "video_frame"]
        
        dt_min = dt_all.iloc[train_idx][sort_cols + ["action"]].copy()
        dt_min["order_idx"] = np.arange(len(dt_min), dtype="int32")
        dt_min["y"] = dt_min["action"] == action
        dt_min = dt_min.sort_values(sort_cols)
        
        dt_min["change"] = (
            dt_min.groupby(["video_id", "agent_id", "target_id"])["y"]
            .shift(1, fill_value=False) != dt_min["y"]
        )
        dt_min["run_id"] = dt_min.groupby(
            ["video_id", "agent_id", "target_id"]
        )["change"].cumsum()
        
        run_len = (
            dt_min[dt_min["y"]]
            .groupby(["video_id", "agent_id", "target_id", "run_id"])
            .size()
            .reset_index(name="run_len")
        )
        dt_min["weight"] = 1.0
        
        if not run_len.empty:
            dt_min = dt_min.merge(
                run_len,
                on=["video_id", "agent_id", "target_id", "run_id"],
                how="left"
            )
            mask_pos = dt_min["y"].to_numpy()
            inv_len = 1.0 / dt_min.loc[mask_pos, "run_len"].to_numpy()
            dt_min.loc[mask_pos, "weight"] = inv_len
            if mask_pos.any():
                sum_pos = float(dt_min.loc[mask_pos, "weight"].sum())
                if sum_pos > 0.0:
                    k = mask_pos.sum() / sum_pos
                    dt_min.loc[mask_pos, "weight"] *= k
        dt_min = dt_min.sort_values("order_idx")
        weights = dt_min["weight"].to_numpy(dtype="float32", copy=False)
        return weights
    
    def _build_pseudo_mask_by_action(
        self,
        train_dt: pd.DataFrame,
        pseudo_labels: Optional[pd.DataFrame],
        whitelist: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Constructs boolean pseudo-label masks for each action:
        1. Projects pseudo-labels onto train_dt rows by aligning keys and recovering row indices
        2. Ensures that pseudo-labels are applied only to frames without annotations
        3. Excludes all (video_id, agent_id, target_id, action) combinations that already appear
        in the whitelist (in such cases the absence of an annotation indicates a true negative,
        which must not be replaced by a pseudo-label).
        """
        if pseudo_labels is None or pseudo_labels.empty:
            return {}

        keys = ["video_id", "agent_id", "target_id", "video_frame"]
        tmp_idx = train_dt.reset_index().rename(columns={"index": "row_id"})
        tmp_idx = tmp_idx[keys + ["row_id", "action"]]

        pseudo_join = pseudo_labels.merge(
            tmp_idx,
            on=keys,
            how="inner",
        )
        pseudo_join = pseudo_join.rename(
            columns={"action_x": "pseudo_action", "action_y": "orig_action"}
        )
        pseudo_join = pseudo_join[pseudo_join["orig_action"] == "Nothing"].copy()
        n_rows = len(train_dt)
        pseudo_mask_by_action: Dict[str, np.ndarray] = {}

        for a in sorted(pseudo_join["pseudo_action"].unique()):
            pj_a = pseudo_join[pseudo_join["pseudo_action"] == a]
            wl_a = whitelist[whitelist["action"] == a]
            combos_with_a = set(
                zip(
                    wl_a["video_id"].to_numpy(),
                    wl_a["agent_id"].to_numpy(),
                    wl_a["target_id"].to_numpy(),
                )
            )
            mask_not_wl = np.fromiter(
                (
                    (v, ag, tg) not in combos_with_a
                    for v, ag, tg in zip(
                        pj_a["video_id"].to_numpy(),
                        pj_a["agent_id"].to_numpy(),
                        pj_a["target_id"].to_numpy(),
                    )
                ),
                dtype=bool,
            )
            pj_a = pj_a[mask_not_wl]
            if pj_a.empty:
                continue

            row_ids = pj_a["row_id"].to_numpy(dtype=np.int64)
            mask_all = np.zeros(n_rows, dtype=bool)
            mask_all[row_ids] = True
            pseudo_mask_by_action[a] = mask_all

        return pseudo_mask_by_action
    
    def _build_labels_with_pseudo(
        self,
        labels: np.ndarray,
        action: str,
        train_idx: np.ndarray,
        pseudo_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Builds new labels array with pseudo-labels
        applied to the training data if provided.
        """
        if pseudo_mask is None:
            return labels

        n_rows = labels.shape[0]
        train_mask = np.zeros(n_rows, dtype=bool)
        train_mask[train_idx] = True

        pseudo_train_mask = pseudo_mask & train_mask

        if not pseudo_train_mask.any():
            return labels

        n_unlabeled_before = int((labels[train_idx] == "Nothing").sum())

        labels_new = labels.copy()
        labels_new[pseudo_train_mask] = action

        n_changed = int(pseudo_train_mask.sum())
        n_pos_after = int((labels_new[train_idx] == action).sum())

        frac_from_unlabeled = (
            n_changed / n_unlabeled_before if n_unlabeled_before > 0 else 0.0
        )

        frac_pseudo_among_pos = (
            n_changed / n_pos_after if n_pos_after > 0 else 0.0
        )

        print(
            f"[Pseudo labels]: added={n_changed} "
            f"({frac_from_unlabeled*100:.2f}% of unlabeled, "
            f"{frac_pseudo_among_pos*100:.2f}% of final positives)"
        )

        return labels_new
    
    def _fit_lgbm(
        self,
        booster_params: Dict[str, Any],
        train_data: np.ndarray,
        valid_data: np.ndarray,
        train_labels: np.ndarray,
        valid_labels: np.ndarray,
        weights: Optional[np.ndarray],
        seed: int,
    ) -> Tuple[Any, np.ndarray]:
        dtrain = lgb.Dataset(
            train_data,
            label=train_labels,
            weight=weights,
            free_raw_data=True,
        )
        dvalid = lgb.Dataset(
            valid_data,
            label=valid_labels,
            reference=dtrain,
            free_raw_data=True,
        )
        print_every = self.settings.get("print_every", 0)
        verbosity = -1 if print_every == 0 else 1
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.settings["early_stop"],
                verbose=False,
            )
        ]
        if print_every > 0:
            callbacks.append(lgb.log_evaluation(period=print_every))
        model = lgb.train(
            params={**booster_params, "seed": seed, "verbosity": verbosity},
            train_set=dtrain,
            num_boost_round=self.settings["nrounds"],
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=callbacks,
        )
        best_score = model.best_score["valid"]["average_precision"]
        print(
            f"best iter={model.best_iteration} | "
            f"best score={best_score:.6f}"
        )
        preds = model.predict(
            valid_data,
            num_iteration=model.best_iteration,
            raw_score=True,
        ).astype("float32")

        del dtrain, dvalid

        return model, preds, best_score

    def _fit_xgb(
        self,
        booster_params: Dict[str, Any],
        train_data: np.ndarray,
        valid_data: np.ndarray,
        train_labels: np.ndarray,
        valid_labels: np.ndarray,
        weights: Optional[np.ndarray],
        seed: int,
    ) -> Tuple[Any, np.ndarray]:
        dtrain = xgb.DMatrix(
            data=train_data,
            label=train_labels,
            weight=weights,
        )
        dvalid = xgb.DMatrix(
            data=valid_data,
            label=valid_labels,
        )
        print_every = self.settings.get("print_every", 0)
        callbacks = []
        if print_every > 0:
            callbacks.append(
                xgb.callback.EvaluationMonitor(period=print_every)
            )

        model = xgb.train(
            params={**booster_params, "seed": seed},
            dtrain=dtrain,
            num_boost_round=self.settings["nrounds"],
            evals=[(dvalid, "valid")],
            verbose_eval=print_every if print_every > 0 else False,
            callbacks=callbacks,
            early_stopping_rounds=self.settings["early_stop"],
        )
        best_iter = model.attributes().get("best_iteration")
        best_score = float(model.attributes().get("best_score"))
        if best_iter:
            print(
                f"best iter={best_iter} | "
                f"best score={best_score:.6f}"
            )
        preds = model.predict(
            dvalid,
            output_margin=True,
        ).astype("float32")

        del dtrain, dvalid

        return model, preds, best_score

    def _fit_catboost(
        self,
        booster_params: Dict[str, Any],
        train_data: np.ndarray,
        valid_data: np.ndarray,
        train_labels: np.ndarray,
        valid_labels: np.ndarray,
        weights: Optional[np.ndarray],
        seed: int,
    ) -> Tuple[Any, np.ndarray]:
        dtrain = Pool(
            data=train_data,
            label=train_labels,
            weight=weights,
        )
        dvalid = Pool(
            data=valid_data,
            label=valid_labels,
        )
        print_every = self.settings.get("print_every", 0)
        verbose = print_every if print_every > 0 else 0

        params = dict(booster_params)

        model = CatBoostClassifier(
            iterations=self.settings["nrounds"],
            random_seed=seed,
            verbose=verbose,
            early_stopping_rounds=self.settings["early_stop"],
            **params,
        )
        model.fit(
            dtrain,
            eval_set=dvalid,
            use_best_model=True,
        )
        best_iter = model.get_best_iteration()
        val_scores = model.get_best_score().get("validation", {})
        metric_name = params.get("eval_metric")
        if metric_name is None:
            metric_name = next(iter(val_scores.keys()))
        best_score = float(val_scores[metric_name])
        print(
            f"best iter={best_iter} | "
            f"best score={best_score:.6f}"
        )
        preds = model.predict(
            valid_data,
            prediction_type="RawFormulaVal",
        ).astype("float32")

        del dtrain, dvalid

        return model, preds, best_score

    def _fit_pyboost(
        self,
        booster_params: Dict[str, Any],
        train_data: np.ndarray,
        valid_data: np.ndarray,
        train_labels: np.ndarray,
        valid_labels: np.ndarray,
        weights: Optional[np.ndarray],
        seed: int,
    ) -> Tuple[Any, np.ndarray]:
        if weights is not None:
            model = GradientBoosting(**booster_params)
        else:
            model = GradientBoosting(**booster_params)

        model.fit(
            train_data,
            train_labels,
            eval_sets=[{"X": valid_data, "y": valid_labels}],
        )

        probs = model.predict(valid_data)
        probs = np.asarray(probs, dtype="float32")[:, 1]
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        raw_scores = np.log(probs / (1.0 - probs)).astype("float32")

        best_score = float(average_precision_score(valid_labels, probs))
        print(f"best score={best_score:.6f}")

        return model, raw_scores, best_score
    
    def fit(self, train_dt: pd.DataFrame, pseudo_labels: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        key_cols = ["video_id", "agent_id", "target_id", "video_frame", "action", "aug"]
        
        train_dt = train_dt.copy().reset_index(drop=True)
        feature_cols = [c for c in train_dt.columns if c not in key_cols]
        
        if train_dt[feature_cols].isna().any().any():
            raise ValueError("NA values found in feature columns")
        
        # get ground truth labels and feature matrix
        labels = train_dt["action"].to_numpy()
        full_feature_matrix = train_dt[feature_cols].to_numpy(
            dtype="float32",    
            copy=False
        )
        column_index = {c: i for i, c in enumerate(feature_cols)}
        
        # build pseudo-label masks for each action if pseudo_labels are not None
        pseudo_mask_by_action = self._build_pseudo_mask_by_action(
            train_dt=train_dt,
            pseudo_labels=pseudo_labels,
            whitelist=self.whitelist,
        )
        
        # get unique actions and video ids
        acts = sorted([
            a for a in train_dt["action"].unique()
            if a != "Nothing"
        ])
        vids_all = sorted(train_dt["video_id"].unique().tolist())
        
        # build params_by_action dictionary
        self.params_by_action, self.params_per_action = self._build_params_by_action(acts)

        # initialize storages
        fold_scores_by_action: Dict[str, List[float]] = {a: [] for a in acts}
        cv_scores_all_actions: List[float] = []
        
        self.model_paths = {a: [] for a in acts}
        self.val_by_action = {a: [] for a in acts}
        self.feature_cols = {a: [] for a in acts}
        self.calibs = {a: None for a in acts}
        
        oof = train_dt[[
            "video_id", "agent_id", "target_id", "video_frame", "action"
        ]].copy()
        
        oof = oof.rename(columns={"action": "true_action"})
        
        # iterate over actions
        for a in acts:
            a_seed = self.seed + acts.index(a) + 1
            
            # prepare action-specific data
            action_data = self._prepare_action_data(a, train_dt, feature_cols)
            
            act_mask = action_data["act_mask"]
            act_feature_cols = action_data["a_feature_cols"]
            
            self.feature_cols[a] = act_feature_cols
            col_idx = [column_index[c] for c in act_feature_cols]
            
            # get pseudo-label mask for the current action (or None)
            pseudo_mask_a = pseudo_mask_by_action.get(a)

            # make folds
            num_models=self.settings.get("num_models", None)
            folds = make_folds(
                pos_all=train_dt[act_mask & (labels == a)],
                vids_all=vids_all,
                k_folds=self.settings.get("k_folds"),
                num_models=num_models,
                seed=a_seed
            )
            
            print(
                f"\n==== Action {acts.index(a) + 1}/{len(acts)}: {a} "
                f"({len(folds)} folds, {len(act_feature_cols)} features)\n"
            )
            print(self._format_meta_str(self.booster_type, action=a))
            print()
            
            self.model_paths[a] = [None] * len(folds)
            self.val_by_action[a] = [None] * len(folds)
            oof["raw_sum"] = 0.0
            oof["raw_n"] = 0
            
            # iterate over folds
            for f, split in enumerate(folds):
                print(f"[fold {f + 1}/{len(folds)}] ", end="", flush=True)
                f_seed = self.seed + a_seed * 100 + f
                
                train_vids = split["train"]
                val_vids = split["val"]
                
                self.val_by_action[a][f] = val_vids
                
                # replace unlabeled train frames with pseudo-labels
                train_vids_mask = train_dt["video_id"].isin(train_vids).to_numpy()
                train_idx = np.nonzero(train_vids_mask & act_mask)[0]

                labels_for_action = self._build_labels_with_pseudo(
                    labels=labels,
                    action=a,
                    train_idx=train_idx,
                    pseudo_mask=pseudo_mask_a,
                )

                # get train indices for the current fold (all positive and dawnsampled negative frames)
                pos_mask_fold = (labels_for_action == a) & act_mask
                neg_mask_fold = (labels_for_action != a) & act_mask

                fold_sets = make_fold_sets(
                    pos_all=train_dt[pos_mask_fold],
                    neg_all=train_dt[neg_mask_fold],
                    train_vids=train_vids,
                    val_vids=val_vids,
                    whitelist=self.whitelist,
                    sample_neg_percent=self.feature_params.get("sample_neg_percent"),
                    sample_unl_percent=self.feature_params.get("sample_unl_percent"),
                    seed=f_seed,
                )

                train_idx = fold_sets["train_idx"]
                valid_idx = fold_sets["valid_idx"]

                if pseudo_mask_a is not None:
                    w_pseudo = float(self.settings.get("pseudo_pos_weight"))
                    weights = np.ones(train_idx.shape[0], dtype="float32")
                    mask_pseudo_train = pseudo_mask_a[train_idx]
                    if mask_pseudo_train.any():
                        weights[mask_pseudo_train] = w_pseudo
                elif self.settings.get("use_weights", False):
                    weights = self._compute_weights(train_dt, a, train_idx)
                else:
                    weights = None

                train_data = full_feature_matrix[train_idx][:, col_idx]
                valid_data = full_feature_matrix[valid_idx][:, col_idx]

                train_labels = (labels_for_action[train_idx] == a).astype("int8")
                valid_labels = (labels_for_action[valid_idx] == a).astype("int8")

                print_data_counts(train_labels, valid_labels)

                fit_args = dict(
                    booster_params=self.params_by_action[a],
                    train_data=train_data,
                    valid_data=valid_data,
                    train_labels=train_labels,
                    valid_labels=valid_labels,
                    weights=weights,
                    seed=f_seed,
                )

                if self.booster_type == "lgbm_multihead":
                    model, preds_va, best_score = self._fit_lgbm(**fit_args)
                elif self.booster_type == "xgb_multihead":
                    model, preds_va, best_score = self._fit_xgb(**fit_args)
                elif self.booster_type == "cat_multihead":
                    model, preds_va, best_score = self._fit_catboost(**fit_args)
                elif self.booster_type == "pyboost_multihead":
                    model, preds_va, best_score = self._fit_pyboost(**fit_args) 
                else:
                    raise ValueError(f"Invalid booster type: {self.booster_type}")
                
                fold_scores_by_action[a].append(best_score)

                model_path = ""
                if self.save_dir is not None:
                    action_dir = self.save_dir / a
                    action_dir.mkdir(parents=True, exist_ok=True)
                    if self.booster_type == "lgbm_multihead":
                        model_path = str(action_dir / f"{a}_fold{f:02d}.txt")
                        model.save_model(model_path)
                    elif self.booster_type == "xgb_multihead":
                        model_path = str(action_dir / f"{a}_fold{f:02d}.ubj")
                        model.save_model(model_path)
                    elif self.booster_type == "cat_multihead":
                        model_path = str(action_dir / f"{a}_fold{f:02d}.cbm")
                        model.save_model(model_path)
                    elif self.booster_type == "pyboost_multihead":
                        model_path = str(action_dir / f"{a}_fold{f:02d}.pkl")
                        with open(model_path, "wb") as f_out:
                            pickle.dump(model, f_out)
                
                del model, train_data, valid_data, train_labels, valid_labels, weights
                
                self.model_paths[a][f] = model_path if model_path else None
                raw_sum = oof["raw_sum"].to_numpy()
                raw_n = oof["raw_n"].to_numpy()
                raw_sum[valid_idx] += preds_va
                raw_n[valid_idx] += 1
                oof["raw_sum"] = raw_sum
                oof["raw_n"] = raw_n

                del fold_sets, raw_sum, raw_n, train_idx, valid_idx
            
            del act_mask, act_feature_cols, col_idx, pseudo_mask_a, folds, num_models

            cv_score_a = float(np.mean(fold_scores_by_action[a]))
            cv_scores_all_actions.append(cv_score_a)

            mask = oof["raw_n"] > 0
            oof.loc[mask, "raw_score"] = (
                oof.loc[mask, "raw_sum"] /
                oof.loc[mask, "raw_n"].astype("float32")
            )
            oof = oof.drop(columns=["raw_sum", "raw_n"])
            oof["true_label"] = (oof["true_action"] == a).astype("int8")
            
            if self.settings.get("use_calibration", True):
                fit_idx = oof["raw_score"].notna()
                self.calibs[a] = fit_platt(
                    oof.loc[fit_idx, "raw_score"].values,
                    oof.loc[fit_idx, "true_label"].values
                )
                for f, val_vids in enumerate(self.val_by_action[a]):
                    idx_va = oof["video_id"].isin(val_vids)
                    idx_tr = ~idx_va
                    
                    true_label_tr = oof.loc[idx_tr & fit_idx, "true_label"].values
                    score_tr = oof.loc[idx_tr & fit_idx, "raw_score"].values
                    
                    if np.unique(true_label_tr).size != 2:
                        raise ValueError(f"No both classes for {a} fold {f}")
                    
                    a_k, b_k, pi_k = fit_platt(score_tr, true_label_tr)
                    
                    oof.loc[idx_va, a] = apply_platt(
                        oof.loc[idx_va, "raw_score"].values,
                        a_k, b_k, pi_k
                    )
                oof = oof.drop(columns=["raw_score", "true_label"])
            else:
                oof.loc[mask, a] = oof.loc[mask, "raw_score"].values
                oof = oof.drop(columns=["raw_score", "true_label"])
        
        oof = oof.drop(columns=["true_action"])
        oof = oof.melt(
            id_vars=["video_id", "agent_id", "target_id", "video_frame"],
            value_vars=acts,
            var_name="action",
            value_name="score"
        )
        oof = oof[oof["score"].notna()].reset_index(drop=True)
        
        self.oof = oof
        self.actions = acts

        params_summary = (
            {"params": "(different per action)"}
            if self.params_per_action
            else {**self.default_params, **self.override_params}
        )
        self.summary = {
            "model_name": self.booster_type,
            "cv_valid_metric": np.mean(cv_scores_all_actions),
            **params_summary, **self.settings,
        }
        result = {
            "actions": self.actions,
            "feature_cols": self.feature_cols,
            "val_by_action": self.val_by_action,
            "calibs": self.calibs,
            "summary": self.summary,
            "model_paths": self.model_paths,
            "oof": self.oof,
        }

        return result
    
    def predict(self, new_dt: pd.DataFrame) -> pd.DataFrame:
        preds = new_dt[[
            "video_id", "agent_id", "target_id", "video_frame"
        ]].copy()
        
        all_feature_cols = sorted(
            {c for cols in self.feature_cols.values() for c in cols}
        )
        full_dtest = new_dt[all_feature_cols].to_numpy(dtype="float32", copy=False)
        col_index = {c: i for i, c in enumerate(all_feature_cols)}

        for a in self.actions:
            model_paths = self.model_paths[a]
            feature_cols = self.feature_cols[a]
            col_idx = [col_index[c] for c in feature_cols]
            dtest = full_dtest[:, col_idx]
            
            fold_scores = []
            for model_path in model_paths:
                if not model_path:
                    continue
                model_path_obj = Path(model_path)
                if not model_path_obj.exists():
                    continue
                if self.booster_type == "lgbm_multihead":
                    model = lgb.Booster(model_file=str(model_path_obj))
                    score = model.predict(
                        dtest,
                        num_iteration=model.best_iteration,
                        raw_score=True
                    )
                elif self.booster_type == "xgb_multihead":
                    model = xgb.Booster()
                    model.load_model(str(model_path_obj))
                    score = model.predict(xgb.DMatrix(dtest), output_margin=True)
                elif self.booster_type == "cat_multihead":
                    model = CatBoostClassifier()
                    model.load_model(str(model_path_obj))
                    score = model.predict(dtest, prediction_type="RawFormulaVal")
                elif self.booster_type == "pyboost_multihead":
                    with open(model_path_obj, "rb") as f_in:
                        model = pickle.load(f_in)
                    probs = model.predict(dtest)
                    probs = np.asarray(probs, dtype="float32")[:, 1]
                    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
                    score = np.log(probs / (1.0 - probs)).astype("float32")
                else:
                    raise ValueError(f"Invalid booster type: {self.booster_type}")
                
                fold_scores.append(score.astype("float32"))
                del model
            
            if not fold_scores:
                continue
            
            calib_params = self.calibs.get(a)
            if calib_params is None:
                z = np.mean(fold_scores, axis=0, dtype="float32")
            else:
                a_c, b_c, pi_c = calib_params
                z_folds = [
                    apply_platt(score, a_c, b_c, pi_c).astype("float32")
                    for score in fold_scores
                ]
                z = np.mean(z_folds, axis=0, dtype="float32")
            preds[a] = z
            
            del dtest, fold_scores, z
        return preds
