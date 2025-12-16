# TabM models

Implementation lives in `py_pipeline/models/tabm_multiclass.py` and plugs
into the same training/tuning runners under `py_pipeline/` as boostings.

## High-level idea

- Train one TabM neural network per lab.
- Two  modes: pure multiclass (softmax) or one-vs-rest multilabel heads (independent sigmoid per class).
For downstream compatibility with boosting models, out-of-fold and inference outputs (ensemble of the resulting fold models)
are converted from probabilities to log-likelihood ratios by clipping the probabilities
and applying a logit transform.

## Data

Augmented rows can optionally be used (but did not improve score in current implementation). 
Class list is derived from unique `action` values, with `Nothing` treated as background.
Numerical preprocessing uses either StandardScaler or QuantileTransformer fitted per fold and persisted to disk.
Optional numeric embeddings include Linear+ReLU, Periodic, or Piecewise Linear bins stored per fold.

## Subsampling negatives

Optional downsampling of unlabeled frames keeps all positives and actions
while sampling unlabeled frames per `(video, agent, target)` track.

## Cross-validation

GroupKFold operates over non-augmented rows grouped by `video_id`.
Number of splits equals the minimum of requested `n_folds` and available unique videos.

## Training loop

For each fold:
  - Apply the scaler and optional embeddings to train/validation subsets.
  - Build TabM with configurable depth, width, dropout, and number of heads.
  - Train with AdamW, AMP gradient scaling, shuffled mini-batches, and
    patience-based early stopping on validation logloss.
  - Loss is cross-entropy for multiclass or sigmoid BCE for multilabel
    heads, averaged across heads.

## Outputs

- Validation probabilities per fold are averaged across heads, clipped,
  converted to logits, and reshaped into long format with one row per
  `(video, agent, target, frame, action)`.
- Predictions are filtered by the whitelist table so only allowed
  `(video, agent, target, action)` pairs remain, with background class always retained.
- Paths to saved fold weights, scalers, and embedding bins are returned for
  inference.

## Inference

- Requires a `save_dir` populated by training artifacts.
- Recompute features in the same order, load each fold's scaler,
  embeddings, and TabM weights, then generate probabilities per fold.
- Average fold probabilities, clip, convert to logits, and emit one column
  per action alongside the identifying columns in the same log-LR space as
  boosting models.

