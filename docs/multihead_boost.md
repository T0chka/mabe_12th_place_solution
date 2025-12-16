# Multihead boosting models

The implementation lives in `py_pipeline/models/boostings_multihead.py`
and is orchestrated by the scripts in `py_pipeline/` (run_models.py, tune_models.py).
Supported models are: XGBoost, LightGBM, Catboost, and PyBoost.

## High‑level idea

Conceptually, the multihead boosting pipeline does the following:
- converts multi‑class sequence labeling into a set of per‑action binary tasks,
- uses video-level cross-validation and ensembles the resulting fold models at inference,
- samples background frames to control the ratio of true negatives and unlabeled data,
- outputs calibrated, prior-adjusted log-likelihood-ratio scores suitable for cross-action thresholding.

## Features:

For frames where `agent_id == target_id` only "self‑related" (not based on target mouse) features are kept.
The base feature table could be augmented with an action-specific window features (a selected set of kinematic variables expanded by multi-scale rolling means over lagged windows proportional to the video frame rate).

As a result, each action sees the part of feature space that is most
relevant for its semantics, while reusing the same underlying feature
matrix.

## Subsampling negatives by video

Negatives are sampled by video and classified into
two groups with respect to a given action:
1) Real negative frame: has an annotation with a different action, or
is unlabeled while the action is present in the video’s whitelist
(meaning it coud be annotated, the action just did not happen).
2) Maybe negative frame: is unlabeled and the action is absent from the video’s whitelist
(meaning the action could have occurred, but was outside the annotator’s labeling scope)

The final training set for a fold consists of all positives plus sampled negatives
from the two pools above.

## Weighting

If weigths are used, for positive runs the weight of each frame is set to be inversely
proportional to the run length (so long runs do not overpower the loss),
weights are then normalized so that the total positive weight remains comparable.

## Video‑level cross‑validation

We build folds by `video_id` so that each fold contains at least one video
with positives for the current action.

## Calibration

Raw scores from boosting models are not directly comparable, as they depend on the action,
the prevalence of positives, and specific training noise. Thus, calibration is required.
The calibration is learned only from data that were never used to fit
the underlying boosting models (out-of-fold predictions, OOF).

The pipeline applies a two‑step Platt calibration with prior correction:

1. A logistic mapping is fitted between raw margins and binary labels,
producing parameters `(a, b)` and the observed training prior `pi_train`.
2. The calibrated probability is then converted back to the logit
   scale, and the logit of `pi_train` is subtracted.

The final quantity is a **log‑likelihood ratio (log‑LR)**: how much
more likely the data are under "action present" vs "action absent",
relative to the training prior.

These log‑LR scores have two important properties:
- they are closer to being comparable across folds and actions,
- they can be combined with an arbitrary reference prior later, without
  retraining the model.

At inference time, each fold model for a given action produces a raw margin
for every frame. These margins are calibrated using the same Platt parameters
fitted on the OOF data. The calibrated log-LR scores from all folds
are then averaged to obtain the final score.