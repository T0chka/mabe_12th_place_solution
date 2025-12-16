# MABe Challenge - Social Action Recognition in Mice

The task is formulated as a frame-level action recognition problem. For each video and each valid (agent, target) mouse pair, the model predicts which social action is being performed at each frame, including self-directed actions. Frame-level predictions are subsequently merged into action intervals. The set of valid actions varies across laboratories, videos, and mouse pairs.

The training data span 19 laboratories and include a limited number of annotated videos (~850) alongside a substantially larger collection of unlabeled videos with pose tracking only (~8k videos). The test set consists of approximately 200 videos with pose data and no action annotations.

# EDA notebooks (kaggle)

Several exploratory notebooks were used to understand the structure, limitations, and implicit assumptions of the dataset. These analyses informed key modeling choices, including task formulation, negative sampling, calibration strategy, and postprocessing.

- [Overall data overview](https://www.kaggle.com/code/antoninadolgorukova/mabe-an-exploratory-data-safari)
- [Deeper analysis of data constraints and their modeling implications](https://www.kaggle.com/code/antoninadolgorukova/mabe-data-constraints-modeling-implications)
- [Data quality checks and identified issues](https://www.kaggle.com/code/antoninadolgorukova/mabe-sanity-checks-data-issues)


# Solution pipeline

The shared pipeline starts inside `py_pipeline/` and branches into model-specific trainers
described in the linked documents.

## Data assembly

Data were prepared as a combination of all frames with tracks for each video_id and each (agent_id, target_id) pair from the whitelist (behaviors_labeled column in train/test csv files). Frames annotated with a given behavior were treated as positive examples for that action; unlabeled frames were treated as background.

## Shared feature space

All models reuse a common feature representation built in `py_pipeline/utils/features_new.py` and composed of three groups:

1. Invariant, agent-centric geometry and kinematics (relative distances,
   angles, approach/retreat cues, motion alignment, and time since contact).
2. Single-mouse descriptors (forward and lateral velocities, accelerations,
   total speed, turning rate, path curvature, motion smoothness, body
   length, ear gap, and related stability metrics).
3. Pairwise distances between selected body parts within each mouse and across the agent-target pair, computed either on raw landmarks or on aggregated body regions obtained by averaging upper and lower body parts.

Invariants and single-mouse descriptors can optionally be normalized by agent body length
(or its median within a track), measured as the Euclidean distance between the centroid
of upper body landmarks and the centroid of lower body landmarks.

For a selected subset of columns, the pipeline can optionally append temporal
summary features that store the mean value over up to three lagged windows,
aligned with the per-track lag scale (history-aware covariates).

## Cross-validation

Folds are built at the video level. Each model applies k-fold (or repeated) cross-validation
and produces complete out-of-fold (OOF) predictions for the entire training set. 
All currently implemented models ensemble their fold-specific models at inference.

## Shared outputs

The models emit per-frame log-likelihood ratios for every action.

## Model-specific docs

- [Multihead boostings](./docs/multihead_boost.md)
- [TabM](./docs/tabm_multiclass.md)

## Postprocessing
The frame-level predictions were filtered to the whitelist, keeping only valid (video, agent, target, action) combinations. Smoothing was then applied using rolling windows whose sizes correspond to the median action durations observed in the training data. Action-specific thresholds were fitted on OOF, and applied after smoothing; a single action per frame was selected using argmax. Consecutive frames with the same chosen action were then merged into intervals, forming the final submission.

# Repository Structure

```
mabe_12th_place_solution/
├── py_pipeline/             # Main Python pipeline
│   ├── models/              # Model implementations
│   │   ├── boostings_multihead.py  # LightGBM, XGBoost, CatBoost, Py-Boost
│   │   └── tabm_multiclass.py      # TabM neural network model
│   ├── utils/                # Utility functions
│   │   ├── data_loads.py     # Data loading and preparation
│   │   ├── features.py       # Feature engineering
│   │   ├── training.py       # Training utilities
│   │   ├── scoring.py        # Evaluation metrics
│   │   ├── calibration.py    # Platt scaling
│   │   ├── thresholds.py     # Threshold optimization
│   │   └── postprocessing.py # Postprocessing functions
│   ├── run_models.py         # Main training script
│   ├── tune_models.py        # Grid search tuning
│   └── optune_params.py      # Optuna hyperparameter tuning
├── configs/                  # Configuration files
│   ├── lab_config.yaml       # Lab-specific settings
│   ├── feats_config_*.yaml   # Feature configurations
│   └── params_config_*.yaml  # Model parameter configurations
├── docs/                     # Model documentation
│   ├── multihead_boost.md
│   └── tabm_multiclass.md
└── requirements.txt          # Python dependencies
```

# Usage

## Setup

**Requirements:**
- Python >=3.11

**Installation:**

Clone the repository:
```bash
git clone <repository-url>
cd mabe_12th_place_solution
```

Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

Install PyTorch with CUDA support (required for TabM model, must be installed separately):
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install remaining dependencies:
```bash
pip install -r requirements.txt
```

Prepare data directory:
```bash
mkdir -p data
# Place competition data files in the data/ directory
```

**Note:** PyTorch with CUDA must be installed separately using `--index-url` as it's not available on standard PyPI. All other dependencies (including `cupy-cuda12x` and `py-boost`) are listed in `requirements.txt`.


## Running Models

The main training script is `py_pipeline/run_models.py`. To train a model:

1. **Edit configuration** in `run_models.py`:
   - Set `EXP_PARAMS["submit_num"]` - unique number for this experiment (used in output directory name)
   - Set `EXP_PARAMS["model_name"]` to one of:
     - `"lgbm_multihead"` - LightGBM
     - `"xgb_multihead"` - XGBoost
     - `"cat_multihead"` - CatBoost
     - `"pyboost_multihead"` - Py-Boost
     - `"tabm_multiclass"` - TabM
   - Set `EXP_PARAMS["labs"]` to a list of lab names or `None` for all labs
   - Set `EXP_PARAMS["feats_config"]` to path to feature config or `None`
   - Set `EXP_PARAMS["params_config"]` to path to params config or `None`
   - Set `EXP_PARAMS["pseudo_path"]` to path to pseudo labels CSV or `None` (optional)
   - Adjust `MODEL_SETTINGS` for model-specific options (k_folds, seed, etc.)

2. **Run the script**:
```bash
python py_pipeline/run_models.py
```

The script will:
- Load data and build features according to the configuration
- Train models using cross-validation for each lab
- Save out-of-fold predictions, calibration parameters, and thresholds
- Write training logs to `submits_data/submit{submit_num}/training_log.csv`

## Tuning Models

### Grid Search Tuning

Use `py_pipeline/tune_models.py` for grid search over feature sets or other parameters:

1. **Edit configuration** in `tune_models.py`:
   - Set `EXP_PARAMS` (model_name, labs, feats_config, etc.)
   - Define `WINDOW_SETS` for feature window tuning
   - Define `LAG_SEC_GRID`, `SAMPLE_NEG_GRID`, `SAMPLE_UNL_GRID` for other parameters

2. **Run the script**:
```bash
python py_pipeline/tune_models.py
```

Results are saved to `submits_data/{log_num}/tuning_log.csv`.

### Optuna Hyperparameter Tuning

Use `py_pipeline/optune_params.py` for Bayesian optimization of model hyperparameters:

1. **Edit configuration** in `optune_params.py`:
   - Set `EXP_PARAMS` (model_name, labs, feats_config)
   - Adjust `OPTUNA_CFG` (n_trials, target_metric, etc.)
   - Modify `suggest_booster_params()` to define the search space

2. **Run the script**:
```bash
python py_pipeline/optune_params.py
```

Optuna will optimize hyperparameters and save results to the specified log directory.
