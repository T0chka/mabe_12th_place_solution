"""
Calibration functions.
"""
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize

def clip(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Map values to the closed interval [eps, 1 - eps]."""
    return np.clip(scores, eps, 1 - eps)

def fit_platt(logits: np.ndarray, labels: np.ndarray):
    """Fit Platt scaling on raw margins (logits)."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    
    # Handle degenerate case (all labels are the same)
    if labels.min() == labels.max():
        pi = float(labels.mean())
        return 1.0, 0.0, np.float64(pi)

    # Fit logistic regression: logit(p_calibrated) = a * logits + b
    # This matches R: glm.fit(cbind(1, logits), labels, family=binomial())
    def neg_log_likelihood(params):
        b, a = params  # intercept, slope
        z = b + a * logits
        p = clip(expit(z))
        return -np.sum(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    
    result = minimize(neg_log_likelihood, [0.0, 1.0], method='BFGS')
    b, a = result.x
    
    pi = float(labels.mean())
    
    return a, b, np.float64(pi)

def apply_platt(logits: np.ndarray, a: float, b: float, pi_train: float):
    """Apply Platt calibration and return logit-scale scores."""
    logits = np.asarray(logits, dtype=np.float64)
    
    # Apply calibration: calibrated_logits = a * logits + b
    calibrated_logits = a * logits + b
    
    # Convert to probabilities
    p = clip(expit(calibrated_logits))

    # Prior correction
    pi_train = clip(np.array(pi_train))
    logit_pi = logit(pi_train)

    out = logit(p) - logit_pi
    return out.astype(np.float32)