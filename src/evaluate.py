import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score

def evaluate_classifier(y_true, y_prob):
    """Evaluate binary classifier."""
    auc = roc_auc_score(y_true, y_prob)
    return {'roc_auc': round(auc, 4)}

def evaluate_regressor(y_true, y_pred):
    """Evaluate regression model."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {
        'rmse': round(rmse, 4),
        'r2':   round(r2, 4)
    }