from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    log_loss,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import numpy as np


def eval_metrics(y_true, y_pred, y_pred_proba):
    """
    Calcule les métriques pour une classification multiclasse.

    Paramètres :
    - y_true : Les vraies étiquettes (array de taille [n_samples])
    - y_pred : Les classes prédites (array de taille [n_samples])
    - y_pred_proba : Les probabilités prédites (array de taille [n_samples, n_classes])

    Retour :
    Un dictionnaire contenant les métriques :
    - accuracy
    - precision (pondérée)
    - recall (pondéré)
    - log_loss
    - roc_auc : dict des AUC par classe
    - mean_roc_auc : moyenne des AUC
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Precision et Recall pondérés (prennent en compte le déséquilibre des classes)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    # Log Loss
    logloss = log_loss(y_true, y_pred_proba)

    # ROC AUC par classe
    n_classes = y_pred_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    roc_auc = {}

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr, tpr)

    # Moyenne des AUC
    mean_roc_auc = np.mean(list(roc_auc.values()))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'log_loss': logloss,
        'roc_auc': roc_auc,
        'mean_roc_auc': mean_roc_auc
    }