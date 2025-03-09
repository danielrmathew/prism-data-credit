import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier 

import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(X_train, y_train, X_test, y_test, model_type):
    """
    Trains and evaluates a specified classification model.

    Args:
        X_train (pd.DataFrame): The feature matrix for the training set.
        y_train (pd.Series): The target variable for the training set.
        X_test (pd.DataFrame): The feature matrix for the testing set.
        y_test (pd.Series): The target variable for the testing set.
        model_type (str): The type of model to train and evaluate. Supported options are:
                          "HistGB", "CatBoost", "LightGBM", "XGBoost", "LogisticRegression".
    Returns:
        tuple: A tuple containing:
            - model: The trained model.
            - metrics (dict): A dictionary of evaluation metrics including:
                - ROC_AUC
                - Accuracy
                - Precision
                - Recall
                - F1-Score
                - Confusion Matrix
    Raises:
        ValueError: If an unsupported model type is provided.
    """

    def get_metrics(X, y, train=True):
        # Predict on training data
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
        # Evaluation Metrics
        metrics = {
            "ROC_AUC": roc_auc_score(y, y_prob),
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1-Score": f1_score(y, y_pred),
            # "Confusion Matrix": confusion_matrix(y, y_pred)
        }

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix ({model_type})")
        cm_fp = Path(f"q2_result/{model_type}/confusion_matrix_{'train' if train else 'test'}")
        cm_fp.parent.mkdir(parents=True, exist_ok=True)  
        plt.savefig(cm_fp, bbox_inches="tight")
    
        # Print Metrics
        print(f'{model_type} metrics:')
        for metric, value in metrics.items():
            print(f"\t{metric}: {value}")

        return metrics
    
    models = {
        "HistGB": HistGradientBoostingClassifier(),
        "CatBoost": CatBoostClassifier(random_seed=42, verbose=0),
        "LightGBM": LGBMClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1)
    }
    
    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from {list(models.keys())}")
    
    model = models[model_type]

    # Train model and track training time
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    
    print(f"Training Time: {end - start:.4f} seconds")

    train_metrics = get_metrics(X_train, y_train)
    test_metrics = get_metrics(X_test, y_test)

    return model, test_metrics
    