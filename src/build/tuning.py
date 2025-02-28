# necessary imports
from collections import defaultdict
from functools import reduce
import json
from pathlib import Path
import pickle
import re
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from scipy.stats import pointbiserialr
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from catboost import CatBoostClassifier # pip install
from lightgbm import LGBMClassifier # pip install
from xgboost import XGBClassifier 
import optuna

from q2_feature_gen import standardize_features

def objective(trial, X_train, y_train, X_val, y_val):
    model_name = trial.suggest_categorical("model", ["HistGB", "LightGBM", "XGBoost", "CatBoost"])
    print(f"Trial {trial.number}: model {model_name}")
    logging.info(f"Trial {trial.number}: model {model_name}")
    
    if model_name == "HistGB":
        model = HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
            l2_regularization=trial.suggest_float("l2_regularization", 0.001, 5.0, log=True),
            early_stopping=True
        )
    elif model_name == "CatBoost":
        model = CatBoostClassifier(
            learning_rate=trial.suggest_float("lr", 0.001, 0.3, log=True),
            depth=trial.suggest_int("depth", 3, 12),
            iterations=trial.suggest_int("iterations", 100, 2000),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
            random_strength=trial.suggest_float("random_strength", 0, 10),
            verbose=0
        )
    elif model_name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 2000),
            learning_rate=trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            num_leaves=trial.suggest_int("num_leaves", 8, 256),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            # reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True), for feature selection 
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True), # for overfitting
            # colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # subsample=trial.suggest_float("subsample", 0.5, 1.0),
            # min_split_gain=trial.suggest_float("min_split_gain", 0, 0.1),
            # boosting_type=trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
            verbose=-1
        )
    elif model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 2000),
            learning_rate=trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 10, log=True),
            gamma=trial.suggest_float("gamma", 0, 10),
            # reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            eval_metric='logloss'
        )
    else:
        model = LogisticRegression(
            C=trial.suggest_float("C", 0.01, 10, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 500)
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_prob)
    
    # cv = StratifiedKFold(n_splits=5, shuffle=True)
    # score = np.mean(cross_val_score(model, X, y, cv=cv, scoring='roc_auc'))
    
    print(f"Trial {trial.number} - Model: {model_name}, Score: {score:.4f}")
    
    return score

def run_optimization(X_train, y_train, X_val, y_val, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
    
    print("Best parameters:", study.best_params)
    return study

def split_data(final_features_df, test_size=0.25):
    # currently using train test split -- need to start using train val test split
    X_train, X_test, y_train, y_test = train_test_split(
        final_features_df.drop(columns=['prism_consumer_id', 'DQ_TARGET', 'evaluation_date', 'credit_score']), 
        final_features_df['DQ_TARGET'], test_size=test_size, stratify=final_features_df['DQ_TARGET']
    )
    return X_train, X_test, y_train, y_test

def standardize(X_train, *args):
    # instantiate StandardScaler() to standardize features, excluding binary features
    scaler = StandardScaler()
    exclude_columns_standardize = ['GAMBLING', 'BNPL', 'OVERDRAFT'] # binary features that shouldn't be standardized
    standardize_features = X_train.columns.difference(exclude_columns_standardize)
    
    transformer = ColumnTransformer([
        ('std_scaler', StandardScaler(), standardize_features)  # Standardize all except excluded ones
    ], remainder='passthrough')
    
    X_train_standardized = transformer.fit_transform(X_train)
    X_train_standardized = pd.DataFrame(X_train_standardized, columns=list(standardize_features) + exclude_columns_standardize)

    X_datasets = [X_train_standardized]
    
    # standardize test features
    for X_dataset in args:
        X_test_standardized = transformer.transform(X_dataset)
        X_test_standardized = pd.DataFrame(X_test_standardized, columns=list(standardize_features) + exclude_columns_standardize)
        X_datasets.append(X_test_standardized)

    return X_datasets

print("Loading features df...")

with open("features_df.pkl", "rb") as f:
    features_df = pickle.load(f)

print("Standardizing features...") 

X_train, X_test, y_train, y_test = split_data(features_df, test_size=0.5)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

# when dfs are standardized, object columns (mainly boolean columns of binary variables) get converted to floats
X_train_standardized, X_val_standardized, X_test_standardized = standardize(X_train, X_val, X_test)

print("Resampling data points...")

# Check class counts
new_0 = int(0.75 * len(y_train))  # Target count for balancing
new_1 = int(0.25 * len(y_train))

# Ensure SMOTE does not create more than the original number of samples
smote = SMOTE(sampling_strategy={1: new_1},)

# Undersampling the majority class to match the new minority class count
under = RandomUnderSampler(sampling_strategy={0: new_0},)

# Combine SMOTE & Undersampling in a pipeline
resample_pipeline = Pipeline(steps=[('smote', smote), ('under', under)])

# Apply resampling
X_train_resampled, y_train_resampled = resample_pipeline.fit_resample(X_train_standardized, y_train)

# columns get shuffled after SMOTE, need to reorder them
X_train_resampled = X_train_resampled[X_train_standardized.columns]

print("Beginning tuning...")

study = run_optimization(X_train_resampled, y_train_resampled, X_val_standardized, y_val)

df = study.trials_dataframe()
df.to_csv("optuna_trials.csv", index=False)  # Save as CSV

best_params = {
    "best_trial": study.best_trial.number,
    "best_value": study.best_value,
    "best_params": study.best_params
}

with open("optuna_best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("Best parameters saved!")