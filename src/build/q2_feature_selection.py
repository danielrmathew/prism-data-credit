import pandas as pd
import numpy as np
import re
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def get_lasso_features(X_train, y_train):
    """
    Returns the feature coefficients of the datasets after training a Logistic Regression model with L1 lasso regularization
    Args:
        X_train (pd.DataFrame): the features dataframe 
        y_train (pd.Series): the dataset labels
    Returns:
        list[(str, float)]: a list of tuples (feature, coefficient)
    """
    model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1) # can test different Cs
    model_l1.fit(X_train, y_train)

    feature_coefs = list(zip(model_l1.feature_names_in_, model_l1.coef_[0]))
    feature_coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    return feature_coefs

def extract_category(feature_name):
    """
    Extracts the category from a feature name using regex.
    Args:
        feature_name: category label
    Returns:
        str: the category in the feature name
    """
    match = re.match(r'^([A-Z]+(?:_[A-Z]+)*)', feature_name)
    return match.group(1) if match else feature_name 

def select_top_features(feature_coefs, max_features, limit=2):
    """
    Gets the top features in the dataset, ranked by the feature coefficient
    Args:
        feature_coefs (list[str, float]): the feature coefficients in sorted order
        max_features: the number of features to select
        limit (int): the max number of features to grab from a single category, default 2
    Returns:
        list: the final selected features 
    """
    category_dict = defaultdict(list)

    # Organize features by category
    for feature, coef in feature_coefs:
        category = extract_category(feature)  # Extract category using regex
        category_dict[category].append((feature, coef))

    # Sort each category by absolute coefficient value (descending)
    for category in category_dict:
        category_dict[category].sort(key=lambda item: abs(item[1]), reverse=True)

    # Select top max_features, allowing up to 2 features per category
    selected_features = []
    category_counts = defaultdict(int)  # Track how many features have been selected per category

    # Flatten sorted features by absolute importance while respecting category limits
    sorted_features = sorted(
        [feat for feats in category_dict.values() for feat in feats], 
        key=lambda item: abs(item[1]), reverse=True
    )

    for feature, coef in sorted_features:
        category = extract_category(feature)
        if len(selected_features) < max_features and category_counts[category] < limit:
            selected_features.append((feature, coef))
            category_counts[category] += 1

    return selected_features

def get_feature_selection_datasets(X, selected_features):
    """
    Filters the input dataframe to the selected features.
    Args:
        X (pd.DataFrame): features dataframe
        selected_features (list): the selected features
    Return:
        pd.DataFrame: filtered dataframe
    """
    return X[np.array(selected_features)[:, 0]]
    
def get_point_biserial_features(X_train, X_test, max_features=50):
    """
    Selects the top features based on their point biserial correlation with the target variable.
    Args:
        X_train (pd.DataFrame): The training set features.
        X_test (pd.DataFrame): The testing set features.
        max_features (int): The maximum number of top features to select based on point biserial correlation (default is 50).
    Returns:
        tuple: A tuple containing the following:
            - X_train_pb (pd.DataFrame): The training set with top features selected by point biserial correlation.
            - X_test_pb (pd.DataFrame): The testing set with top features selected by point biserial correlation.
    """
    # use features with highest point biserial correlation
    point_biserial_features = [(feature, pointbiserialr(X_train[feature], y_train).statistic) for feature in X_train.columns]
    point_biserial_features.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features_pb = [feature[0] for feature in point_biserial_features[:max_features]]

    # grab top 50 features by point biseral corr
    X_train_pb = X_train[top_features_pb]
    X_test_pb = X_test[top_features_pb]

    return X_train_pb, X_test_pb
