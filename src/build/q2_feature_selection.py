import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def get_lasso_features(X_train, y_train, X_test, max_features=50):
    model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1) # can test different Cs
    model_l1.fit(X_train, y_train)

    # get top 50 features after lasso regularization and make new train and test set
    selector = SelectFromModel(model_l1, prefit=True, threshold=-np.inf, max_features=max_features)
    top_50_features_l1 = list(X_train.columns[selector.get_support()])
    X_train_l1 = X_train[top_50_features_l1]
    X_test_l1 = X_test[top_50_features_l1]

    return X_train_l1, X_test_l1

def get_point_biserial_features(X_train, X_test, max_features=50):
    # use features with highest point biserial correlation
    point_biserial_features = [(feature, pointbiserialr(X_train[feature], y_train).statistic) for feature in X_train.columns]
    point_biserial_features.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features_pb = [feature[0] for feature in point_biserial_features[:max_features]]

    # grab top 50 features by point biseral corr
    X_train_pb = X_train[top_features_pb]
    X_test_pb = X_test[top_features_pb]

    return X_train_pb, X_test_pb