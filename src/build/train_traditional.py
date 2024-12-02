from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def fit_model(X_train, y_train, X_test, y_test, model_type):
    """
    Fits a machine learning model based on the specified type.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): True training values
        X_test (pd.Series): Test features
        y_test (pd.Series): True test values
        model_type (str): Type of model to fit ('log_reg', 'random_forest', 'xgboost', 'svm', 'multnb').

    Returns:
        model: Trained model.

    Saves:
        {model_type}_{train/test}_cm.png: confusion matrix of model's accuracy for train and test predictions.
    """

    ###############################
    ## MODEL TRAIN BY MODEL TYPE ##
    ###############################
    
    if model_type == 'log_reg':
        model = LogisticRegression(max_iter=1000, n_jobs=-1).fit(X_train, y_train) # TODO: hyperparameter config
                
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1).fit(X_train, y_train) # TODO: hyperparameter config

    elif model_type == 'xgboost':
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.fit_transform(y_test)

        # hyperparameters to consider: n_estimators, max_depth, learning_rate
        model = XGBClassifier(objective='multi:softmax') # TODO: hyperparameter config
        model.fit(X_train, y_train_encoded)

        return model, le

    elif model_type == 'svm':
        model = LinearSVC(C=0.5, dual='auto',) # TODO: hyperparameter config
        model.fit(X_train, y_train)


    elif model_type == 'multnb':
        model = MultinomialNB(fit_prior=True, alpha=5) # TODO: hyperparameter config
        model.fit(X_train, y_train)
        
    else:
        raise Exception('Invalid Model Type')
        
    return model










