from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def predict(X, y, model, le=None):
    """
    Makes predictions using the trained model and evaluates accuracy.

    Args:
        X (pd.DataFrame): Features for prediction.
        y (pd.Series): True labels to compare predictions.
        model: Trained model (LogisticRegression, RandomForestClassifier, or XGBClassifier).
        le (LabelEncoder, optional): Label encoder, required for 'xgboost' model.

    Returns:
        preds (numpy.ndarray): Predicted labels.
    """
    if isinstance(model, LogisticRegression) or isinstance(model, RandomForestClassifier):
        preds = model.predict(X)
    elif isinstance(model, XGBClassifier):
        preds_encoded = model.predict(X)
        preds = le.inverse_transform(preds_encoded)
        
    return preds, (preds == y).mean():.4f


def make_confusion_matrix(y, preds, model_type, train=True):
    """
    Generates and displays a confusion matrix with precision scores.

    Args:
        y (pd.Series): True labels.
        preds (numpy.ndarray): Predicted labels.

    Returns:
        None: Displays a heatmap of the confusion matrix.
    """
    conf_matrix = confusion_matrix(y, preds, labels=y.unique(), normalize='pred')
    classes = y.value_counts().sort_values(ascending=False).index
    conf_matrix_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    
    sns.heatmap(conf_matrix_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, 
                linewidths=0.5, linecolor='gray', square=True, annot_kws={"size": 8}, ax=ax1)
    ax1.set_title('Precision')
    ax1.set_xlabel('Predicted Labels')
    ax1.set_ylabel('True Labels')

    title = 'train' if train else 'test'
    plt.savefig(f'{model_type}_{title}_cm.png',)

    plt.show()


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
        acc_dictionary: Training / Testing accuracy of model

    Saves:
        {model_type}_{train/test}_cm.png: confusion matrix of model's accuracy for train and test predictions.
    """

    def output_metrics(X_train, y_train, X_test, y_test, model, model_type, le=None):
        train_preds, train_acc = predict(X_train, y_train, model, le=le)
        test_preds, test_acc = predict(X_test, y_test, model, le=le)

        make_confusion_matrix(y_train, train_preds, model_type, train=True)
        make_confusion_matrix(y_test, test_preds, model_type, train=False)
        
        return model, {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}

    ###############################
    ## MODEL TRAIN BY MODEL TYPE ##
    ###############################
    
    if model_type == 'log_reg':
        model = LogisticRegression(max_iter=1000, n_jobs=-1).fit(X_train, y_train)
        
        return output_metrics(X_train, y_train, X_test, y_test, model, model_type,)
        
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1).fit(X_train, y_train)

        return output_metrics(X_train, y_train, X_test, y_test, model, model_type,)
        
    elif model_type == 'xgboost':
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.fit_transform(y_test)

        # hyperparameters to consider: n_estimators, max_depth, learning_rate
        model = XGBClassifier(objective='multi:softmax')
        model.fit(X_train, y_train_encoded)

        return output_metrics(X_train, y_train_encoded, X_test, y_test_encoded, model, model_type, le=le)

    elif model_type == 'svm':
        model = LinearSVC(C=0.5, dual='auto',)
        model.fit(X_train, y_train)

        return output_metrics(X_train, y_train, X_test, y_test, model, model_type,)

    elif model_type == 'multnb':
        model = MultinomialNB(fit_prior=True, alpha=5)
        model.fit(X_train, y_train)

        return output_metrics(X_train, y_train, X_test, y_test, model, model_type, le=None)
        
    else:
        raise Exception('Invalid Model Type')
        
    return -1
 

model_type = ...

X_train, y_train, X_test, y_test = ...

output_ = fit_model(X_train, y_train, X_test, y_test, model_type)










