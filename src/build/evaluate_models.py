import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# TODO: in output_metrics or in separate functions, add more metrics like roc-auc, classification_report stuff

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
    plt.savefig(f'{model_type}_{title}_confusion_matrix.png',)

    plt.show()

def output_metrics(y, preds, model_type, train=True):
    # TODO: add other metrics here or in other functions
    make_confusion_matrix(y, preds, model_type, train=train)
    
    return {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}
    