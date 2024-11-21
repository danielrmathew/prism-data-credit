from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TextClassificationPipeline
import evaluate
import torch

import fasttext
from gensim.utils import simple_preprocess
import csv

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns


######################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
######################################################################


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

















