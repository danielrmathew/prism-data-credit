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


def fit_bert(train_dataset, test_dataset, id2label, label2id):

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    accuracy = evaluate.load('accuracy')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # dataset, id2label, label2id = ...

    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert/distilbert-base-uncased', num_labels=9, id2label=id2label, label2id=label2id
    ).to(device)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="bert",
        learning_rate=2e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)

    train_preds = pipe(X_train)
    train_preds = [dct['label'] for dct in train_preds]

    test_preds = pipe(X_test)
    test_preds = [dct['label'] for dct in test_preds]

    y_test_cats = [id2label[cat_id] for cat_id in y_test]

    make_confusion_matrix(y_train, train_preds, model_type, train=True)
    make_confusion_matrix(y_test, test_preds, model_type, train=False)

    return pipe, {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}



def fit_fasttext(train_txt_fp, X_train, y_train, X_test, y_test):
    
    train.to_csv('train_txt_fp', index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")

    model = fasttext.train_supervised('../data/train.txt', wordNgrams = 2)

    train_preds = []
    for i in range(len(X_train)):
        train_preds.append(model.predict(X_train[i])[0][0])

    test_preds = []
    for i in range(len(X_test)):
        test_preds.append(model.predict(X_test[i])[0][0])

    train_acc = (np.array(train_preds) == np.array(y_train)).mean()
    test_acc = (np.array(test_preds) == np.array(y_test)).mean()

    make_confusion_matrix(y_train, train_preds, model_type, train=True)
    make_confusion_matrix(y_test, test_preds, model_type, train=False)
    
    return model, {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}


model_type = ...

if model_type == 'bert':
    train, test = ...
elif model_type == 'fasttext':
    train_txt_fp, X_train, y_train, X_test, y_test = ...
else:
    raise Exception('Invalid Model Type')
    
    return -1












