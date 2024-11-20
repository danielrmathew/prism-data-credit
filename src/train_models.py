## imports
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TextClassificationPipeline
import evaluate
import torch

import fasttext
from gensim.utils import simple_preprocess
import csv

######################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
######################################################################

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


def fit_model(train, test, model_type):
    """
    Fits a machine learning model based on the specified type.

    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.Series): Test dataset.
        model_type (str): Type of model to fit ('log_reg', 'random_forest', 'xgboost', 'svm', 'multnb', 'bert').

    Returns:
        model: Trained model.
        le (optional): Label encoder (only returned for 'xgboost').
    """

    def output_metrics(X_train, y_train, X_test, y_test, model, model_type, le=None):
        train_preds, train_acc = predict(X_train, y_train, model, le=le)
        test_preds, test_acc = predict(X_test, y_test, model, le=le)

        make_confusion_matrix(y_train, train_preds, model_type, train=True)
        make_confusion_matrix(y_test, test_preds, model_type, train=False)
        
        return model, {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}

    # Prepping data for training:
    llms = ['bert', 'fasttext']

    if model_type == 'fasttext':
        #data prep
        pass
        
    else:
        X_train = train.drop(columns='category')
        y_train = train['category']
        X_test = test.drop(columns='category')
        y_test = test['category']
    
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
        
    elif model_type == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        accuracy = evaluate.load('accuracy')
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # label variables
        unique_categories = outflows_with_memo.category.unique()
        id2label = {i : unique_categories[i] for i in range(len(unique_categories))}
        label2id = {unique_categories[i] : i for i in range(len(unique_categories))}

        cleaned_memo_list = list(outflows_with_memo.cleaned_memo)
        categories_encoded = [label2id[cat] for cat in outflows_with_memo.category]

        X_train_tokenized = tokenizer(X_train, truncation=True, padding=True)

        class MemoDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
        
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
        
            def __len__(self):
                return len(self.labels)

        train_dataset = MemoDataset(X_train_tokenized, y_train)

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

        return pipe

    elif model_type == 'fasttext':
        ## supposed to have train UNSPLIT from y_true

        ################################ SHOULD BE IN FEATURE GEN ########################################
        #################################################################################################
        ft_data = outflows_with_memo.copy()[['prism_consumer_id', 'cleaned_memo', 'category']]
        ft_data.category = ft_data.category.apply(lambda x: '__label__' + x)
        ft_data['clean_memo_proc'] = ft_data.cleaned_memo.apply(lambda x: ' '.join(simple_preprocess(x)))

        train.to_csv('../data/train.txt', index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")

        y_train = train.category.values
        y_test = test.category.values
        #################################################################################################

        model = fasttext.train_supervised('../data/train.txt', wordNgrams = 2)

        train_preds = []
        for i in range(len(train.clean_memo_proc.values)):
            train_preds.append(model.predict(train.clean_memo_proc.values[i])[0][0])

        test_preds = []
        for i in range(len(test.clean_memo_proc.values)):
            test_preds.append(model.predict(test.clean_memo_proc.values[i])[0][0])

        train_acc = (np.array(train_preds) == np.array(y_train)).mean()
        test_acc = (np.array(test_preds) == np.array(y_test)).mean()

        make_confusion_matrix(y_train, train_preds, model_type, train=True)
        make_confusion_matrix(y_test, test_preds, model_type, train=False)
        
    return model, {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}

model_type = ...

train, test = ...

output_ = fit_model(train, test, model_type)


    
## read in model features

## depending on model, train model functions 

## predict model functions


## metrics functions



## returns model, train accuracy, test accuracy, save confusion matrix to results folder (filename: model_train/test_conf_matrix.png)