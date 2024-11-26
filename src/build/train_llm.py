from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TextClassificationPipeline
import evaluate
import torch

import fasttext
from gensim.utils import simple_preprocess
import csv

from sklearn.model_selection import train_test_split


######################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
######################################################################

def fit_bert(X_train, X_test, train_dataset, test_dataset, id2label, label2id):

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    accuracy = evaluate.load('accuracy')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert/distilbert-base-uncased', num_labels=9, id2label=id2label, label2id=label2id
    ).to(device)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments( # TODO: hyperparameter config
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

    return pipe



def fit_fasttext(train_txt_fp):
    
    train.to_csv(train_txt_fp, index=False, sep=' ', header=None, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

    model = fasttext.train_supervised('../data/train.txt', wordNgrams=2) # TODO: hyperparameter config

    # train_acc = (np.array(train_preds) == np.array(y_train)).mean()
    # test_acc = (np.array(test_preds) == np.array(y_test)).mean()

    return model













