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

def fit_bert(train_dataset, test_dataset, id2label, label2id, hp):

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

    training_args = TrainingArguments( 
        output_dir="result/models/bert",
        learning_rate=hp['learning_rate'],
        per_device_train_batch_size=hp['batch_size'],
        per_device_eval_batch_size=hp['batch_size'],
        num_train_epochs=hp['num_epochs'],
        weight_decay=hp['weight_decay'],
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


def fit_fasttext(train_fp, ngrams=2):
    model = fasttext.train_supervised(train_fp, wordNgrams=ngrams)

    return model
    