############################################################
# FEATURE GENERATION AND TRAIN TEST SPLIT FOR LLM FEATURES #
############################################################

# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gensim.utils import simple_preprocess
from transformers import AutoTokenizer
from pathlib import Path
import torch
import re
import csv

## llm features (just clean memo, tagging if possible)
def encode_labels(dataset):
    """
    Encodes category labels into numerical format for modeling purposes.

    Args:
        dataset (pd.DataFrame): Input dataset containing a `category` column with categorical labels.

    Returns:
        tuple:
            pd.DataFrame: Updated dataset with an additional `encoded_category` column containing numerical labels.
            dict: id2label which maps category ids to the category labels.
            dict: label2id which maps category labels to the category ids.
    """
    unique_categories = dataset.category.unique()
    id2label = {i : unique_categories[i] for i in range(len(unique_categories))}
    label2id = {unique_categories[i] : i for i in range(len(unique_categories))}

    dataset['encoded_category'] = [label2id[cat] for cat in dataset.category]
    
    return dataset, id2label, label2id

def tokenize_for_bert(dataset, tokenizer_name="distilbert-base-uncased"):
    """
    Prepares tokenized input from the `cleaned_memo` column for BERT-based models.

    Args:
        dataset (pd.DataFrame): Input dataset containing a `cleaned_memo` column with preprocessed text.
        tokenizer_name (str, optional): The name of the BERT tokenizer to use. Defaults to "distilbert-base-uncased".

    Returns:
        dict: Tokenized data in PyTorch tensor format, including input IDs, attention masks, and other necessary components.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_data = tokenizer(
        dataset['cleaned_memo'].tolist(),
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    return tokenized_data

# torch Dataset object for bert training
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

def train_test_split_llm(dataset):
    """
    Creates a train test split of the data for LLM training

    Args:
        dataset (pd.DataFrame): Input dataset containing a `category` column with categorical labels 
            and `encoded_category` column with the categories encoded.

    Returns:
        tuple: train test split of the pre-tokenized data
        tuple:
            torch.Dataset of tokenized train dataset
            torch.Dataset of tokenized test dataset
    """
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.drop(columns='encoded_category'), dataset['encoded_category'].tolist(), test_size=0.25
    )
    
    X_train_tokenized = tokenize_for_bert(X_train)
    X_test_tokenized = tokenize_for_bert(X_test)
    
    train_dataset = MemoDataset(X_train_tokenized, y_train)
    test_dataset = MemoDataset(X_test_tokenized, y_test)

    return (X_train, X_test, y_train, y_test), (train_dataset, test_dataset)
    
def prepare_fasttext_data(dataset):
    """
    Prepares text data in FastText-compatible format for supervised text classification.

    Args:
        dataset (pd.DataFrame): Input dataset with `category` and `cleaned_memo` columns.

    Returns:
        str: Path to the saved FastText-compatible text file for train and test
             hard-coded to 'train.txt' and 'test.txt'
    """
    ##################################
    train_fp = 'data/train.txt'
    test_fp = 'data/test.txt'
    ##################################
    
    ft_data = dataset.copy()
    ft_data['category'] = ft_data['category'].apply(lambda x: '__label__' + x)
    ft_data['cleaned_memo_proc'] = ft_data.cleaned_memo.apply(lambda x: ' '.join(simple_preprocess(x)))

    ids = ft_data.prism_consumer_id.unique() ## CHECK IF THIS IS POSSIBLE FOR FT_DATA BASED ON DATASET.COPY()
    test_ratio = 0.25
    test_size = int(len(ids) * 0.25)
    
    test_ids = np.random.choice(ids, size=test_size, replace=False)
    
    train = ft_data[~ft_data.prism_consumer_id.isin(test_ids)].get(['cleaned_memo_proc', 'category'])
    test = ft_data[ft_data.prism_consumer_id.isin(test_ids)].get(['cleaned_memo_proc', 'category'])
    
    train.to_csv(Path(train_fp), index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")
    test.to_csv(Path(test_fp), index = False, sep = ' ', header = None, quoting = csv.QUOTE_NONE, quotechar = "", escapechar = " ")

    return train_fp, test_fp
