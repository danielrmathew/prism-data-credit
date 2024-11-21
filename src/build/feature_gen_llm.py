############################################################
# FEATURE GENERATION AND TRAIN TEST SPLIT FOR LLM FEATURES #
############################################################

# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
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
            LabelEncoder: Fitted label encoder for decoding or mapping future labels.
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
    """
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.drop(columns='encoded_category'), dataset['encoded_category'].tolist(), test_size=0.25
    )
    
    X_train_tokenized = tokenize_for_bert(X_train, truncation=True, padding=True)
    X_test_tokenized = tokenize_for_bert(X_test, truncation=True, padding=True)
    
    train_dataset = MemoDataset(X_train_tokenized, y_train)
    test_dataset = MemoDataset(X_test_tokenized, y_test)

    return train_dataset, test_dataset
    
def prepare_fasttext_data(dataset, output_path):
    """
    Prepares text data in FastText-compatible format for supervised text classification.

    Args:
        dataset (pd.DataFrame): Input dataset with `category` and `cleaned_memo` columns.
        output_path (str): Path to save the formatted FastText data file.

    Returns:
        str: Path to the saved FastText-compatible text file.
    """
    ft_data = dataset.copy()
    ft_data['category'] = ft_data['category'].apply(lambda x: '__label__' + x)
    ft_data['cleaned_memo_proc'] = ft_data['cleaned_memo']
    
    ft_data[['category', 'cleaned_memo_proc']].to_csv(
        output_path, index=False, sep=' ', header=None, 
        quoting=csv.QUOTE_NONE, quotechar="", escapechar=" "
    )
    return output_path
