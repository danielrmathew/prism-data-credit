### imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import re
import csv

## read data generate clean memos
def read_outflows(path)
    """
    Reads the outflows dataset from a given path and filters out rows where the `memo` column matches the `category` column. 
    Resets the index and removes the redundant `index` column.

    Args:
        path (str): File path to the Parquet file containing the outflows dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered outflows data, excluding rows where `memo` equals `category`.
    """
    outflows = pd.read_parquet(path)
    outflows_with_memo = outflows[~(outflows.memo == outflows.category)].reset_index(drop=True)
    return outflows_with_memo

def clean_memos(dataset):
    """
    Cleans the `memo` column in the dataset by removing patterns such as masked characters, dates, and unnecessary special characters. 
    Adds a new column `cleaned_memo` to the dataset with the processed text.

    Args:
        dataset (pd.DataFrame): Input dataset containing a `memo` column to be cleaned.

    Returns:
        pd.DataFrame: The original dataset with an additional `cleaned_memo` column containing the cleaned text.
    """
    memos = dataset.memo
    pattern1 = r'\b\w*x{2,}\w*\b'
    pattern2 = r'\b(0[1-9]|1[0-2])(\/|-)[0-9]{2}\b'
    pattern3 = r"[,'*#_-]"

    cleaned_memos = (
        memos.str.lower()
            .str.replace(pattern1, '', regex=True)                   # removing XXXX                        
            .str.replace(pattern2, '', regex=True)                   # removing dates
            .str.replace(pattern3, '', regex=True)                   # removing unnecessary special characters
            .str.replace(r'~', '', regex=True)                       # removing ~ (can't include in character class above)
            .str.replace('purchase.* authorized on', '', regex=True) # removing common phrase
            .str.replace('checkcard', '')                            # removing common phrase
            .str.strip()                                             # removing leading and trailing characters
    )

    cleaned_memos = cleaned_memos.reset_index().drop(columns='index').squeeze()
    dataset['cleaned_memo'] = cleaned_memos

    return dataset


## non-llm features
    
def get_tfidf_features(dataset, max_features):
    """
    Generates a dataset of TF-IDF (Term Frequency-Inverse Document Frequency) features from the `cleaned_memo` column 
    of the input dataset. The number of TF-IDF features is determined by the `max_features` parameter.

    Args:
        dataset (pd.DataFrame): Input dataset containing a `cleaned_memo` column with preprocessed text.
        max_features (int): Maximum number of TF-IDF features to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the TF-IDF features as columns, with each column name prefixed by 'tfidf_' 
                      followed by the corresponding term.
    """
    cleaned_memos = dataset.cleaned_memo
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf = vectorizer.fit_transform(cleaned_memos)
    
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf)
    tfidf_df.columns = 'tfidf_' + vectorizer.get_feature_names_out()
    
    return tfidf_df

def get_date_features(dataset):
    """
    Creates one-hot encoded features from the `posted_date` column, capturing temporal spending patterns.

    Features:
        - month: Encodes the month (e.g., 'month_1' for January).
        - weekday: Encodes the day of the week (e.g., 'weekday_0' for Monday).
        - day_of_month: Encodes the day of the month (e.g., 'day_of_month_1').

    Args:
        dataset (pd.DataFrame): Dataset with a `posted_date` column in datetime format.

    Returns:
        pd.DataFrame: DataFrame of one-hot encoded date features.
    """
    posted_date = dataset.posted_date
    date_features = pd.DataFrame()
    
    date_features['month'] = 'month_' + posted_date.dt.month.astype(str)
    date_features['weekday'] = 'weekday_' + posted_date.dt.weekday.astype(str)
    date_features['day_of_month'] = 'day_of_month_' + posted_date.dt.day.astype(str)

    date_enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    date_one_hot = date_enc.fit_transform(date_features)

    date_columns = np.concatenate([feature[1:] for feature in date_enc.categories_])
    date_one_hot_df = pd.DataFrame.sparse.from_spmatrix(date_one_hot, columns=date_columns)

    return date_one_hot_df

def get_amount_features(dataset):
    """
    Creates one-hot encoded features from the `amount` column, capturing patterns in transaction amounts.

    Features:
        - is_even: Indicates if the amount ends in an even value (e.g., x.00 as 'amount_even').
        - decile_amounts: Bins transaction amounts into deciles (e.g., 'decile_0' for lowest 10%).

    Args:
        dataset (pd.DataFrame): Dataset with an `amount` column.

    Returns:
        pd.DataFrame: DataFrame of one-hot encoded amount features.
    """
    amount = dataset.amount

    amount_features = pd.DataFrame()
    amount_features['is_even'] = amount.apply(lambda x: 'amount_even' if x % 1 == 0 else 'amount_odd')
    amount_features['decile_amounts'] = pd.qcut(amount, q=10, labels=['decile_0', 'decile_1', 'decile_2', 'decile_3', 'decile_4', 'decile_5', 'decile_6', 'decile_7', 'decile_8', 'decile_9'])

    amount_enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    amount_one_hot = amount_enc.fit_transform(amount_features)

    amount_columns = np.concatenate([feature[1:] for feature in amount_enc.categories_])
    amount_one_hot_df = pd.DataFrame.sparse.from_spmatrix(amount_one_hot, columns=amount_columns)
    
    return amount_one_hot_df


## llm features
    # just clean memo, tagging if possible
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
    le = LabelEncoder()
    dataset['encoded_category'] = le.fit_transform(dataset['category'])
    return dataset, le

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

## returns model features
def get_features_df(dataset, tfidf_max_features):
    """
    Generates a combined feature set for the dataset.

    Includes:
        - TF-IDF features from `cleaned_memo`
        - Date features from `posted_date`
        - Amount features from `amount`

    Args:
        dataset (pd.DataFrame): Input dataset with transaction data.
        tfidf_max_features (int): Maximum number of TF-IDF features.

    Returns:
        pd.DataFrame: Dataset with added features.
    """
    tfidf_df = get_tfidf_features(dataset, tfidf_max_features)
    date_df = get_date_features(dataset)
    amount_df = get_amount_features(dataset)

    features_df =  pd.concat([dataset, tfidf_df, date_df, amount_df], axis=1)
    return features_df

def train_test_split_features(features_df):
    """
    Splits the feature dataset into training and testing sets.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and target variable.

    Returns:
        tuple: X_train (features for training), y_train (labels for training),
               X_test (features for testing), y_test (labels for testing).
    """
    train_df, test_df = dataset_split(features_df)
    
    X_train = train_df.iloc[:, 8:]
    y_train = train_df['category']
    X_test = test_df.iloc[:, 8:]
    y_test = test_df['category']

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    return X_train, y_train, X_test, y_test
    