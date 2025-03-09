import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from src.build.q2_feature_gen import read_data, create_features_df, split_data, standardize, resample_data
from src.build.q2_feature_selection import get_lasso_features, select_top_features, get_feature_selection_datasets, get_point_biserial_features
from src.build.q2_gen_balanceDF import calc_balances_all
from src.build.q2_train_and_evaluate import train_and_evaluate

if __name__ == "__main__":
    with open("q2_config.yml", "r") as f:
        config = yaml.safe_load(f)

    # LOAD CONFIG VARIABLES
    ACCT_PATH = config['ACCOUNT_DF_PATH']
    CONS_PATH = config['CONSUMER_DF_PATH']
    TRXN_PATH = config['TRANSACTION_DF_PATH']
    CAT_MAP_PATH = config['CATEGORY_MAP_PATH']
    
    FEATURE_SELECTION = config['FEATURE_SELECTION']
    MAX_FEATURES = config['MAX_FEATURES']

    MODELS = config["MODELS"]

    print("Reading data...")
    acctDF, consDF, trxnDF = read_data(ACCT_PATH, CONS_PATH, TRXN_PATH, CAT_MAP_PATH)\

    # create features dataframe
    print("Generating features...")
    features_fp = Path('q2data/q2_features.csv')
    features_fp.parent.mkdir(parents=True, exist_ok=True)  
    if features_fp.exists():
        print("Found existing features file, reading data...")
        features_df = pd.read_csv(features_fp)
    else:
        features_df = create_features_df(consDF, acctDF, trxnDF)
        features_df.to_csv(features_fp, index=False)

    features_df = features_df[~features_df.DQ_TARGET.isna()]
    
    # train test split features
    print("Splitting features into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(features_df)
    # standardize features
    print("Standardizing features...")
    X_train, X_test = standardize(X_train, X_test)

    # select top X features
    print("Conducting feature selection...")
    if FEATURE_SELECTION == 'lasso':
        # X_train_top_features, X_test_top_features = get_lasso_features(X_train, y_train, X_test, max_features=MAX_FEATURES)
        feature_coefs = get_lasso_features(X_train, y_train)
        selected_features = select_top_features(feature_coefs, MAX_FEATURES, limit=np.inf)
        X_train_top, X_test_top = get_feature_selection_datasets(X_train, selected_features), \
                                  get_feature_selection_datasets(X_test, selected_features)
    elif FEATURE_SELECTION == 'point_biserial':
        X_train_top, X_test_top = get_point_biserial_features(X_train, X_test, max_features=MAX_FEATURES)

    # resample features
    print("Resampling data...")
    X_train_resampled, y_train_resampled = resample_data(X_train_top, y_train)

    # train and evaluate models
    model_metrics = {}
    for model_type in MODELS:
        if MODELS[model_type]:
            print(f"Training {model_type}") 
            model, metrics = train_and_evaluate(
                X_train_resampled, y_train_resampled, X_test_top, y_test, model_type=model_type
            )
            model_path = Path(f"q2_result/{model_type}/{model_type}.pkl")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)                                                             
            model_metrics[model_type] = (model, metrics)

    

    
    
    

    
