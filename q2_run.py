import yaml
from pathlib import Path
import pandas as pd

from src.build.q2_feature_gen import read_data, feature_gen, split_features, standardize_features
from src.build.q2_feature_selection import get_lasso_features, get_point_biserial_features
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
    acctDF, consDF, trxnDF, cat_map = read_data(ACCT_PATH, CONS_PATH, TRXN_PATH, CAT_MAP_PATH)
        
    # create balanceDF
    print("Creating balance dataframe...")
    balanceDF = calc_balances_all(acctDF, trxnDF, cat_map)

    # create features dataframe
    print("Generating features...")
    if Path('q2data/q2_features.csv').exists():
        features_df = pd.read_csv('q2data/q2_features.csv')
    else:
        features_df = feature_gen(cat_map, balanceDF, trxnDF, consDF)
        features_df.to_csv('q2data/q2_features.csv', index=False)
    print(features_df.shape)

    # train test split features
    print("Splitting features into train and test sets...")
    X_train, X_test, y_train, y_test = split_features(features_df)
    # standardize features
    print("Standardizing features...")
    X_train, X_test = standardize_features(X_train, X_test)

    # select top X features
    print("Conducting feature selection...")
    if FEATURE_SELECTION == 'lasso':
        X_train_top_features, X_test_top_features = get_lasso_features(X_train, y_train, X_test, max_features=MAX_FEATURES)
    elif FEATURE_SELECTION == 'point_biserial':
        X_train_top_features, X_test_top_features = get_point_biserial_features(X_train, X_test, max_features=MAX_FEATURES)

    model_metrics = {}
    for model_type in MODELS:
        if MODELS[model_type]:
            print(f"Training {model_type}") 
            model, metrics = train_and_evaluate(
                X_train_top_features, y_train, X_test_top_features, y_test, model_type=model_type
            )
            model_metrics[model_type] = (model, metrics)

    

    
    
    

    
