import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from functools import reduce
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, log_loss
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def read_data(acct_path, cons_path, trxn_path, cat_map_path):
    """
    Reads in data from specified file paths (account, consumer, transaction, and category map files),
    processes necessary columns (such as converting dates and ensuring correct data types), and returns 
    the cleaned DataFrames.

    Args:
        acct_path (str): Path to the account data file.
        cons_path (str): Path to the consumer data file.
        trxn_path (str): Path to the transaction data file.
        cat_map_path (str): Path to the category mapping data file.

    Returns:
        - acctDF (pd.DataFrame): processed account data
        - consDF (pd.DataFrame): processed consumer data
        - trxnDF (pd.DataFrame): Processed transaction data
        - cat_map (pd.DataFrame): category mapping data
    """
    # reading in data
    acctDF = pd.read_parquet(Path(acct_path))
    consDF = pd.read_parquet(Path(cons_path))
    trxnDF = pd.read_parquet(Path(trxn_path))
    cat_map = pd.read_csv(Path(cat_map_path))
    
    # changing column types for processing
    acctDF.balance_date = pd.to_datetime(acctDF.balance_date)
    acctDF = acctDF.astype({'prism_consumer_id': int, 'prism_account_id': int})
    
    consDF.evaluation_date = pd.to_datetime(consDF.evaluation_date)
    consDF = consDF.astype({'prism_consumer_id': int})
    
    trxnDF.posted_date = pd.to_datetime(trxnDF.posted_date)
    trxnDF = trxnDF.astype({'prism_consumer_id': int, 'prism_transaction_id': int})

    return acctDF, consDF, trxnDF, cat_map
    
def feature_gen(cat_map, balanceDF, trxnDF, consDF):
    """
    Generates features for transactions by mapping categories to a more human-readable format and filtering
    transaction categories based on the list of acceptable categories from 'cat_map'.

    Args:
        cat_map (pd.DataFrame): DataFrame containing a mapping of category IDs to category names.
        balanceDF (pd.DataFrame): DataFrame containing balance information for consumers.
        trxnDF (pd.DataFrame): DataFrame containing transaction data, including the 'category' field to be processed.
        consDF (pd.DataFrame): DataFrame containing consumer-related information.

    Returns:
        None
    """
    cat_mappings = dict(zip(cat_map['category_id'], cat_map['category']))
    trxnDF['category_id'] = trxnDF.category
    trxnDF.category = trxnDF.category.replace(cat_mappings)

    cat_ok = ['SELF_TRANSFER', 'EXTERNAL_TRANSFER', 'DEPOSIT', 'PAYCHECK',
       'MISCELLANEOUS', 'PAYCHECK_PLACEHOLDER', 'REFUND',
       'INVESTMENT_INCOME', 'OTHER_BENEFITS', 
       'SMALL_DOLLAR_ADVANCE', 'TAX', 'LOAN', 'INSURANCE',
       'FOOD_AND_BEVERAGES', 'UNCATEGORIZED', 'GENERAL_MERCHANDISE',
       'AUTOMOTIVE', 'GROCERIES', 'ATM_CASH', 'ENTERTAINMENT', 'TRAVEL',
       'ESSENTIAL_SERVICES', 'ACCOUNT_FEES', 'HOME_IMPROVEMENT',
       'OVERDRAFT', 'CREDIT_CARD_PAYMENT', 'HEALTHCARE_MEDICAL', 'PETS',
       'EDUCATION', 'GIFTS_DONATIONS', 'BILLS_UTILITIES', 'MORTGAGE',
       'RENT', 'BNPL', 'AUTO_LOAN',
       'BANKING_CATCH_ALL', 'DEBT', 'FITNESS', 'TRANSPORATION', 'LEGAL',
       'GOVERNMENT_SERVICES', 'RISK_CATCH_ALL', 'RTO_LTO', 'INVESTMENT',
       'GAMBLING', 'CORPORATE_PAYMENTS', 'TIME_OR_STUFF', 'PENSION']

    def filter_time_window(df, days=None, months=None, years=None):
        """
        Filters transactions for each consumer over some time period
        Args:
            df (pd.DataFrame): dataframe to window over 
            days (int): number of days to go back
            months (int): number of months to go back
            years (int): number of years to go back
        Returns:
            pd.DataFrame: windowed dataframe
        """
        def filter_group(group):
            latest_date = group['posted_date'].max()  # Get latest transaction date per consumer
            cutoff_date = latest_date - pd.DateOffset(days=days or 0, months=months or 0, years=years or 0)
            return group[group['posted_date'] >= cutoff_date]  # Filter transactions
    
        return df.groupby('prism_consumer_id', group_keys=False).apply(filter_group)

    def compute_balance_delta(balanceDF, balance_filtered, label):
        """
        Computes the change in balance between the last available balance and the earliest balance 
        within a specified filtered dataset for each consumer.

        Args:
            balanceDF (pd.DataFrame): DataFrame containing balance information for consumers.
            balance_filtered (pd.DataFrame): A dataframe containing the filtered balance data.
            label (str): A label to be used in the column name of the first balance in the filtered dataset.

        Returns:
            pd.DataFrame: dataframe with the 'prism_consumer_id' and the computed balance delta.
        """
        # Get last balance per consumer
        last_balance = balanceDF.groupby('prism_consumer_id')['curr_balance'].last().reset_index()
        last_balance = last_balance.rename(columns={'curr_balance': 'curr_balance_last'})
        
        # Get earliest balance in the filtered dataset
        first_balance = balance_filtered.groupby('prism_consumer_id')['curr_balance'].first().reset_index()
        first_balance = first_balance.rename(columns={'curr_balance': f'curr_balance_{label}'})
        
        # Merge and compute balance delta
        df_merged = last_balance.merge(first_balance, on='prism_consumer_id', how='left')
        df_merged[f'balance_delta_{label}'] = df_merged['curr_balance_last'] - df_merged[f'curr_balance_{label}']
        
        # Keep only relevant columns
        return df_merged[['prism_consumer_id', f'balance_delta_{label}']]


    def generate_category_features(trxnDF, cat_map, categories):
        """
        Generates transaction-based features for each selected category over multiple time windows.
        
        Parameters:
            trxnDF (pd.DataFrame): DataFrame containing transaction data.
            cat_map (pd.DataFrame): DataFrame mapping category IDs to category names.
            categories (str or list): One or more transaction categories to filter.
    
        Returns:
            pd.DataFrame: Aggregated features per prism_consumer_id.
        """
        if isinstance(categories, str):
            categories = [categories]
    
        # trxnDF = trxnDF.merge(cat_map, left_on='category', right_on='category_id', how='left')
        # trxnDF['category'] = trxnDF['category_y']
        # trxnDF = trxnDF.drop(columns=['category_id', 'category_y'])
    
        trxnDF['posted_date'] = pd.to_datetime(trxnDF['posted_date'])
    
        time_windows = {
            'overall': None,
            'last_14_days': 14,
            'last_30_days': 30,
            'last_3_months': 90,
            'last_6_months': 180,
            'last_year': 365
        }
    
        features_dict = {}
    
        for category in categories:
            # filter transactions for the current category
            filtered_trxn = trxnDF[trxnDF['category'] == category].copy()
    
            # get last posted date per consumer
            last_posted_dates = filtered_trxn.groupby('prism_consumer_id')['posted_date'].max()
    
            for window_name, days in time_windows.items():
                if days is None:
                    df_time_filtered = filtered_trxn
                else:
                    # consumer-specific time filters
                    df_time_filtered = filtered_trxn.merge(last_posted_dates, on='prism_consumer_id', suffixes=('', '_latest'))
                    df_time_filtered = df_time_filtered[df_time_filtered['posted_date'] >= (df_time_filtered['posted_date_latest'] - pd.Timedelta(days=days))]
                    df_time_filtered = df_time_filtered.drop(columns=['posted_date_latest'])
    
                # aggregate features
                agg_features = df_time_filtered.groupby('prism_consumer_id')['amount'].agg(
                    mean='mean',
                    median='median',
                    std='std',
                    max='max',
                    min='min',
                    count='count'
                )
    
                # total transaction count per consumer in the time window
                total_trxn_counts = trxnDF.groupby('prism_consumer_id')['amount'].count()
    
                # compute percentage of transactions in this category for the time window
                percentage_trxn = (agg_features['count'] / total_trxn_counts).fillna(0)
                percentage_trxn = percentage_trxn.rename(f"{category}_{window_name}_percent")
    
                agg_features = agg_features.rename(columns=lambda x: f"{category}_{window_name}_{x}")
    
                features_dict[f"{category}_{window_name}"] = pd.concat([agg_features, percentage_trxn], axis=1)
    
    
        final_features = pd.concat(features_dict.values(), axis=1).fillna(0)
        return final_features

    def compute_threshold_stats(df, thresholds, cat_label):
        """
        Computes statistics on the number of transactions and checks if transaction amounts exceed specified thresholds for each consumer.
        Args:
            df (pd.DataFrame)
            thresholds (list of float): A list of threshold values to check against the transaction amounts.
            cat_label (str): A label for the category being analyzed, which is used to name the output columns.
    
        Returns:
            pd.DataFrame: A dataframe with total number of transaction per consumer and binary for whether any transaction exceeeds specified threshold  

        """
        # Count total gambling transactions
        counts = df.groupby('prism_consumer_id').size().reset_index(name=f'{cat_label}_count')
    
        # Check thresholds
        threshold_flags = df.groupby('prism_consumer_id')['amount'].agg(lambda x: [any(x >= t) for t in thresholds]).apply(pd.Series)
        threshold_flags.columns = [f'{cat_label}_over_{t}' for t in thresholds]
        threshold_flags = threshold_flags.astype(int)  # Convert column to integer
    
        # Merge counts and flags
        result = counts.merge(threshold_flags, on='prism_consumer_id', how='left')
    
        return result


    # creating relevant outflows df with only expenses
    debits_not_expenses = ['SELF_TRANSFER', 'ATM_CASH']
    outflows_agg_df = trxnDF[~trxnDF['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    outflows_agg_df.columns = ['outflows_amt_' + col for col in outflows_agg_df.columns]

    trxn_df_last_14_days = filter_time_window(trxnDF, days=14)
    trxn_df_last_30_days = filter_time_window(trxnDF, days=30)
    trxn_df_last_3_months = filter_time_window(trxnDF, months=3)
    trxn_df_last_6_months = filter_time_window(trxnDF, months=6)
    trxn_df_last_year = filter_time_window(trxnDF, years=1)
    

    balance_ftrs = balanceDF.groupby('prism_consumer_id')['curr_balance'].agg(['mean', 'std', 'median', 'min', 'max'])
    balance_ftrs.columns = ['balance_' + x for x in balance_ftrs.columns]
    balance_last_14_days = filter_time_window(balanceDF, days=14)
    balance_last_30_days = filter_time_window(balanceDF, days=30)
    balance_last_3_months = filter_time_window(balanceDF, months=1)
    balance_last_6_months = filter_time_window(balanceDF, months=6)
    balance_last_year = filter_time_window(balanceDF, years=1)
    
    balance_last_14_days_metrics = balance_last_14_days.groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    balance_last_14_days_metrics.columns = ['balance_last_14_days_' + x for x in balance_last_14_days_metrics.columns]
    
    balance_last_30_days_metrics = balance_last_30_days.groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    balance_last_30_days_metrics.columns = ['balance_last_30_days_' + x for x in balance_last_30_days_metrics.columns]
    
    balance_last_3_months_metrics = balance_last_3_months.groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    balance_last_3_months_metrics.columns = ['balance_last_3_months_' + x for x in balance_last_3_months_metrics.columns]
    
    balance_last_6_months_metrics = balance_last_6_months.groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    balance_last_6_months_metrics.columns = ['balance_last_6_months_' + x for x in balance_last_6_months_metrics.columns]
    
    balance_last_year_metrics = balance_last_year.groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    balance_last_year_metrics.columns = ['balance_last_1_year_' + x for x in balance_last_year_metrics.columns]
    
    balance_dfs = [consDF, balance_ftrs, balance_last_14_days_metrics, balance_last_30_days_metrics, balance_last_3_months_metrics, balance_last_6_months_metrics, balance_last_year_metrics]
    balance_ftrs = reduce(lambda left, right: pd.merge(left, right, on='prism_consumer_id', how='left'), balance_dfs)
    balance_ftrs = balance_ftrs.drop(columns=['evaluation_date', 'credit_score', 'DQ_TARGET'])

    balance_delta_overall = compute_balance_delta(balanceDF, balanceDF, 'overall')
    balance_delta_14d = compute_balance_delta(balanceDF, balance_last_14_days, '14d')
    balance_delta_30d = compute_balance_delta(balanceDF, balance_last_30_days, '30d')
    balance_delta_3m = compute_balance_delta(balanceDF, balance_last_3_months, '3m')
    balance_delta_6m = compute_balance_delta(balanceDF, balance_last_6_months, '6m')
    balance_delta_1y = compute_balance_delta(balanceDF, balance_last_year, '1y')
    
    
    balance_deltas_dfs = [consDF, balance_delta_overall, balance_delta_14d, balance_delta_30d, balance_delta_3m, balance_delta_6m, balance_delta_1y]
    balance_deltas_ftrs = reduce(lambda left, right: pd.merge(left, right, on='prism_consumer_id', how='left'), balance_deltas_dfs)
    balance_deltas_ftrs = balance_deltas_ftrs.drop(columns=['evaluation_date', 'credit_score', 'DQ_TARGET'])


    # creating windowed expenses aggregate metrics
    outflows_ftrs = trxnDF[~trxnDF['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    outflows_ftrs.columns = ['outflows_amt_' + col for col in outflows_ftrs.columns]
    
    outflows_last_14_days_agg_df = trxn_df_last_14_days[~trxn_df_last_14_days['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    outflows_last_14_days_agg_df.columns = ['outflows_amt_last_14_days_' + col for col in outflows_last_14_days_agg_df.columns]
    
    outflows_last_30_days_agg_df = trxn_df_last_30_days[~trxn_df_last_30_days['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    outflows_last_30_days_agg_df.columns = ['outflows_amt_last_30_days_' + col for col in outflows_last_30_days_agg_df.columns]
    
    outflows_last_3_months_agg_df = trxn_df_last_3_months[~trxn_df_last_3_months['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    outflows_last_3_months_agg_df.columns = ['outflows_amt_last_3_months_' + col for col in outflows_last_3_months_agg_df.columns]
    
    outflows_last_6_months_agg_df = trxn_df_last_6_months[~trxn_df_last_6_months['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    outflows_last_6_months_agg_df.columns = ['outflows_amt_last_6_months_' + col for col in outflows_last_6_months_agg_df.columns]
    
    outflows_last_year_agg_df = trxn_df_last_year[~trxn_df_last_year['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max'])
    outflows_last_year_agg_df.columns = ['outflows_amt_last_year_' + col for col in outflows_last_year_agg_df.columns]
    
    outflows_df = [consDF, outflows_ftrs, outflows_last_14_days_agg_df, outflows_last_30_days_agg_df, outflows_last_3_months_agg_df, outflows_last_6_months_agg_df, outflows_last_year_agg_df]
    outflows_ftrs = reduce(lambda left, right: pd.merge(left, right, on='prism_consumer_id', how='left'), outflows_df)
    outflows_ftrs = outflows_ftrs.drop(columns=['evaluation_date', 'credit_score', 'DQ_TARGET'])

    category_features = generate_category_features(trxnDF, cat_map, cat_ok)

    gambling_df_all = balanceDF[balanceDF['cat_name'] == 'GAMBLING']
    gambling_thresholds = [50, 100, 500, 1000]
    
    # Filter for different time periods
    gambling_last_month = filter_time_window(gambling_df_all, months=1)
    gambling_last_6m = filter_time_window(gambling_df_all, months=6)
    gambling_last_year = filter_time_window(gambling_df_all, years=1)
    
    gambling_stats_all = compute_threshold_stats(gambling_df_all, gambling_thresholds, 'all')
    gambling_stats_month = compute_threshold_stats(gambling_last_month, gambling_thresholds, '1m')
    gambling_stats_6m = compute_threshold_stats(gambling_last_6m, gambling_thresholds, '6m')
    gambling_stats_year = compute_threshold_stats(gambling_last_year, gambling_thresholds, '1y')
    
    gambling_df = [consDF, gambling_stats_all, gambling_stats_month, gambling_stats_6m, gambling_stats_year]
    gambling_ftrs = reduce(lambda left, right: pd.merge(left, right, on='prism_consumer_id', how='left'), gambling_df)
    gambling_ftrs = gambling_ftrs.drop(columns=['evaluation_date', 'credit_score', 'DQ_TARGET'])


    feature_dfs = [consDF, balance_ftrs, balance_deltas_ftrs, category_features, outflows_ftrs, gambling_ftrs,]
    features_all = reduce(lambda left, right: pd.merge(left, right, on='prism_consumer_id', how='left'), feature_dfs)
    features_all = features_all.drop(columns=['prism_consumer_id', 'evaluation_date', 'credit_score'])
    features_all = features_all.fillna(0)
    return features_all

def split_features(features_df):
    """
    Splits the features dataframe into 80/20 training and testing sets.

    Args:
        features_df (pd.DataFrame): The dataframe containing the features and the target column 'DQ_TARGET'.

    Returns:
        tuple: A tuple containing the following:
            - X_train (pd.DataFrame)
            - X_test (pd.DataFrame)
            - y_train (pd.Series)
            - y_test (pd.Series
    
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features_df.drop(columns='DQ_TARGET'), features_df['DQ_TARGET'], test_size=0.25
    )

    return X_train, X_test, y_train, y_test

def standardize_features(X_train, X_test):
    """
    Standardizes the feature data, excluding binary features, using StandardScaler.

    Args:
        X_train (pd.DataFrame): The training set features.
        X_test (pd.DataFrame): The testing set features.

    Returns:
        tuple: containing the following:
            - X_train_standardized (pd.DataFrame): The standardized training set features.
            - X_test_standardized (pd.DataFrame): The standardized testing set features.
    """
    # instantiate StandardScaler() to standardize features, excluding binary features
    exclude_columns_standardize = [col for col in X_train.columns if '_over_' in col] # _over_ columns are threshold columns (binary)
    standardize_columns = X_train.columns.difference(exclude_columns_standardize)
    # exclude_columns_standardize = ['GAMBLING', 'BNPL', 'OVERDRAFT'] # binary features that shouldn't be standardized
    
    transformer = ColumnTransformer([('std_scaler', StandardScaler(), standardize_columns)], remainder='passthrough')

    X_train_standardized = transformer.fit_transform(X_train)
    X_train_standardized = pd.DataFrame(
        X_train_standardized, columns=(list(standardize_columns) + exclude_columns_standardize)
    )

    # standardize test features
    X_test_standardized = transformer.transform(X_test)
    X_test_standardized = pd.DataFrame(
        X_test_standardized, columns=(list(standardize_columns) + exclude_columns_standardize)
    )

    return X_train_standardized, X_test_standardized





        