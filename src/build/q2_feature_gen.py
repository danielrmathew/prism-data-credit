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

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN

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

    cat_mappings = dict(zip(cat_map['category_id'], cat_map['category']))
    
    # changing column types for processing
    acctDF.balance_date = pd.to_datetime(acctDF.balance_date)
    acctDF = acctDF.astype({'prism_consumer_id': int, 'prism_account_id': int})
    
    consDF.evaluation_date = pd.to_datetime(consDF.evaluation_date)
    consDF = consDF.astype({'prism_consumer_id': int})
    
    trxnDF.posted_date = pd.to_datetime(trxnDF.posted_date)
    trxnDF = trxnDF.astype({'prism_consumer_id': int, 'prism_transaction_id': int})
    trxnDF.category = trxnDF.category.replace(cat_mappings)

    return acctDF, consDF, trxnDF
    
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

def compute_balance_delta(balance_filtered, label):
    """
    Computes the change in balance for a given dataframe.
    Args:
        balance_filtered (pd.DataFrame): dataframe to window over 
        label (str): the time period that is windowed over
    Returns:
        pd.DataFrame: windowed balance delta
    """
    # Get last balance per consumer
    last_balance = balance_filtered.groupby('prism_consumer_id')['curr_balance'].last().reset_index()
    last_balance = last_balance.rename(columns={'curr_balance': 'curr_balance_last'})
    
    # Get earliest balance in the filtered dataset
    first_balance = balance_filtered.groupby('prism_consumer_id')['curr_balance'].first().reset_index()
    first_balance = first_balance.rename(columns={'curr_balance': 'curr_balance_first'})
    
    # Merge and compute balance delta
    df_merged = last_balance.merge(first_balance, on='prism_consumer_id', how='left')
    df_merged[f'balance_delta_{label}'] = df_merged['curr_balance_last'] - df_merged['curr_balance_first']
    
    # Keep only relevant columns
    return df_merged[['prism_consumer_id', f'balance_delta_{label}']]

def calc_balances_all(acctDF, trxnDF):
    """
    Creates a balance dataframe (balanceDF) that has the current balance for a consumer after every transaction.
    Args:
        acctDF (pd.DataFrame): the account dataframe
        trxnDF (pd.DataFrame): the transaction dataframe
    Returns:
        pd.DataFrame: the dataframe with the running current balance
    """
    check_acct_totals = acctDF[acctDF.account_type == 'CHECKING'].groupby(['prism_consumer_id', 'balance_date']).sum()
    check_acct_totals = check_acct_totals.reset_index()
    check_acct_totals = check_acct_totals.drop(axis=1,labels='prism_account_id')

    # Merge transactions with account balances
    merged = trxnDF.merge(
        check_acct_totals[['prism_consumer_id', 'balance_date', 'balance']], 
        on='prism_consumer_id', how='left'
    )

    # Identify pre and post transactions
    merged['is_pre'] = merged['posted_date'] <= merged['balance_date']

    # Set adjustment values based on pre/post period
    merged['adjustment'] = 0.0
    merged.loc[merged['is_pre'] & (merged['credit_or_debit'] == 'CREDIT'), 'adjustment'] = -merged['amount']
    merged.loc[merged['is_pre'] & (merged['credit_or_debit'] == 'DEBIT'), 'adjustment'] = merged['amount']
    merged.loc[~merged['is_pre'] & (merged['credit_or_debit'] == 'CREDIT'), 'adjustment'] = merged['amount']
    merged.loc[~merged['is_pre'] & (merged['credit_or_debit'] == 'DEBIT'), 'adjustment'] = -merged['amount']

    # Pre-balance transactions: Sort descending and apply reverse cumsum
    pre_trans = merged[merged['is_pre']].sort_values(by=['prism_consumer_id', 'posted_date'], ascending=[True, False])
    pre_trans['curr_balance'] = pre_trans.groupby('prism_consumer_id')['adjustment'].cumsum() + pre_trans.groupby('prism_consumer_id')['balance'].transform('first')

    # Post-balance transactions: Sort ascending and apply forward cumsum
    post_trans = merged[~merged['is_pre']].sort_values(by=['prism_consumer_id', 'posted_date'], ascending=[True, True])
    post_trans['curr_balance'] = post_trans.groupby('prism_consumer_id')['adjustment'].cumsum() + post_trans.groupby('prism_consumer_id')['balance'].transform('first')

    # Combine results
    result = pd.concat([pre_trans, post_trans]).sort_values(by=['prism_consumer_id', 'posted_date'])

    # returns balanceDF
    return result[['prism_consumer_id', 'prism_transaction_id', 'category', 'amount', 'credit_or_debit', 'posted_date', 'curr_balance']].sort_values(by='posted_date', ascending=False)

def create_balance_features(balanceDF, consDF):
    """
    Creates balance features of different summary statistics across different time windows.
    Args:
        balanceDF (pd.DataFrame): the output of the previous function, has the running balance
        consDF (pd.DataFrame): the consumer dataframe
    Returns:
        pd.DataFrame: features dataframe of balance statistics and deltas 
    """
    balance_ftrs = balanceDF.groupby('prism_consumer_id')['curr_balance'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    balance_ftrs.columns = ['balance_' + x for x in balance_ftrs.columns]
    balance_last_14_days = filter_time_window(balanceDF, days=14)
    balance_last_30_days = filter_time_window(balanceDF, days=30)
    balance_last_3_months = filter_time_window(balanceDF, months=1)
    balance_last_6_months = filter_time_window(balanceDF, months=6)
    balance_last_year = filter_time_window(balanceDF, years=1)
    
    balance_last_14_days_metrics = balance_last_14_days.groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    balance_last_14_days_metrics.columns = ['balance_last_14_days_' + x for x in balance_last_14_days_metrics.columns]
    
    balance_last_30_days_metrics = balance_last_30_days.groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    balance_last_30_days_metrics.columns = ['balance_last_30_days_' + x for x in balance_last_30_days_metrics.columns]
    
    balance_last_3_months_metrics = balance_last_3_months.groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    balance_last_3_months_metrics.columns = ['balance_last_3_months_' + x for x in balance_last_3_months_metrics.columns]
    
    balance_last_6_months_metrics = balance_last_6_months.groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    balance_last_6_months_metrics.columns = ['balance_last_6_months_' + x for x in balance_last_6_months_metrics.columns]
    
    balance_last_year_metrics = balance_last_year.groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    balance_last_year_metrics.columns = ['balance_last_1_year_' + x for x in balance_last_year_metrics.columns]
    
    balance_dfs = [consDF[['prism_consumer_id']], balance_ftrs, balance_last_14_days_metrics, balance_last_30_days_metrics, balance_last_3_months_metrics, balance_last_6_months_metrics, balance_last_year_metrics]
    balance_ftrs = reduce(lambda left, right: pd.merge(left, right, on='prism_consumer_id', how='left'), balance_dfs)

    balance_delta_overall = compute_balance_delta(balanceDF, 'overall')
    balance_delta_14d = compute_balance_delta(balance_last_14_days, '14d')
    balance_delta_30d = compute_balance_delta(balance_last_30_days, '30d')
    balance_delta_3m = compute_balance_delta(balance_last_3_months, '3m')
    balance_delta_6m = compute_balance_delta(balance_last_6_months, '6m')
    balance_delta_1y = compute_balance_delta(balance_last_year, '1y')

    balance_deltas_dfs = [consDF[['prism_consumer_id']], balance_delta_overall, balance_delta_14d, balance_delta_30d, balance_delta_3m, balance_delta_6m, balance_delta_1y]
    balance_deltas_ftrs = reduce(lambda left, right: pd.merge(left, right, on='prism_consumer_id', how='left'), balance_deltas_dfs)

    return balance_ftrs, balance_deltas_ftrs

def create_inflows_outflows_features(trxn_df):
    """
    Creates a features dataframe of statistics for inflows and outflows transactions
    Args:
        trxn_df (pd.DataFrame): the transaction dataframe
    Returns:
        pd.DataFrame: the features dataframe of inflows and outflows features
    """
    # creating relevant outflows df with only expenses
    debits_not_expenses = ['SELF_TRANSFER', 'ATM_CASH']
    credits_not_income = ['SELF_TRANSFER', 'LOAN', 'REFUND'] # maybe TAX
    
    outflows_agg_df = trxn_df[(trxn_df.credit_or_debit == 'DEBIT') & (~trxn_df['category'].isin(debits_not_expenses))] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    outflows_agg_df.columns = ['OUTFLOWS_amt_' + col for col in outflows_agg_df.columns]
    inflows_agg_df = trxn_df[(trxn_df.credit_or_debit == 'CREDIT') & (~trxn_df['category'].isin(credits_not_income))] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    inflows_agg_df.columns = ['INFLOWS_amt_' + col for col in inflows_agg_df.columns]
    
    ### creating filtered window transactions dfs for inflows and outflows ###
    trxn_df_last_14_days = filter_time_window(trxn_df, days=14)
    trxn_df_last_30_days = filter_time_window(trxn_df, days=30)
    trxn_df_last_3_months = filter_time_window(trxn_df, months=3)
    trxn_df_last_6_months = filter_time_window(trxn_df, months=6)
    trxn_df_last_year = filter_time_window(trxn_df, years=1)

    # grabbing debits and credits across time periods
    debits_df_last_14_days = trxn_df_last_14_days[trxn_df_last_14_days.credit_or_debit == 'DEBIT']
    debits_df_last_30_days = trxn_df_last_30_days[trxn_df_last_30_days.credit_or_debit == 'DEBIT']
    debits_df_last_3_months = trxn_df_last_3_months[trxn_df_last_3_months.credit_or_debit == 'DEBIT']
    debits_df_last_6_months = trxn_df_last_6_months[trxn_df_last_6_months.credit_or_debit == 'DEBIT']
    debits_df_last_year = trxn_df_last_year[trxn_df_last_year.credit_or_debit == 'DEBIT']
    
    credits_df_last_14_days = trxn_df_last_14_days[trxn_df_last_14_days.credit_or_debit == 'CREDIT']
    credits_df_last_30_days = trxn_df_last_30_days[trxn_df_last_30_days.credit_or_debit == 'CREDIT']
    credits_df_last_3_months = trxn_df_last_3_months[trxn_df_last_3_months.credit_or_debit == 'CREDIT']
    credits_df_last_6_months = trxn_df_last_6_months[trxn_df_last_6_months.credit_or_debit == 'CREDIT']
    credits_df_last_year = trxn_df_last_year[trxn_df_last_year.credit_or_debit == 'CREDIT']

    # creating windowed expenses aggregate metrics
    outflows_last_14_days_agg_df = debits_df_last_14_days[~debits_df_last_14_days['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    outflows_last_14_days_agg_df.columns = ['OUTFLOWS_amt_last_14_days_' + col for col in outflows_last_14_days_agg_df.columns]
    
    outflows_last_30_days_agg_df = debits_df_last_30_days[~debits_df_last_30_days['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    outflows_last_30_days_agg_df.columns = ['OUTFLOWS_amt_last_30_days_' + col for col in outflows_last_30_days_agg_df.columns]
    
    outflows_last_3_months_agg_df = debits_df_last_3_months[~debits_df_last_3_months['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    outflows_last_3_months_agg_df.columns = ['OUTFLOWS_amt_last_3_months_' + col for col in outflows_last_3_months_agg_df.columns]
    
    outflows_last_6_months_agg_df = debits_df_last_6_months[~debits_df_last_6_months['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    outflows_last_6_months_agg_df.columns = ['OUTFLOWS_amt_last_6_months_' + col for col in outflows_last_6_months_agg_df.columns]
    
    outflows_last_year_agg_df = debits_df_last_year[~debits_df_last_year['category'].isin(debits_not_expenses)] \
                            .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    outflows_last_year_agg_df.columns = ['OUTFLOWS_amt_last_year_' + col for col in outflows_last_year_agg_df.columns]

    # creating windowed inflows aggregate metrics
    inflows_last_14_days_agg_df = credits_df_last_14_days[~credits_df_last_14_days['category'].isin(credits_not_income)] \
                    .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    inflows_last_14_days_agg_df.columns = ['INFLOWS_amt_last_14_days_' + col for col in inflows_last_14_days_agg_df.columns]
    
    inflows_last_30_days_agg_df = credits_df_last_30_days[~credits_df_last_30_days['category'].isin(credits_not_income)] \
                    .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    inflows_last_30_days_agg_df.columns = ['INFLOWS_amt_last_30_days_' + col for col in inflows_last_30_days_agg_df.columns]
    
    inflows_last_3_months_agg_df = credits_df_last_3_months[~credits_df_last_3_months['category'].isin(credits_not_income)] \
                    .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    inflows_last_3_months_agg_df.columns = ['INFLOWS_amt_last_3_months_' + col for col in inflows_last_3_months_agg_df.columns]
    
    inflows_last_6_months_agg_df = credits_df_last_6_months[~credits_df_last_6_months['category'].isin(credits_not_income)] \
                    .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    inflows_last_6_months_agg_df.columns = ['INFLOWS_amt_last_6_months_' + col for col in inflows_last_6_months_agg_df.columns]
    
    inflows_last_year_agg_df = credits_df_last_year[~credits_df_last_year['category'].isin(credits_not_income)] \
                    .groupby('prism_consumer_id')['amount'].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    inflows_last_year_agg_df.columns = ['INFLOWS_amt_last_year_' + col for col in inflows_last_year_agg_df.columns]

    all_outflows_dfs = [
                             outflows_agg_df, outflows_last_14_days_agg_df, outflows_last_30_days_agg_df,
                             outflows_last_3_months_agg_df, outflows_last_6_months_agg_df, outflows_last_year_agg_df,
                         ]

    all_inflows_dfs = [
                             inflows_agg_df, inflows_last_14_days_agg_df, inflows_last_30_days_agg_df,
                             inflows_last_3_months_agg_df, inflows_last_6_months_agg_df, inflows_last_year_agg_df,
                         ]

    return all_outflows_dfs, all_inflows_dfs

def create_binary_features(acct_df, trxn_df):
    """
    Creates a features dataframe of binary features, such as having gambling transactions, overdraft fees, etc.
    Args:
        acct_df (pd.DataFrame): the account dataframe
        trxn_df (pd.DataFrame): the transaction dataframe
    Returns:
        pd.DataFrame: a features dataframe of binary features
    """
    # creating flags for risky categories
    risky_categories = ['GAMBLING', 'BNPL', 'OVERDRAFT'] 
    trxn_category_table = trxn_df.pivot_table(index='prism_consumer_id', columns='category', values='amount', aggfunc='size', fill_value=0)
    
    gambling_flag_df = pd.DataFrame((trxn_category_table['GAMBLING'] > 0).astype(int)).reset_index()
    bnpl_flag_df = pd.DataFrame((trxn_category_table['BNPL'] > 0).astype(int)).reset_index()
    overdraft_flag_df = pd.DataFrame((trxn_category_table['OVERDRAFT'] > 0).astype(int)).reset_index()
    has_savings_df = acct_df.groupby('prism_consumer_id', as_index=False).agg(HAS_SAVINGS_ACCT=('account_type', lambda x: int('SAVINGS' in x.values)))

    return [gambling_flag_df, bnpl_flag_df, overdraft_flag_df, has_savings_df]

def generate_category_features(trxnDF, categories):
    """
    Generates transaction-based features for each selected category over multiple time windows.
    Args:
        trxnDF (pd.DataFrame): DataFrame containing transaction data.
        categories (str or list): One or more transaction categories to filter.
    Returns:
        pd.DataFrame: Aggregated features per prism_consumer_id.
    """
    if isinstance(categories, str):
        categories = [categories]

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
                count='count',
                sum='sum'
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
    Creates a features dataframe of thresholded stats, like calculating if some transaction went over some threshold
    Args:
        df (pd.DataFrame): transaction dataframe
        thresholds (list): list of thresholds to check
        cat_label (str): category label
    Returns:
        pd.DataFrame: the features dataframe of inflows and outflows features
    """
    # Count total gambling transactions
    counts = df.groupby('prism_consumer_id').size().reset_index(name=f'{cat_label}_count')

    # Check thresholds
    threshold_flags = df.groupby('prism_consumer_id')['amount'].agg(lambda x: [any(x >= t) for t in thresholds]).apply(pd.Series)
    threshold_flags.columns = [f'{cat_label}_over_{t}' for t in thresholds]
    threshold_flags = threshold_flags.astype(bool)  # Convert to True/False

    # Merge counts and flags
    result = counts.merge(threshold_flags, on='prism_consumer_id', how='left')

    return result

def create_features_df(cons_df, acct_df, trxn_df):
    """
    Creates a complete features dataframe of all the engineered features
    Args:
        cons_df (pd.DataFrame): the consumer dataframe
        acct_df (pd.DataFrame): the account dataframe
        trxn_df (pd.DataFrame): the transaction dataframe
    Returns:
        pd.DataFrame: the final features dataframe
    """
        
    # creating sum of balances feature
    sum_of_balance_df = pd.DataFrame(acct_df.groupby('prism_consumer_id')['balance'].sum()) \
                       .rename(columns={'balance': 'sum_acct_balances'}).reset_index()

    # creating credit_minus_debit feature
    credit_minus_debit_df = pd.DataFrame(trxn_df.groupby('prism_consumer_id') \
                            .apply(
                                lambda group: group.loc[group["credit_or_debit"] == "CREDIT", "amount"].sum()
                                            - group.loc[group["credit_or_debit"] == "DEBIT", "amount"].sum()
                                )).reset_index().rename(columns={0: "credit_minus_debit"})

    # number of "income" sources for each consumer -- TODO: disregard credits that aren't actual income
    num_income_sources = trxn_df[trxn_df.credit_or_debit == 'CREDIT'].groupby('prism_consumer_id')['category'].nunique()
    num_income_source_df = pd.DataFrame(num_income_sources).reset_index() \
                                    .rename(columns=
                                            {'index': 'prism_consumer_id', 
                                             'category': 'num_income_source'}
                                           )
    
    # number of accounts for each consumer
    num_accounts_df = acct_df.groupby('prism_consumer_id')['account_type'].count().reset_index() \
                             .rename(columns={'account_type': 'num_accounts'})
    
    all_outflows_dfs, all_inflows_dfs = create_inflows_outflows_features(trxn_df)
    binary_features_dfs = create_binary_features(acct_df, trxn_df)

    # create category features
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
    category_features = generate_category_features(trxn_df, cat_ok)

    # create balance features
    balanceDF = calc_balances_all(acct_df, trxn_df)
    balance_ftrs, balance_deltas_ftrs = create_balance_features(balanceDF, cons_df)

    gambling_df_all = balanceDF[balanceDF['category'] == 'GAMBLING']
    gambling_thresholds = [50, 100, 500, 1000]
    
    # Filter for different time periods
    gambling_last_month = filter_time_window(gambling_df_all, months=1)
    gambling_last_6m = filter_time_window(gambling_df_all, months=6)
    gambling_last_year = filter_time_window(gambling_df_all, years=1)
    
    gambling_stats_all = compute_threshold_stats(gambling_df_all, gambling_thresholds, 'all')
    gambling_stats_month = compute_threshold_stats(gambling_last_month, gambling_thresholds, '1m')
    gambling_stats_6m = compute_threshold_stats(gambling_last_6m, gambling_thresholds, '6m')
    gambling_stats_year = compute_threshold_stats(gambling_last_year, gambling_thresholds, '1y')
    
    gambling_df = [cons_df[['prism_consumer_id']], gambling_stats_all, gambling_stats_month, gambling_stats_6m, gambling_stats_year]
    gambling_ftrs = reduce(lambda left, right: pd.merge(left, right, on='prism_consumer_id', how='left'), gambling_df)
    
    # merging all features into features df
    features_df = reduce(lambda left, right: left.merge(right, on='prism_consumer_id', how='left'), 
                         [cons_df, sum_of_balance_df, credit_minus_debit_df, num_income_source_df, 
                          num_accounts_df, balance_ftrs, balance_deltas_ftrs, category_features, gambling_ftrs] + 
                         all_outflows_dfs + all_inflows_dfs + binary_features_dfs
                        )
    fill_na_cols = features_df.columns.difference(['DQ_TARGET'])
    features_df[fill_na_cols] = features_df[fill_na_cols].fillna(0)
 
    return features_df

def split_data(final_features_df, drop_credit_score=True, test_size=0.25):
    """
    Splits the features dataframe into a train and test set
    Args:
        final_features_df (pd.DataFrame): the complete features dataframe
        drop_credit_score (bool): whether or not to drop the credit score feature, default True
        test_size (float): the proportion of the dataset that is the test set, default 0.25
    Returns:
        X_train (pd.DataFrame): the train features
        X_test (pd.DataFrame): the test features
        y_train (pd.Series): the train labels
        y_test (pd.Series): the test labels
        train_ids (pd.Series): the train consumer ids
        test_ids (pd.Series): the test consumer ids
    """
    drop_cols = ['DQ_TARGET', 'evaluation_date']
    if drop_credit_score:
        drop_cols.append('credit_score')
    
    X_train, X_test, y_train, y_test = train_test_split(
        final_features_df.drop(columns=drop_cols), 
        final_features_df['DQ_TARGET'], test_size=test_size, stratify=final_features_df['DQ_TARGET']
    )

    train_ids, test_ids = X_train.pop('prism_consumer_id'), X_test.pop('prism_consumer_id')
    return X_train, X_test, y_train, y_test, train_ids, test_ids

def standardize(X_train, *args):
    """
    Standardizes the input dataframes
    Args:
        X_train (pd.DataFrame): the train features 
        *args (list[pd.DataFrame]): at least one other dataframe that will be standardized after getting fit from the train df
    Returns:
        list[pd.DataFrame]: a list of standardized dataframes
    """
    # instantiate StandardScaler() to standardize features, excluding binary features
    scaler = StandardScaler()
    exclude_columns_standardize = ['GAMBLING', 'BNPL', 'OVERDRAFT', 'HAS_SAVINGS_ACCT'] # binary features that shouldn't be standardized
    standardize_features = X_train.columns.difference(exclude_columns_standardize)
    
    transformer = ColumnTransformer([
        ('std_scaler', StandardScaler(), standardize_features)  # Standardize all except excluded ones
    ], remainder='passthrough')
    
    X_train_standardized = transformer.fit_transform(X_train)
    X_train_standardized = pd.DataFrame(X_train_standardized, columns=list(standardize_features) + exclude_columns_standardize)

    X_datasets = [X_train_standardized]
    
    # standardize test features
    for X_dataset in args:
        X_test_standardized = transformer.transform(X_dataset)
        X_test_standardized = pd.DataFrame(X_test_standardized, columns=list(standardize_features) + exclude_columns_standardize)
        X_datasets.append(X_test_standardized)

    return X_datasets

def resample_data(X, y, new_0=0.75, new_1=0.25):
    """
    Resamples the datasets to the new class proportions
    Args:
        X (pd.DataFrame): features dataframe
        y (pd.Series): dataset labels
        new_0 (float): the proportion of the output dataset that is class 0, default 0.75
        new_1 (float): the proportion of the output dataset that is class 1, default 0.25
    Returns:
        X_resampled (pd.DataFrame): the resampled features df
        y_resampled (pd.Series): the resampled class labels
    """
    # Check class counts
    new_0 = int(new_0 * len(y))  # Target count for balancing
    new_1 = int(new_1 * len(y))
    
    # Ensure SMOTE does not create more than the original number of samples
    smote = SMOTE(sampling_strategy={1: new_1},)
    
    # Undersampling the majority class to match the new minority class count
    under = RandomUnderSampler(sampling_strategy={0: new_0},)

    smoteenn = SMOTEENN(sampling_strategy={1: new_1,})
    
    # Combine SMOTE & Undersampling in a pipeline
    resample_pipeline = Pipeline(steps=[('smote', smote), ('under', under)])
    
    X_resampled, y_resampled = resample_pipeline.fit_resample(X, y)

    X_resampled = X_resampled[X.columns]
    cols_dtype_object = X_resampled.dtypes[X_resampled.dtypes == 'object'].index
    X_resampled[cols_dtype_object] = X_resampled[cols_dtype_object].astype(float)

    return X_resampled, y_resampled

        