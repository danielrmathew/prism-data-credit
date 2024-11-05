# Building Better Credit Scores: Machine Learning and NLP for Optimized Risk Assessment

This project aims to enhance credit scoring by using machine learning and NLP to analyze transaction data, creating a fairer, more comprehensive measure of creditworthiness.

## Description

Credit scores are pivotal in today’s financial landscape, influencing everything from rental eligibility to access to health insurance, yet the formula for calculating creditworthiness has long been shrouded in mystery and often overlooks important nuances. Typically, the credit score is determined based on five factors: payment history, the amount owed, new credit, credit history, and credit mix. This structure can place individuals with limited credit history— especially young adults who are just starting out building their credits —at a compounded disadvantage, restricting their access to loans, credit cards, employment opportunities, and insurance. This report aims to address this unfairness by creating a more comprehensive measure of creditworthiness by incorporating detailed account transaction analysis into the equation. To achieve this, we will build a transaction categorization model using NLP techniques, enabling a deeper and fairer evaluation of financial behavior.


## Getting Started

### Executing program

* Run all cells to ensure the code executes sequentially and that all dependencies and variables are initialized properly.
* This will verify that outputs display as expected throughout the notebook.
* Master.ipynb contains the culmination of our finalized work.
* For individual contributions, refer to the notebooks labeled with our respective names.

## Progress

### Dataset Explanation:

The outflows dataset provides a detailed record of consumer transactions, capturing various attributes that contribute to a comprehensive understanding of spending patterns and financial behavior. Each row represents a unique transaction associated with a specific consumer account

|Column	                 |Description|
|---                     |---        |
|`'prism_consumer_id'	`  |ID of Prism consumer|
|`'prism_account_id'`	   |ID of Prism account|
|`'memo'`	         |Description of the transaction|
|`'amount'`	         |Transaction amount|
|`'posted_date'`	     |Date the transaction was posted|
|`'category'`	             |Category assigned to the transaction|

For the purpoe of this project, we only work with outflows_with_memo because that is the subset of the data we were told to work with. This doesn't include rows where memo == category, and thereby includes rows that are pivotal to our prediction task.

###  Train Test Split by Customers:
* Function Definition: Creates dataset_split to divide a dataset into training (75%) and testing (25%) sets based on unique prism_consumer_id.
* Data Filtering: Generates outflows_with_memo by excluding rows where memo equals category.
* Data Splitting: Splits outflows_with_memo into training (outflows_memo_train) and testing (outflows_memo_test) sets.
* Unique IDs Count: Counts and prints unique consumer IDs in the original, training, and testing datasets.
* Descriptive Statistics: Computes and formats descriptive statistics for the original, training, and testing datasets.
* Category Distribution Visualization: Plots horizontal bar charts to visualize category distributions in both training and testing sets.

###  Memo Cleaning:
Complex Preprocessing:
* Remove dates using regex (mm/yy).
* Eliminate location addresses.
* TODO: Remove email addresses.
* TODO: Handle transactions in the format "trans 1 @ $1.00".
  
Simple Preprocessing:
* Convert text to lowercase.
* Remove punctuation (e.g., ,-*#_').
* Remove "XXXX" and any sequence of X's.
* Remove phrases "purchase authorized on" and "purchase, checkcard".
  
Exclusion Criteria:
* Skip preprocessing for memos that match the category.


### Feature Creation and Model Training:
Feature Creation:
* TF-IDF: Extract features from cleaned_memo using TfidfVectorizer.
* Date Features: Create month, weekday, and day-of-month features; apply one-hot encoding.
* Amount Features: Identify even/odd amounts and bin into deciles; apply one-hot encoding.
* Combine Features: Merge original dataset with TF-IDF, date, and amount features.
  
Train-Test Split:
* Split dataset into training and testing sets (X_train, y_train, X_test, y_test).
  
Model Training:
* Logistic Regression: Fit model, evaluate training/testing accuracy.
* XGBoost: Fit model with encoded labels, evaluate accuracy.
* Random Forest: Fit model, evaluate accuracy.
  
Model Evaluation:
* Predictions: Generate and print accuracy for each model.
* Confusion Matrices: Visualize training/testing predictions with heatmaps.



## Authors

* Aman Kar, akar@ucsd.edu 
* Daniel Mathew, drmathew@ucsd.edu
* Tracy Pham, tnp003@ucsd.edu
