# Building Better Credit Scores: Machine Learning and NLP for Optimized Risk Assessment

Website: https://danielrmathew.github.io/prism-data-credit/

This project aims to enhance credit scoring by using machine learning and NLP to analyze transaction data, creating a fairer, more comprehensive measure of creditworthiness.

## Description

Credit scores are pivotal in today’s financial landscape, influencing everything from rental eligibility to access to health insurance, yet the formula for calculating creditworthiness has long been shrouded in mystery and often overlooks important nuances. Typically, the credit score is determined based on five factors: payment history, the amount owed, new credit, credit history, and credit mix. This structure can place individuals with limited credit history— especially young adults who are just starting out building their credits —at a compounded disadvantage, restricting their access to loans, credit cards, employment opportunities, and insurance. This report aims to address this unfairness by creating a more comprehensive measure of creditworthiness by incorporating detailed account transaction analysis into the equation. To achieve this, we will build a model that generates probability-based scores reflecting the likelihood of delinquency, leveraging detailed bank transaction data to provide a fairer and more transparent assessment of financial responsibility.

## Getting Started

### Executing program

* The dataset has been removed for confidentiality, as it is proprietary data provided by Prism Data.
* For individual contributions, refer to the notebooks labeled with our respective names.

## Progress

### Dataset Explanation:

The project utilizes several datasets for building features and training models. These datasets provide comprehensive information about accounts, consumers, transactions, and category mappings, which are used to build a feature set for prediction tasks. Each dataset serves a different purpose in the data pipeline.

| Dataset                | Description                                      |
|-----------------------|--------------------------------------------------|
| `'q2-ucsd-acctDF'`    | Account-level information|
| `'q2-ucsd-consDF'`    | Consumer-level information|
| `'q2-ucsd-trxnDF'`    | Transaction records|
| `'q2-ucsd-cat-map'`   | Mapping of categories to corresponding cateogryIDs |

### **Core Capabilities**

1. **Feature Creation**
- **Transaction Features**: Aggregated transaction stats (mean, median, std, count) by category over time windows (14 days, 30 days, 3 months, 6 months, 1 year), with category-specific spending flags.
- **Balance Features**: Aggregated balance stats (mean, median, min/max, std) over different time windows, including balance deltas (recent vs earlier balances).
- **Categorical Features**: Mapped transaction categories for spending patterns and transaction counts within time windows.
- **Amount Features**: Statistics (mean, median, etc.) of amounts to track spending behavior.
- **Risky Features**: Flag gambling or high-risk transactions, tracking frequency, amount, and spending fluctuations in related categories (e.g., casinos, betting).
  
2. **Feature Selection**
- **Lasso Regularization Features**: 
  - Logistic Regression with L1 regularization (Lasso) selects the top features by their non-zero coefficients.
- **Point-Biserial Correlation Features**: 
  - Features are ranked by point-biserial correlation with the target variable.

3. **Resampling**
- **SMOTE**
  - Synthetic data generation and sampling that undersamples the majority class and oversamples the minority class

5. **Model Development**

The following models are developed and trained using the provided dataset:

- **HistGradientBoostingClassifier**
- **CatBoostClassifier**
- **LightGBMClassifier**
- **XGBoostClassifier**
- **LogisticRegression**

The `train_and_evaluate` function takes in the model type and trains the selected model, tracking the training time.


4. **Model Evaluation**

After training, the model's performance is evaluated using the following metrics:

- **ROC AUC**: Measures the model's ability to distinguish between classes.
- **Accuracy**: The ratio of correct predictions to total predictions.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall**: The proportion of actual positives that are correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A matrix showing true positives, true negatives, false positives, and false negatives.

The confusion matrix is visualized using a heatmap to highlight performance across both classes (Negative and Positive). 


# **Set Up Instructions**

To ensure reproducibility of the project and its results, follow the steps below:

## **1. Clone the Repository**
Start by cloning the repository to your local machine:
```bash
git clone https://github.com/danielrmathew/prism-data-credit.git
cd prism-data-credit
```

## **2. Set Up the Environment**
Ensure you have Python 3.8 or later installed. Create a virtual environment and activate it:
```bash
python -m venv .prism_env
source .prism_env/bin/activate
```
Install the required dependencies:
```bash
pip install -r q2_requirements.txt
```
## **3. Configure the Project**
Update the `q2_config.yml` file to match your setup:

### **Dataset Configuration**  
- **`ACCOUNT_DF_PATH`**: Path to account dataset (default: `data/q2-ucsd-acctDF.pqt`).  
- **`CONSUMER_DF_PATH`**: Path to consumer dataset (default: `data/q2-ucsd-consDF.pqt`).  
- **`TRANSACTION_DF_PATH`**: Path to transaction dataset (default: `data/q2-ucsd-trxnDF.pqt`).  
- **`CATEGORY_MAP_PATH`**: Path to category mapping file (default: `data/q2-ucsd-cat-map.csv`).  

### **Feature Selection Configuration**  
- **`FEATURE_SELECTION`**: Method for feature selection (**options**: `lasso`, `point_biserial`) (default: `lasso`).  
- **`MAX_FEATURES`**: Maximum number of selected features (default: `50`).
- **`DROP_CREDIT_SCORE`**: Whether or not to include credit score as a feature to train on

### **Model Configuration**  
For models you would like to train and generate reason codes for, set the respective parameters to True
- **Models to Train**:  
  - **Logistic Regression**
  - **HistGB**
  - **CatBoost**
  - **LightGBM**
  - **XGBoost**

After making changes, save the file and run the script.

## **4. Running the Script**
```bash
python q2_run.py
```
## **5. Output**  

After running the script, you should see new files in the `result` folder:  

- The saved model and following files in `q2_result/Model/`.  
- The respective model's confusion matrix image (`confusion_matrix.png`).
- The respective model's reason code distribution (`reason_code_distribution.png`)
- The respective model's scores and top 3 reason codes on the test set (`test_scores.csv`)

These files contain information about the model's performance, including various accuracy metrics and visualizations.  

### **Model Metrics Output**  
The script prints evaluation metrics for each trained model in the following format:  
```bash
<Model Name> metrics:
ROC_AUC: <value>
Accuracy: <value>
Precision: <value>
Recall: <value>
F1-Score: <value>
Confusion Matrix: [[TP FP] [FN TN]]
```
