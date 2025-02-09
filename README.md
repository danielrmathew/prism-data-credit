# Building Better Credit Scores: Machine Learning and NLP for Optimized Risk Assessment

This project aims to enhance credit scoring by using machine learning and NLP to analyze transaction data, creating a fairer, more comprehensive measure of creditworthiness.

## Description

Credit scores are pivotal in today’s financial landscape, influencing everything from rental eligibility to access to health insurance, yet the formula for calculating creditworthiness has long been shrouded in mystery and often overlooks important nuances. Typically, the credit score is determined based on five factors: payment history, the amount owed, new credit, credit history, and credit mix. This structure can place individuals with limited credit history— especially young adults who are just starting out building their credits —at a compounded disadvantage, restricting their access to loans, credit cards, employment opportunities, and insurance. This report aims to address this unfairness by creating a more comprehensive measure of creditworthiness by incorporating detailed account transaction analysis into the equation. To achieve this, we will build a model that generates probability-based scores reflecting the likelihood of delinquency, leveraging detailed bank transaction data to provide a fairer and more transparent assessment of financial responsibility.


## Getting Started

### Executing program

* The dataset has been removed for confidentiality, as it is proprietary data provided by Prism Data.
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

For the purpose of this project, we only work with outflows_with_memo because that is the subset of the data we were told to work with. This doesn't include rows where memo == category, and thereby includes rows that are pivotal to our prediction task.

## **Features**

### **Core Capabilities**
1. **Data Cleaning & Preprocessing**
   - Removes sensitive and irrelevant data like dates, addresses, and email patterns.
   - Handles text standardization (e.g., lowercase conversion, punctuation removal).
   - Filters transactions for relevant memos using exclusion criteria.

2. **Feature Engineering**
   - **NLP Features**: TF-IDF vectorization for transaction descriptions.
   - **Date Features**: Extracted and one-hot encoded month, weekday, and day-of-month.
   - **Amount Features**: Categorized even/odd amounts and deciles with one-hot encoding.

3. **Model Development**
   - **Traditional Models**: Implemented Logistic Regression, Random Forest, XGBoost, Multinomial Naive Bayes (MNB), and Support Vector Machine (SVM) models for baseline performance and benchmark comparisons.
   - **LLM-Based Models**: 
     - **DistilBERT**: Leveraged the lightweight and efficient DistilBERT model for NLP tasks, balancing performance and computational cost.
   - **FastText**: Integrated FastText for fast, efficient, and interpretable word embeddings, particularly useful for handling out-of-vocabulary words and generating robust text representations.

4. **Evaluation**
   - Metrics: accuracy, precision, recall, f1-score, support
   - Visualization: Confusion matrices and Multi-Class AUC curves


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
pip install -r requirements.txt
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

### **Model Configuration**  
Set the models you would like to train to True
- **Models to Train**:  
  - **Logistic Regression**
  - **HistGB**
  - **CatBoost**: 
  - **LightGBM**:
  - **XGBoost**: 

After making changes, save the file and run the script.

## **4. Running the Script**
```bash
python q2_run.py
```
## **5. Output**  

After running the script, you should see new files in the `result` folder:  

- The saved models in `result/models`.  
- The respective model's confusion matrix image (`confusion_matrix.png`).  

These files contain information about the model's performance, including various accuracy metrics and visualizations.  

### **Model Metrics Output**  
The script prints evaluation metrics for each trained model in the following format:  
```bash
<Model Name> metrics: ROC_AUC: <value> Accuracy: <value> Precision: <value> Recall: <value> F1-Score: <value> Confusion Matrix: [[TP FP] [FN TN]]
```
