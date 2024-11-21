# **Building Better Credit Scores: Machine Learning and NLP for Optimized Risk Assessment**

## **Overview**
This project leverages machine learning and natural language processing (NLP) to enhance credit scoring by analyzing transaction data. Our goal is to create a fairer, more comprehensive measure of creditworthiness, addressing the biases in traditional credit scoring methods.

---

## **Motivation**
Traditional credit scores often disadvantage individuals with limited credit history, such as young adults or those without access to credit cards. By incorporating detailed account transaction analysis, we aim to provide a nuanced and equitable view of financial behavior, improving access to loans, employment, and insurance.

---

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
   - Traditional models: Logistic Regression, Random Forest, XGBoost.
   - LLM-based models: Experimentation with Electra and fine-tuned transformers.
   - Hyperparameter optimization for robust performance.

4. **Evaluation**
   - Metrics: AUC/ROC, accuracy, precision, recall, and confusion matrices.
   - Visualization: Charts for feature importance, category distribution, and prediction performance.

---

## **Repository Structure**

```plaintext
├── src/
│   ├── build/
│   │   ├── feature_gen.py            # Generates features (NLP, date, amount)
│   │   ├── feature_gen_llm.py        # Handles LLM-specific preprocessing
│   │   ├── train_llm.py              # Trains LLM models with label mappings
│   │   ├── train_models.py           # Prepares and saves models for production
│   │   ├── train_traditional.py      # Implements traditional ML methods
│   ├── notebooks/
│   │   ├── master.ipynb              # Final consolidated notebook
│   │   ├── Aman’s.ipynb              # Aman's contributions (e.g., cleaning)
│   │   ├── tracy_nn.ipynb            # Tracy’s work on neural networks
│   │   ├── drm_notebook.ipynb        # Daniel's contributions
│   │   └── broken_Electra_llm.ipynb  # Electra experimentation (in-progress)
├── requirements.txt                  # Dependencies
├── config.yaml                       # Configurations (file paths, hyperparams)
├── README.md                         # Documentation
