## imports config and relevant functions
from src.build.feature_gen import (
    read_outflows, clean_memos, get_features_df,
    train_test_split_features
)
from src.build.feature_gen_llm import encode_labels, train_test_split_llm, prepare_fasttext_data
from src.build.train_llm import fit_bert, fit_fasttext

from sklearn.model_selection import train_test_split
# from train_models import ...
# from train_models_llm import ... 
import yaml



if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(config)

    # GET CONFIG
    OUTFLOWS_PATH = config['OUTFLOWS_PATH']
    num_tfidf_features = config['FEATURES']['num_tfidf_features']
    include_date_features = config['FEATURES']['include_date_features']
    include_amount_features = config['FEATURES']['include_amount_features']
    NON_LLM_MODELS = config['MODELS']['NON_LLM']
    train_bert = config['MODELS']['LLM']['bert']
    train_fasttext = config['MODELS']['LLM']['fasttext']
    
    
    assert 1 == 2

    outflows_with_memo = read_outflows(OUTFLOWS_PATH)
    outflows_with_memo = clean_memos(outflows_with_memo)

    ## train, test = output(feature_gen.py)
    if True in NON_LLM_MODELS.values():
        features_df = get_features_df(
            outflows_with_memo, num_tfidf_features, include_date_features, include_amount_features
        )

        X_train, y_train, X_test, y_test = train_test_split_features(features_df)

    if train_bert:
        # generate bert features
        outflows_with_memos_encoded, id2label, label2id = encode_labels(outflows_with_memos)
        pre_tokenized_split, (train_dataset, test_dataset) = train_test_split_llm(outflows_with_memos_encoded)
        pipe, accuracy_dct = fit_bert(train_dataset, test_dataset, id2label, label2id)
        
    if train_fasttext:
        # generate fasttext features
        fasttext_data = prepare_fasttext_data(outflows_with_memo, fasttext_data) # TODO: define fasttext_data


    ## model, ... = output(train_models.py(train, test))
    






    