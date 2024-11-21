## imports config and relevant functions
from src.build.feature_gen import (
    read_outflows, clean_memos, get_features_df,
    train_test_split_features
)
from src.build.feature_gen import 
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

    if True in NON_LLM_MODELS.values():
        features_df = get_features_df(
            outflows_with_memo, num_tfidf_features, include_date_features, include_amount_features
        )

        X_train, y_train, X_test, y_test = train_test_split_features(features_df)

    if train_bert:
        
    

    ## train, test = output(feature_gen.py)
    ## model, ... = output(train_models.py(train, test))
