## imports config and relevant functions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.build.feature_gen import (
    read_outflows, clean_memos, get_features_df,
    dataset_split, train_test_split_features
)
from src.build.feature_gen_llm import encode_labels, train_test_split_llm, prepare_fasttext_data
from src.build.train_traditional import fit_model
from src.build.train_llm import fit_bert, fit_fasttext
from src.build.evaluate_models import make_confusion_matrix 
from src.build.predict_traditional import predict
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    print(config)

    #################### 
    #### GET CONFIG ####
    ####################

    # features config
    OUTFLOWS_PATH = config['OUTFLOWS_PATH']
    num_tfidf_features = config['FEATURES']['num_tfidf_features']
    include_date_features = config['FEATURES']['include_date_features']
    include_amount_features = config['FEATURES']['include_amount_features']
    features_fp = config['FEATURES']['SAVE_FILEPATH']
    # non llm config
    NON_LLM_MODELS = config['MODELS']['NON_LLM']
    # llm config
    train_bert = config['MODELS']['LLM']['bert']
    bert_hp = config['MODELS']['LLM']['bert']['hyperparameters']
    train_fasttext = config['MODELS']['LLM']['fasttext']
    fasttext_ngrams = config['MODELS']['LLM']['fasttext']['hyperparameters']['ngrams']
    
    assert 1 == 2

    outflows_with_memo = read_outflows(OUTFLOWS_PATH)
    outflows_with_memo = clean_memos(outflows_with_memo)
    outflows_with_memo_train, outflows_with_memo_test = dataset_split(outflows_with_memo)

    if True in NON_LLM_MODELS.values():
        if features_fp is not None:
            features_fp = Path(features_fp)
            if features_fp.is_file():
                print('Existing features found, loading now')
                features_df = pd.read_pickle(features_fp)
            else:
                print(f'No file exists, will save to {features_fp}')
                features_df = get_features_df(
                    outflows_with_memo, num_tfidf_features, include_date_features, include_amount_features
                )
                features_df.to_pickle(features_fp)
        else:
            features_df = get_features_df(
                outflows_with_memo, num_tfidf_features, include_date_features, include_amount_features
            )

        X_train, X_test, y_train, y_test = train_test_split_features(features_df)

        models = {}
        
        for model, train_model in NON_LLM_MODELS.items():
            if train_model:
                # train models
                print(f"Training {model}")
                model_instance = fit_model(X_train, y_train, X_test, y_test, model)
                models[model] = model_instance # saving model object to dict
                print(f"{model} training done, saving to result/{model}.pkl")
                with open(f'result/models/{model}.pkl', 'wb') as f:
                    pickle.dump(model_instance, f)
                
                # predict
                print(f"Making {model} train and test inferences")
                train_preds = predict(X_train, y_train, model_instance)
                test_preds = predict(X_test, y_test, model_instance)

                # evaluate (classification report (TODO), roc curves (TODO))
                print(f"Creating {model} confusion matrices")
                make_confusion_matrix(y_train, train_preds, model, train=True)
                make_confusion_matrix(y_test, test_preds, model, train=False)

    if train_bert or train_fasttext:
        # generate llm features -- labels are encoded to whole numbers
        outflows_with_memos_encoded, id2label, label2id = encode_labels(outflows_with_memos)
        # pre_tokenized_split is train and test set with the memos not tokenized, train_dataset and test_dataset are torch.Datasets with memos tokenized
        pre_tokenized_split, (train_dataset, test_dataset) = train_test_split_llm(outflows_with_memos_encoded)
        X_train_llm, X_test_llm, y_train_llm, y_test_llm = pre_tokenized_split
        # decoded categories back to their real labels
        y_train_llm_cats = [id2label[cat_id] for cat_id in y_train_llm]
        y_test_llm_cats = [id2label[cat_id] for cat_id in y_test_llm]
        
        
    if train_bert:
        # train bert model
        print("Training DistilBert model")
        pipe = fit_bert(train_dataset, test_dataset, id2label, label2id, bert_hp)
        print("DistilBert training done, saving to result/models/bert")
        
        # predict
        print("Making DistilBert train and test inferences")
        train_preds_bert = predict_bert(pipe, X_train_llm)
        test_preds_bert = predict_bert(pipe, X_test_llm)

        # evaluate (TODO: classification report)
        print("Creating DistilBert confusion matrices")
        make_confusion_matrix(np.array(y_train_llm), np.array(train_preds_bert), 'bert', train=True)
        make_confusion_matrix(np.array(y_test_llm), np.array(test_preds_bert), 'bert', train=False)
        
    if train_fasttext:
        # generate fasttext train file
        print("Creating fastText features")
        fasttext_train_fp, fasttext_test_fp = prepare_fasttext_data(outflows_with_memo)
        
        # train fasttext
        print("Training fastText model")
        fasttext_model = fit_fasttext(fasttext_train_fp, ngrams=fasttext_ngrams) # hyperparameter config
        print("fastText training done, saving to result/models/fasttext.bin")
        fasttext_model.save_model('result/models/fasttext.bin')
        
        # predict
        print("Making fastText train and test inferences")
        train_preds_fastext = predict_fasttext(fasttext_model, X_train_llm)
        test_preds_fasttext = predict_fasttext(fasttext_model, X_test_llm)

        # evaluate (TODO: classification report, TODO: roc auc curves)
        # acc, fpr, tpr, roc_auc = output_metrics_fasttext(fasttext_train_fp, fasttext_test_fp, fasttext_model)
        make_confusion_matrix(np.array(y_train_llm_cats), np.array(train_preds_fastext), 'fasttext', train=True)
        make_confusion_matrix(np.array(y_test_llm_cats), np.array(test_preds_fasttext), 'fasttext', train=False)


    # TODO: make final dataframe/csv of metrics of all models



    
