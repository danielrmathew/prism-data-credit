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
from src.build.evaluate_models import make_confusion_matrix, make_classification_report_csv, roc_score_curve, output_metrics_fasttext
from src.build.predict_traditional import predict
from src.build.predict_llm import predict_bert, predict_fasttext
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    #################### 
    #### GET CONFIG ####
    ####################

    # features config
    OUTFLOWS_PATH = config['OUTFLOWS_PATH']
    num_tfidf_features = config['FEATURES']['num_tfidf_features']
    include_date_features = config['FEATURES']['include_date_features']
    include_amount_features = config['FEATURES']['include_amount_features']
    features_fp = config['FEATURES']['SAVE_FILEPATH']
    train_test_fp = config['FEATURES']['SAVE_TRAIN_TEST_SPLIT']
    # non llm config
    non_llm_models = config['MODELS']['NON_LLM']
    train_non_llm_models = [dct['train'] for dct in non_llm_models.values()]
    predict_non_llm_models = [dct['predict'] for dct in non_llm_models.values()]
    # llm config
    train_bert = config['MODELS']['LLM']['bert']['train']
    train_fasttext = config['MODELS']['LLM']['fasttext']['train']
    bert_hp = config['MODELS']['LLM']['bert']['hyperparameters']
    train_fasttext = config['MODELS']['LLM']['fasttext']['train']
    fasttext_ngrams = config['MODELS']['LLM']['fasttext']['hyperparameters']['ngrams']

    print("Reading data...")
    outflows_with_memo = read_outflows(OUTFLOWS_PATH)
    print("Cleaning memo field...")
    outflows_with_memo = clean_memos(outflows_with_memo)
    outflows_with_memo_train, outflows_with_memo_test = dataset_split(outflows_with_memo)

    if True in train_non_llm_models or True in predict_non_llm_models:
        if features_fp is not None:
            features_fp = Path(features_fp)
            if features_fp.exists():
                print('Existing features found, loading now...')
                features_df = pd.read_pickle(features_fp)
            else:
                print(f'No features file exists, will save features data to {features_fp}')
                features_df = get_features_df(
                    outflows_with_memo, num_tfidf_features, include_date_features, include_amount_features
                )
                print(features_df.head())
                print(features_df.shape)
                features_df.to_pickle(features_fp)
        else:
            features_df = get_features_df(
                outflows_with_memo, num_tfidf_features, include_date_features, include_amount_features
            )

        print("Splitting data into train and test sets. If no splits are saved, this may take a few minutes...")
        if train_test_fp is not None:
            train_test_fp = Path(train_test_fp)
            if train_test_fp.exists() and train_test_fp.is_dir():
                print("Existing train test split found, loading now...")
                X_train = pd.read_pickle(train_test_fp / "X_train.pkl")
                X_test = pd.read_pickle(train_test_fp / "X_test.pkl")
                y_train = pd.read_pickle(train_test_fp / "y_train.pkl")
                y_test = pd.read_pickle(train_test_fp / "y_test.pkl")
            else:
                print(f'No train test spilt exists, will save train test split data to {train_test_fp}')
                X_train, X_test, y_train, y_test = train_test_split_features(features_df)
                train_test_fp.mkdir(parents=True, exist_ok=True)
                X_train.to_pickle(train_test_fp / "X_train.pkl")
                X_test.to_pickle(train_test_fp / "X_test.pkl")
                y_train.to_pickle(train_test_fp / "y_train.pkl")
                y_test.to_pickle(train_test_fp / "y_test.pkl")
        else:
            X_train, X_test, y_train, y_test = train_test_split_features(features_df)

        models = {}
        
        for model_type, train_model, predict_model in zip(non_llm_models.keys(), train_non_llm_models, predict_non_llm_models):
            model_path = Path(f'result/models/{model_type}.pkl')
            if train_model:
                # train models
                print(f"Training {model_type}...")
                model_instance = fit_model(X_train, y_train, X_test, y_test, model_type)
                models[model_type] = model_instance # saving model object to dict
                
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_path, 'wb') as f:
                    pickle.dump(model_instance, f)
                print(f"{model_type} training done, saved to result/{model_type}.pkl")
            elif not train_model and model_path.exists():
                with open(model_path, 'rb') as f:
                    model_instance = pickle.load(f)
            
            if predict_model and model_path.exists():                    
                # predict
                print(f"Making {model_type} train and test inferences...")
                train_preds = predict(X_train, y_train, model_instance)
                test_preds = predict(X_test, y_test, model_instance)

                print(f"Creating {model_type} confusion matrices...")
                make_confusion_matrix(y_train, train_preds, model_type, train=True)
                make_confusion_matrix(y_test, test_preds, model_type, train=False)
                print(f"Saved {model_type} confusion matrices to result/{model_type}_confusion_matrix.png") 

                print(f"Creating {model_type} classifcation reports...")
                make_classification_report_csv(y_train, train_preds, model_type, train=True)
                make_classification_report_csv(y_test, test_preds, model_type, train=False)
                print(f"Saved {model_type} classifcation reports to result/{model_type}_metrics.csv") 

                print(f"Creating {model_type} ROC curves...")
                roc_score_curve(X_train, y_train, train_preds, model_instance, model_type, train=True)
                roc_score_curve(X_test, y_test, test_preds, model_instance, model_type, train=False)
                print(f"Saved {model_type} ROC curves to result/{model_type}_roc_auc_curve.png")
                
            elif predict_model and not model_path.exists():
                print(f"You have set predict to True for {model_type} with no model trained. Skipping...")
                continue

    
    if train_bert or train_fasttext:
        # generate llm features -- labels are encoded to whole numbers
        outflows_with_memo_encoded, id2label, label2id = encode_labels(outflows_with_memo)
        # pre_tokenized_split is train and test set with the memos not tokenized, train_dataset and test_dataset are torch.Datasets with memos tokenized
        pre_tokenized_split, (train_dataset, test_dataset) = train_test_split_llm(outflows_with_memo_encoded)
        X_train_llm, X_test_llm, y_train_llm, y_test_llm = pre_tokenized_split
        # decoded categories back to their real labels
        y_train_llm_cats = [id2label[cat_id] for cat_id in y_train_llm]
        y_test_llm_cats = [id2label[cat_id] for cat_id in y_test_llm]
        
    if train_bert:
        # train bert model_type
        print("Training DistilBert model_type...")
        pipe = fit_bert(train_dataset, test_dataset, id2label, label2id, bert_hp)
        print("DistilBert training done, saving to result/models/bert")
        
        # predict
        print("Making DistilBert train and test inferences...")
        train_preds_bert = predict_bert(pipe, X_train_llm)
        test_preds_bert = predict_bert(pipe, X_test_llm)

        # evaluate (TODO: ROC if time)
        print("Creating DistilBert confusion matrices...")
        make_confusion_matrix(np.array(y_train_llm), np.array(train_preds_bert), 'bert', train=True)
        make_confusion_matrix(np.array(y_test_llm), np.array(test_preds_bert), 'bert', train=False)
        print("Saved DistilBert confusion matrices to result/bert_confusion_matrix.png") 

        print("Creating DistilBert classifcation reports...")
        make_classification_report_csv(y_train, train_preds, model_type, train=True)
        make_classification_report_csv(y_test, test_preds, model_type, train=False)
        print("Saved DistilBert classifcation reports to result/bert_metrics.csv") 
    
        
    if train_fasttext:
        # generate fasttext train file
        print("Creating fastText features")
        fasttext_train_fp, fasttext_test_fp = prepare_fasttext_data(outflows_with_memo)
        
        # train fasttext
        print("Training fastText model_type")
        fasttext_model = fit_fasttext(fasttext_train_fp, ngrams=fasttext_ngrams) 
        print("fastText training done, saving to result/models/fasttext.bin")
        fasttext_model.save_model('result/models/fasttext.bin')
        
        # predict
        print("Making fastText train and test inferences")
        train_preds_fastext = predict_fasttext(fasttext_model, X_train_llm)
        test_preds_fasttext = predict_fasttext(fasttext_model, X_test_llm)

        # evaluate (all evaluation done in this function)
        acc, fpr, tpr, roc_auc = output_metrics_fasttext(fasttext_train_fp, fasttext_test_fp, fasttext_model)
        
        
        

    
