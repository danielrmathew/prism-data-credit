
def predict_bert(pipe, X):
    """
    Makes predictions using a finetuned DistilBert model

    Args:
        pipe (transformers.TextClassificationPipeline): DistilBert model pipeline for inference
        X (list): List of memos

    Returns:
        list: Category predictions
    """

    preds = pipe(X)
    preds = [dct['label'] for dct in preds]
    
    return preds

def predict_fasttext(model, X):
    """
    Makes predictions using a finetuned DistilBert model

    Args:
        pipe (fasttext): fasttext model object
        X (list): List of memos

    Returns:
        list: Category predictions
    """
    
    preds = []
    for text in X:
        preds.append(model.predict(text)[0][0])

    return preds
