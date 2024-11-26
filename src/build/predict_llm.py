

def predict_bert(pipe, X):

    preds = pipe(X)
    preds = [dct['label'] for dct in preds]

    # make_confusion_matrix(y_train, train_preds, model_type, train=True)
    # make_confusion_matrix(y_test, test_preds, model_type, train=False)

    return preds

def predict_fasttext(model, X):
    preds = []
    for i in range(len(X)):
        preds.append(model.predict(X[i])[0][0])

    return preds

    # train_acc = (np.array(train_preds) == np.array(y_train)).mean()
    # test_acc = (np.array(test_preds) == np.array(y_test)).mean()
