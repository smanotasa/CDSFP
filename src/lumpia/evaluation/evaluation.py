from sklearn.metrics import roc_auc_score

def get_roc_auc_score(trained: tuple):
    y_train, y_test, y_train_pred_proba, y_test_pred_proba = trained
    
    # obtain scores
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    return train_auc, test_auc