import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

def train_model(X_train, X_test, y_train, y_test, feature_cols: list, target_cols: list):

    rf_clf = RandomForestClassifier(n_estimators=500, random_state=10)
    rf_clf.fit(X_train, y_train)
    
    y_train_predicted = rf_clf.predict(X_train)
    y_train_pred_proba = np.max(rf_clf.predict_proba(X_train), axis=1)
    X_train['train_pred'] = y_train_predicted
    
    y_test_predicted = rf_clf.predict(X_test)
    y_test_pred_proba = np.max(rf_clf.predict_proba(X_test), axis=1)
    X_test['test_pred'] = y_test_predicted
    
    return(y_train, y_test, y_train_pred_proba, y_test_pred_proba)