from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from ClassifierWrapper import ClassifierWrapper


def get_classifier(method, n_estimators, max_features=None, gbm_learning_rate=None, max_depth=None, random_state=None, min_cases_for_training=30, learning_rate=None, subsample=None, colsample_bytree=None, min_child_weight=None):

    if method == "rf":
        return ClassifierWrapper(
            cls=RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                       random_state=random_state), min_cases_for_training=min_cases_for_training)
               
    elif method == "gbm":
        return ClassifierWrapper(
            cls=GradientBoostingClassifier(n_estimators=n_estimators, max_features=max_features, learning_rate=gbm_learning_rate, random_state=random_state), 
            min_cases_for_training=min_cases_for_training)
    elif method == "dt":
        return ClassifierWrapper(
            cls=DecisionTreeClassifier(random_state=random_state), 
            min_cases_for_training=min_cases_for_training)
    
    elif method == "xgboost":
        return ClassifierWrapper(cls=xgb.XGBClassifier(objective='binary:logistic',
                                 n_estimators=n_estimators,
                                 learning_rate= learning_rate,
                                 subsample=subsample,
                                 max_depth=max_depth,
                                 colsample_bytree=colsample_bytree,
                                 min_child_weight=min_child_weight),
                                 min_cases_for_training=min_cases_for_training)
    
    else:
        print("Invalid classifier type")
        return None