import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Logistic Regression": LogisticRegression(max_iter=500),
            "KNN Regressor": KNeighborsRegressor(),
            "Support Vector Regressor": SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "XGBoost Regressor": XGBRegressor(),
            "CatBoost Regressor": CatBoostRegressor(verbose=False)
            }

            # Define hyperparameter grid
            params = {
                "Decision Tree Regressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': ['sqrt', 'log2'],
                },
                "Random Forest Regressor": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting Regressor": {
                    # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},  # No hyperparameters to tune
                "Ridge Regression": {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                "Lasso Regression": {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                "KNN Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "Support Vector Regressor": {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100]
                },
                "Logistic Regression": {
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'C': [0.1, 1, 10, 100]
                },
                "XGBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    # 'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"Best found model is {best_model}.")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
        )
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)