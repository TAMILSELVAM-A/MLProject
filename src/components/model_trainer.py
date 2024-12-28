import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig():
    trained_model_file_path : str = os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("SPLITTING TRAINING AND TEST INPUT DATA")

            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Linear Regression" : LinearRegression(),
                "CataBoost Regressor" : CatBoostRegressor(verbose=False),
                "XGBoost Regressor" : XGBRegressor(),
                "K-Neighnors Regressor" : KNeighborsRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "AdaBoosting Regressor" : AdaBoostRegressor(),
            }

            params = {
                "Random Forest" :{
                    'n_estimators' : [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighnors Regressor":{
                    'n_neighbors' : [5,7,9,11]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    "subsample":[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "XGBoost Regressor":{
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'n_estimators':[8,16,32,64,128,256]
                },
                "CataBoost Regressor":{
                    # 'depth':[6,8,10],
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'iterations':[30,50,100]
                },
                "AdaBoosting Regressor":{
                    #  'learning_rate':[.1,.01,.05,.001],
                    # 'n_estimators':[8,16,32,64,128,256]
                } 

            }

            model_report : dict = evaluate_model(X_train=x_train,Y_train=y_train,
                                                 X_test = x_test,Y_test = y_test,
                                                 models=models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                raise CustomException("No Best Model Found")
            
            logging.info('Best Model score found on both training and testing dataset')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted = best_model.predict(x_test)

            r2_score_value = r2_score(y_test,predicted)

            return r2_score_value

        except Exception as e:
           raise CustomException(e,sys)