import logging
from abc import abstractmethod,ABC
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass


class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating")
            mse = mean_squared_error(y_true,y_pred)
            logging.info("success...!")
            return mse
        except Exception as e:
            raise e

class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating")
            r2 = r2_score(y_true,y_pred)
            logging.info("success...!")
            return r2
        except Exception as e:
            raise e
        
# class RMSE(Evaluation):
#     def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
#         try:
#             logging.info("calculating")
#             rmse = r2_score(y_true,y_pred,squared=False)
#             logging.info("success...!")
#             return rmse
#         except Exception as e:
#             raise e