import logging
import pandas as pd
from zenml import step 
from src.model_dev import LinearregressionModel
from sklearn.base import RegressorMixin
from src.model_dev import LinearregressionModel
from .config import ModelNameConfig

@step
def train_model(
    Xtrain:pd.DataFrame,
    Xtest:pd.DataFrame,
    ytrain:pd.DataFrame,
    ytest:pd.DataFrame,
    config: ModelNameConfig
)->RegressorMixin:
    model = None
    try:
        if config.model_name == "LinearRegression":
            model = LinearregressionModel()
            trained_model = model.train(Xtrain,ytrain)
            return trained_model
        else:
            raise ValueError("Error...!")
    except Exception as e:
        print(e)