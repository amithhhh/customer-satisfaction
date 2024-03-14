import logging
import pandas as pd
from zenml import step 
from src.model_dev import LinearregressionModel
from sklearn.base import RegressorMixin

@step
def train_model(
    Xtrain:pd.DataFrame,
    Xtest:pd.DataFrame,
    ytrain:pd.DataFrame,
    ytest:pd.DataFrame,
)->RegressorMixin:
    model = None