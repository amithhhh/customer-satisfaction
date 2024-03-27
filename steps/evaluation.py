import logging
from zenml import step
import pandas as pd
from src.evaluation import MSE,R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

@step
def evaluate_model(
    model:RegressorMixin,
    Xtest:pd.DataFrame,
    ytest:pd.DataFrame
)->Tuple[
    Annotated[float,"mse"],
    Annotated[float,"r2"],
    # Annotated[float,"rmse"]
]:
    try:
        prediction = model.predict(Xtest)
        mse_class = MSE()
        mse = mse_class.calculate_scores(ytest,prediction)
        r2_class = R2()
        r2 = r2_class.calculate_scores(ytest,prediction)
        # rmse_class = RMSE()
        # rmse = rmse_class.calculate_scores(ytest,prediction)
        print(f"mse: {mse}")
        print(f"r2: {r2}")
        return mse,r2
    except Exception as e:
        logging.error("error....!")
        raise e

    