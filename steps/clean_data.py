import pandas as pd
import logging
from zenml import step
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreProcessingStrategy
from typing import Annotated,Tuple


@step
def clean_data(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,'Xtrain'],
    Annotated[pd.DataFrame,'Xtest'],
    Annotated[pd.DataFrame,'ytrain'],
    Annotated[pd.DataFrame,'ytest']
]:
    try:
        preprocess = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df,preprocess)
        processed_data = data_cleaning.handle_data()
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data,divide_strategy)
        Xtrain,Xtest,ytrain,ytest = data_cleaning.handle_data()
        logging.info("completed")
    except Exception as e:
        logging.error("error!")
        raise e

