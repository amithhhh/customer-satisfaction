import pandas as pd
import logging
from zenml import step

@step
def clean_data(df:pd.DataFrame)->pd.DataFrame:
    pass