from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.model_train import train_model

@pipeline(enable_cache=True)
def training_pipeline(data_path:str):
    df = ingest_data(data_path)
    Xtrain,Xtest,ytrain,ytest = clean_data(df)
    model = train_model(Xtrain,Xtest,ytrain,ytest)
    mse,r2 = evaluate_model(model,Xtest,ytest)
