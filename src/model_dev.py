import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    @abstractmethod
    def train(self,Xtrain,ytrain):
        pass
class LinearregressionModel(Model):
    def train(self,Xtrain,ytrain,**kwargs):
        try:
            reg = LinearRegression()
            reg.fit(Xtrain,ytrain)
            logging.info("success!")
            return reg
        except Exception as e:
            logging.error("error!")
            raise e