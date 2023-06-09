import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            logging.info("done prediction")
            return pred
            logging.info("returned pred value")

            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)




class CustomData:
    def __init__(self,
    age:int,
    sex:str,
    bmi:float,
    children:int,
    smoker:str,
    region:str):
    
        
        self.age=age
        self.sex=sex
        self.bmi=bmi
        self.children=children
        
        self.smoker=smoker
        self.region=region
        
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'sex':[self.sex],
                'bmi':[self.bmi],
                'children':[self.children],
                
                'smoker':[self.smoker],
                'region':[self.region]

                }

            df = pd.DataFrame(custom_data_input_dict)
           
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)