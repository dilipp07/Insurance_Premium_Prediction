import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_injestion import DataInjestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

##run data

if __name__=="__main__":
    try:

        obj=DataInjestion()
        train_data_path,test_data_path=obj.initate_data_injestion()
        
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
        model_trainer=ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr,test_arr)
    
    except Exception as e :
        logging.info("error occured in trainering_pipeline")
        raise CustomException(e,sys)