import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, OrdinalEncoder


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
   
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
         
         '''
         This function is responsible for data transformation
        
         '''
         try:

            numerical_columns=['age', 'bmi', 'children']
            oh_features=['sex']
            od_features=['smoker','region']
            # Define the custom ranking forordinal variable
            smoker_map=['no','yes']
            region_map=['northwest','southwest','northeast','southeast']
            logging.info(f"Categorical columns for one hot encoding: {oh_features}")
            logging.info(f"Categorical columns for ordinal encoding: {od_features}")
            logging.info(f"Numerical columns: {numerical_columns}")



            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())])
            # Categorigal Pipeline
            cat_pipeline1=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[smoker_map,region_map])),
                    ('scaler',StandardScaler(with_mean=False))
                    ])
            cat_pipeline2=Pipeline(
                steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))])
                        


            preprocessor=ColumnTransformer([('num_pipeline',num_pipeline,numerical_columns),
            ('cat_pipeline1',cat_pipeline1,od_features),
            ('cat_pipeline2',cat_pipeline2,oh_features)])

            logging.info("column transformation is done")

            return preprocessor


         except Exception as e:

            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
          
          
          try:
                
                
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)
                logging.info("Read train and test data completed")

                logging.info("obtaining preprocessing object")

                preprocessing_obj=self.get_data_transformer_object()
                target_column_name="expenses"
                numerical_columns =['age', 'bmi', 'children']
                input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
                target_feature_train_df=train_df[target_column_name]
                # Convert the series to a DataFrame with a single column
                target_feature_train_df = target_feature_train_df.to_frame()

                # Access the underlying numpy array and reshape it
                target_feature_train_df = target_feature_train_df.values.reshape(-1, 1)
                # print(target_feature_train_df.shape)
                input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
                target_feature_test_df=test_df[target_column_name]
                # Convert the series to a DataFrame with a single column
                target_feature_test_df = target_feature_test_df.to_frame()

                # Access the underlying numpy array and reshape it
                target_feature_test_df = target_feature_test_df.values.reshape(-1, 1)
                logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]

                # print(train_arr.shape)
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
                # print(test_arr.shape)

                logging.info(f"Saved preprocessing object.")

                save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)

                return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
                )
          except Exception as e:
            raise CustomException(e,sys)

        