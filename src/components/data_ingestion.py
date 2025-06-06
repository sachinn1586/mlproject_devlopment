
import os
import sys

from src.exceptionhandling import customExceptiom
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class dataIngestionConfig:
    train_data_path : str=os.path.join('artifacts','train.csv')
    test_data_path : str=os.path.join('artifacts','test.csv')
    raw_data_path : str=os.path.join('artifacts','raw.csv')
    
class dataIngestion:
    def __init__(self):
        self.ingestion_config=dataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion component or method")
        try:
            df=pd.read_csv("notebook/data/model_training_file.csv")
            logging.info("Read the dataset from logging info ")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train Test Initiated")
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=43)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion has be completed")
            return self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
            
            
        except Exception as e:
            raise customExceptiom(e,sys)
            

if __name__=="__main__":
    obj=dataIngestion()
    obj.initiate_data_ingestion()