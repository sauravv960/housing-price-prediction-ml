import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def Build_Pipeline(Num_Attributes,Cat_Attributes):
    # Numerical Pipeline
    Num_Pipeline =  Pipeline([
        ('Imputer',SimpleImputer(strategy='median')),
        ('Scalar',StandardScaler()),
    ])

    # Categorical Pipeline
    Cat_Pipeline = Pipeline([
        ('Encoder',OneHotEncoder(sparse_output=False,handle_unknown='ignore')),
    ])

    # Full Pipeline
    Full_Pipeline = ColumnTransformer([
        ('Num',Num_Pipeline,Num_Attributes),
        ('Cat',Cat_Pipeline,Cat_Attributes),
    ])

    return Full_Pipeline

if not os.path.exists(MODEL_FILE):
    # Let's Train The Model
    # Load the dataset
    Housing = pd.read_csv('housing.csv')

    # Create a stratified test set based on income category
    Housing['Income_Cat'] = pd.cut(Housing['median_income'],
                                bins=[0,1.5,3.0,4.5,6,np.inf],labels=['A','B','C','D','E'])

    Split = StratifiedShuffleSplit(n_splits=1,random_state=42,test_size=0.2)

    for train,test in Split.split(Housing,Housing['Income_Cat']):
        Housing_Test_Data = Housing.loc[test].drop('Income_Cat',axis = 1)
        Housing = Housing.loc[train].drop('Income_Cat',axis = 1)

    Housing_Test_Data.to_csv('input_data.csv',index = False)
        
    Housing_Label = Housing['median_house_value'].copy()
    Housing_Features = Housing.drop('median_house_value',axis = 1)

    # 5. Separate numerical and categorical columns
    Num_Attributes = Housing_Features.drop('ocean_proximity',axis = 1).columns.tolist()
    Cat_Attributes = ['ocean_proximity']

    pipeline = Build_Pipeline(Num_Attributes,Cat_Attributes)
    Housing_Prepared = pipeline.fit_transform(Housing_Features)

    Model = RandomForestRegressor(random_state=42)
    Model.fit(Housing_Prepared,Housing_Prepared)

    joblib.dump(Model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)    
    print("Congrats! Model Is Trained")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    Input_Data = pd.read_csv('input_data.csv')
    Transform_Input_Data = pipeline.transform(Input_Data)
    Prediction = model.predict(Transform_Input_Data)
    Input_Data['median_house_value'] = Prediction

    Input_Data.to_csv('Output_Data.csv')
    print('Inference Completed! Your Prediction Is Save In Output_Data.csv.. Thankyou')




