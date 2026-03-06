import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# 1.Load the dataset
Housing = pd.read_csv('housing.csv')

# 2. Create a stratified test set based on income category
Housing['Income_Cat'] = pd.cut(Housing['median_income'],
                               bins=[0,1.5,3.0,4.5,6,np.inf],labels=['A','B','C','D','E'])

Split = StratifiedShuffleSplit(n_splits=1,random_state=42,test_size=0.2)

for train,test in Split.split(Housing,Housing['Income_Cat']):
    Strat_Train_Set = Housing.loc[train].drop('Income_Cat',axis = 1)
    Strat_Test_Set = Housing.loc[test].drop('Income_Cat',axis = 1)
    
# 3.Work On Copy Of Training Set
Housing = Strat_Train_Set.copy()

# 4. Separate predictors and labels
Housing_Label = Housing['median_house_value'].copy()
Housing = Housing.drop('median_house_value',axis = 1)

# 5. Separate numerical and categorical columns
Num_Attributes = Housing.drop('ocean_proximity',axis = 1).columns.tolist()
Cat_Attributes = ['ocean_proximity']

# 6.Pipelines
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

# 7.Tranform The Data
Housing_Prepared = Full_Pipeline.fit_transform(Housing)

# housing_prepared is now a NumPy array ready for training
# print(Housing_Prepared.shape)

# Train The Model 

# Linear Regression
Lin_Reg = LinearRegression()
Lin_Reg.fit(Housing_Prepared,Housing_Label)
Lin_Pred = Lin_Reg.predict(Housing_Prepared) 

# Decision Tree Regressor
Des_Reg = DecisionTreeRegressor(random_state=42,max_depth=5)
Des_Reg.fit(Housing_Prepared,Housing_Label)
Des_Pred = Des_Reg.predict(Housing_Prepared)

# Random Forest
RandomForest_Reg = RandomForestRegressor()
RandomForest_Reg.fit(Housing_Prepared,Housing_Label)
RandomForest_Pred = RandomForest_Reg.predict(Housing_Prepared)

# Check RMSE Of Every Model
# Lin_RMSE = root_mean_squared_error(Lin_Pred,Housing_Label)
# Des_RMSE = root_mean_squared_error(Des_Pred,Housing_Label)
# RandomForest_RMSE = root_mean_squared_error(RandomForest_Pred,Housing_Label)
# Lin_RMSEs =-cross_val_score(Lin_Reg,Housing_Prepared,Housing_Label,scoring="neg_root_mean_squared_error",cv=10)
# Des_RMSEs =-cross_val_score(Des_Reg,Housing_Prepared,Housing_Label,scoring="neg_root_mean_squared_error",cv=10)
RandomForest_RMSEs =-cross_val_score(RandomForest_Reg,Housing_Prepared,Housing_Label,scoring="neg_root_mean_squared_error",cv=10)

# Print
# print(f"The RMSE Of Linear Regression Model Is:- {Lin_RMSE}")
# print(f"The RMSE Of Decison Tree Model Is:- {Des_RMSE}")
# print(f"The RMSE Of Random Forest Model Is:- {RandomForest_RMSE}")

# print(pd.Series(Lin_RMSEs).mean())
# print(pd.Series(Des_RMSEs).mean())
print(pd.Series(RandomForest_RMSEs).mean())


