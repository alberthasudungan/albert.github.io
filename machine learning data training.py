#importing visualised libraries 
import pandas as pd 
import numpy as np 
import seaborn as sns

#reading the data 
data1 = pd.read_csv('housing.csv')

#checking any missing value 
data1.isnull().sum()

#converting variables into array 
np.array([data1.ZSMHC])
np.array([data1.ZINC2])
np.array([data1.BEDRMS])

#defining logistic regression model 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#defining independent predictors 
predictors = ['ZINC2','BEDRMS']

#setting independent and dependent (Y) variable 
X_data1 = data1[predictors].values
Y_data1 = data1['ZSMHC'].values

#making model fit 
model.fit(X_data1, Y_data1)

#predicting the rent housing in further 10 period ahead 
y_predict = model.predict(X_data1)
y_predict[:10]
