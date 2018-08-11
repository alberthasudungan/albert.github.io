

import numpy as np
import os
import pandas as pd
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

#import the libraries 
import numpy as np 
import os 
import pandas as pd 
import tarfile 
from six.moves import urllib

#download sourcing file 
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
DOWNLOAD_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#defining the function to extract file '
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()

#screening the data 
housing =load_housing_data()
housing.head()

#quick look of data description 
housing.info()


#in ocean proximity field looking for categories that exist and how many districts belong to each category
housing["ocean_proximity"].value_counts()


#description of the dataset 
housing.describe()
#importing matplotlib 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
#creating housing histogram 
housing.hist(bins =50,figsize=(20,15))
plt.show
import numpy as np
def sp_test(data, test_ratio):
    s_indiced = np.random.permutation(len(data))
    ts_size = int(len(data)*test_ratio)
    ts_index = s_indiced[:ts_size]
    tr_index =s_indiced[ts_size:]
    return data.iloc[tr_index], data.iloc[ts_index]

print("test")


tr_set, ts_set = sp_test(housing,0.2)
print(len(tr_set),"train +", len(ts_set),"test")
#print(len(train_set), "train +", len(test_set), "test")
#identifier testing 
import hashlib 
def ts_check(identifier, test_ratio, hash): 
    return hash(np.int64(identifier)).digest()[-1]<256*test_ratio
#function to test data set 
def split_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids =data[id_column]
    in_ts_set = ids.apply(lambda id_:ts_check(id_,test_ratio, hash))
    return data.loc[~in_ts_set], data.loc[in_ts_set]
housing_with_id = housing.reset_index()
tr_set, ts_set = split_id(housing_with_id, 0.2,"index")
#simulation of training from housing data 
housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]
tr_set, ts_set = split_id(housing_with_id, 0.2,"id")
#simplest model of import train_test_split 
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2,random_state =42)
#modeling
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace =True) 
#stratifying the population based median income 
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state =42)
for train_index, test_index in split.split(housing, housing["income_cat"]): 
    strat_train_set = housing.loc[train_index]
    strat_test_set  = housing.loc[test_index]
#simulating the model 
strat_test_set["income_cat"].value_counts()//len(strat_test_set)

strat_train_set = housing.loc[train_index]
strat_test_set  = housing.loc[test_index]
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#visualising the data
housing.plot(kind="scatter", x="longitude", y="latitude")
#to distinct house based on low and high density population 
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
#calcatling correlation matrix between median house value with other variables
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

#visualising correlation matrix 
from pandas.plotting import scatter_matrix 
attributes = ["median_house_value","median_income","total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize = (12,8))
#Copying and multiplying the original data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
#machine learning for data cleaning (changing missing value with median value)
from sklearn.preprocessing import Imputer 
#dropping ocean proximity 
housing_num = housing.drop("ocean_proximity", axis=1)
#fit the training 
input = Imputer(strategy="median")
input.fit(housing_num)

#calculating median and store in statistics instant variable 
input.statistics_

#display in the DataFrame 
X=input.transform(housing_num)
housing_tr=pd.DataFrame(X, columns = housing_num.columns)


housing_tr



#checking any missing values 
print(housing_tr.isnull().sum())


from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()


#building pipeline for preprocessing the numerical attribute
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)



housing_num_tr





from future_encoders import OneHotEncoder
from future_encoders import ColumnTransformer


# In[126]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)



housing_prepared




housing_prepared




#testing the fitness of the model 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)




# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))





print("Labels:", list(some_labels))




some_data_prepared





# training regression using RMSE
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)





#executing linear RMSE regression 
lin_rmse





#getting the better predicition using decision tree regressor 
from sklearn.tree import DecisionTreeRegressor 

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[136]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse





from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)




#randomregressor 
from sklearn.ensemble import RandomForestRegressor 
forest_reg= RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)




housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)





forest_rmse







