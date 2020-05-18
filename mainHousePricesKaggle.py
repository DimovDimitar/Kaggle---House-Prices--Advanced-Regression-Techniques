# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:13:55 2020

@author: dimit
"""
import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
color = sns.color_palette()
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------- #
# import the data

test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')

# ----------------------------------------------------------------- #
# preprocessing. I manually went through the variables and converted
# those ones that I felt are relevant when predicting a house price in reality.

train_data = train_data.drop(['Id'], axis=1)
test_data = test_data.drop(['Id'], axis=1)

train_data = train_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
test_data = test_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

train_data.describe()

ms_zone_dic = {"A" : 0.1, "C" : 0.2, "FV" : 0.3, 
               "I" : 0.4, "RH" : 0.5, "RL" : 0.6, "RP" : 0.7, "RM": 0.8}

train_data["MSZoning"] = train_data["MSZoning"].map(ms_zone_dic)
test_data["MSZoning"] = test_data["MSZoning"].map(ms_zone_dic)

landc_dic = {"Lvl" : 0.1, "Bnk" : 0.2, "HLS" : 0.3, "Low" : 0.4}
train_data["LandContour"] = train_data["LandContour"].map(landc_dic)
test_data["LandContour"] = test_data["LandContour"].map(landc_dic)
                                                                                                              
util_dic = {"AllPub" : 0.1, "NoSewr" : 0.2, "NoSeWa" : 0.3, "ELO" : 0.4}
train_data["Utilities"] = train_data["Utilities"].map(util_dic)
test_data["Utilities"] = test_data["Utilities"].map(util_dic)

condition_dic = {"Artery" : 0.1, "Feedr" : 0.2, 
                 "Norm" : 0.3, "RRNn" : 0.4, "RRAn" : 0.5, 
                 "PosN" : 0.6, "PosA" : 0.7, "RRNe" : 0.8, "RRAe" : 0.9 }

train_data["Condition1"] = train_data["Condition1"].map(condition_dic)
train_data["Condition2"] = train_data["Condition2"].map(condition_dic)
test_data["Condition1"] = test_data["Condition1"].map(condition_dic)
test_data["Condition2"] = test_data["Condition2"].map(condition_dic)

house_style_dic = {"1Story" : 0.1, "1.5Fin" : 0.2, 
                   "1.5Unf" : 0.3, "2Story" : 0.4, "2.5Fin" : 0.5, 
                   "2.5Unf" : 0.6, "SFoyer" : 0.7, "SLvl" : 0.8}

train_data["HouseStyle"] = train_data["HouseStyle"].map(house_style_dic)
test_data["HouseStyle"] = test_data["HouseStyle"].map(house_style_dic)

exterior_dic = { "AsbShng" : 0.1, 
       "AsphShn" : 0.2, "BrkComm": 0.3, "BrkFace" : 0.4,
       "CBlock"	: 0.5,"CemntBd": 0.6,
       "HdBoard": 0.7,       "ImStucc": 0.8,
       "MetalSd": 0.9,       "Other"	: 1.0,
       "Plywood": 1.1,       "PreCast": 1.2,
       "Stone"	: 1.2,       "Stucco"	: 1.3,
       "VinylSd": 1.4,      "Wd Sdng": 1.5,       "WdShing": 1.6}

train_data["Exterior1st"] = train_data["Exterior1st"].map(exterior_dic)
test_data["Exterior1st"] = test_data["Exterior1st"].map(exterior_dic)
train_data["Exterior2nd"] = train_data["Exterior2nd"].map(exterior_dic)
test_data["Exterior2nd"] = test_data["Exterior2nd"].map(exterior_dic)

heating_dic = {"Floor":0.1,       "GasA":0.2,
       "GasW":0.3,       "Grav":0.4,
       "OthW":0.5,       "Wall":0.6}

heating_qual_dic = {  "Ex":0.1,
      "Gd":0.2,
       "TA":0.3,
       "Fa":0.4,
       "Po":0.5,
       "NA": 0.6}

central_air_dic = {"N" : 0.1, "Y" : 0.2}

train_data["Heating"] = train_data["Heating"].map(heating_dic)
test_data["Heating"] = test_data["Heating"].map(heating_dic)
train_data["HeatingQC"] = train_data["HeatingQC"].map(heating_qual_dic)
test_data["HeatingQC"] = test_data["HeatingQC"].map(heating_qual_dic)
train_data["GarageQual"] = train_data["GarageQual"].map(heating_qual_dic)
test_data["GarageQual"] = test_data["GarageQual"].map(heating_qual_dic)
train_data["GarageCond"] = train_data["GarageCond"].map(heating_qual_dic)
test_data["GarageCond"] = test_data["GarageCond"].map(heating_qual_dic)

sale_condition_dic = {       "Normal":0.1,
       "Abnorml":0.2,
       "AdjLand":0.3,
       "Alloca":0.4,
       "Family":0.5,
       "Partial":0.6}
train_data["SaleCondition"] = train_data["SaleCondition"].map(sale_condition_dic)
test_data["SaleCondition"] = test_data["SaleCondition"].map(sale_condition_dic)

# ----------------------------------------------------------------- #
# I think this is enough. Now remove all string type columns

train_data = train_data.select_dtypes(exclude=["object"])
test_data = test_data.select_dtypes(exclude=["object"])

# ----------------------------------------------------------------- #
# Now check both dataframes: 

train_data.info()

#dtypes: float64(16), int64(34) = total of 50 cols
#memory usage: 570.4 KB

test_data.info()

#dtypes: float64(24), int64(25) = total of 49 cols (exluding the y)
#memory usage: 558.6 KB

# ----------------------------------------------------------------- #
# Next step are: do some visualisation and imputation of missing values

y_train = train_data["SalePrice"]    

col_names = list(train_data.columns.values)
for col in col_names:
    train_data[col] = train_data[col].fillna(train_data[col].mean())
    
# ----------------------------------------------------------------- #
# Time to merge the two

full_data = pd.concat([train_data, test_data], ignore_index = True)
full_data = full_data.drop(["SalePrice"], axis=1)
full_data.info()

y_train_transformed = np.log(y_train) 

# ----------------------------------------------------------------- #
# without scaling Y

plt.figure(figsize=(25,5))

ax1 = plt.subplot(1, 3, 1)
sns.distplot(y_train, kde=False, fit=st.norm)
plt.title('Normal', size = 15)

ax2 = plt.subplot(1, 3, 2)
sns.distplot(y_train, kde=False, fit=st.lognorm)
plt.title('Log Normal', size = 15)

ax3 = plt.subplot(1, 3, 3)
sns.distplot(y_train, kde=False, fit=st.johnsonsu)
plt.title('Johnson SU', size = 15)

plt.show()

# ----------------------------------------------------------------- #
# log scale Y

plt.figure(figsize=(15,5))

ax1 = plt.subplot(1, 2, 1)
sns.distplot(y_train)
plt.title('Before transformation', size=15)

ax2 = plt.subplot(1, 2, 2)
sns.distplot(y_train_transformed)
plt.title('After transformation', size=15)

plt.show()
full_data_copy = full_data.copy()
# ----------------------------------------------------------------- #
# Split the data to testing and training

X_train = full_data.head(int(len(full_data)*(80/100)))
X_test = full_data[2335:]

# ----------------------------------------------------------------- #
# Feature scaling - Robust scaler or MaxAbs? Should try both

from sklearn.preprocessing import MaxAbsScaler, RobustScaler

robust_scaler = RobustScaler()
max_abs_scaler = MaxAbsScaler()

X_train_scaled_r = robust_scaler.fit(X_train).transform(X_train)
X_test_scaled_r = robust_scaler.fit(X_test).transform(X_test)

X_train_scaled_ma = max_abs_scaler.fit(X_train).transform(X_train)
X_test_scaled_ma = max_abs_scaler.fit(X_test).transform(X_test)

#full_data2 = pd.concat([X_train, X_test])

# ----------------------------------------------------------------- #
# Next step - applying models. Split X-train and X-test. Check equal dimensions

#X_train_scaled = full_data2[:train_data.shape[0]]
#X_test_scaled = full_data2[train_data.shape[0]:]
X_train_scaled_r.shape, X_test_scaled_r.shape, y_train_transformed_r.shape
#((1460, 49), (1459, 49), (1460,))

from sklearn.linear_model import LinearRegression, LogisticRegression,
BayesianRidge, ElasticNet, Lasso, SGDRegressor, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor, 
GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR, SVR

# Creating the models
models = [LinearRegression(), LogisticRegression(), SVR(), SGDRegressor(), SGDRegressor(max_iter=1000, 
            tol=1e-3), GradientBoostingRegressor(), RandomForestRegressor(),
            Lasso(), Lasso(alpha=0.01, max_iter=10000), Ridge(), BayesianRidge(),
            KernelRidge(), KernelRidge(alpha=0.6, kernel='polynomial',degree=2, coef0=2.5),
             ElasticNet(), ElasticNet(alpha=0.001, max_iter=10000), ExtraTreesRegressor()]

names = ['Linear Regression','Logistic Regression', 'Support Vector Regression','Stochastic Gradient Descent','Stochastic Gradient Descent 2','Gradient Boosting Tree','Random Forest',
         'Lasso Regression','Lasso Regression 2','Ridge Regression','Bayesian Ridge Regression','Kernel Ridge Regression','Kernel Ridge Regression 2',
         'Elastic Net Regularization','Elastic Net Regularization 2','Extra Trees Regression']

def rmse(model, X, y, cvv):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cvv))
    return rmse

from sklearn.model_selection import KFold, cross_val_score
warnings.filterwarnings('ignore')

# Perform 5-folds cross-validation to evaluate the models 

def estimator_function(X, y, cvv):
    for model, name in zip(models, names):
        # Root mean square error
        score = rmse(model, X, y, cvv)
        print("- {}: Mean: {:.6f}, Std: {:4f}".format(name, score.mean(), score.std()))


# ----------------------------------------------------------------- #

# Good performing models:
    # Linear Regression
    # Gradient Boost Tree
    # Extra Trees Regression
    # Random Forest

# ----------------------------------------------------------------- #

# Final step - applying models and predictions. Save results
test_data = pd.read_csv('test.csv')
#model1 = LinearRegression()
model2 = GradientBoostingRegressor()
model2.fit(X_train_scaled_ma, y_train_transformed)

# Generate the predictions running the model in the test data
#predictions1 = np.exp(model2.predict(X_test_scaled_ma))
predictions2 = np.exp(model2.predict(X_test_scaled_ma))

# Create the output file 
output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions2})
output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")





