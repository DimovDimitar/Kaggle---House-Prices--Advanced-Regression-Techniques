# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:13:55 2020

@author: dimit
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
color = sns.color_palette()

# ----------------------------------------------------------------- #
# import the data

test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')

# ----------------------------------------------------------------- #
# preprocessing. I manually went through the variables and converted
# those ones that I felt are relevant when predicting a house price in reality.

train_data = train_data.drop(['Id'], axis=1)
test_data = test_data.drop(['Id'], axis=1)

train_data = train_data.drop(['Alley', 'FireplaceQu', 'PoolQC',
                              'Fence', 'MiscFeature'], axis=1)
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

central_air_dic = {"N" : 0, "Y" : 1}

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

plt.figure(figsize=(40,20))
sb.set(font_scale=1.5)
sb.boxplot(x='YearBuilt', y="SalePrice", data=train)
sb.swarmplot(x='YearBuilt', y="SalePrice", data=train, color=".25")
plt.xticks(weight='bold',rotation=90)

train_test=pd.concat([train_data,test_data],axis=0,sort=False)
train_test.loc[train_test['Fireplaces']==0,'FireplaceQu']='Nothing'
train_test['LotFrontage'] = train_test['LotFrontage'].fillna(train_test.groupby('1stFlrSF')['LotFrontage'].transform('mean'))
train_test['LotFrontage'].interpolate(method='linear',inplace=True)
train_test['LotFrontage']=train_test['LotFrontage'].astype(int)
train_test['MasVnrArea'] = train_test['MasVnrArea'].fillna(train_test.groupby('MasVnrType')['MasVnrArea'].transform('mean'))
train_test['MasVnrArea'].interpolate(method='linear',inplace=True)
train_test['MasVnrArea']=train_test['MasVnrArea'].astype(int)


train_test.loc[train_test['BsmtFinSF1']==0,'BsmtFinType1']='Unf'
train_test.loc[train_test['BsmtFinSF2']==0,'BsmtQual']='TA'
train_test['YrBltRmd']=train_test['YearBuilt']+train_test['YearRemodAdd']
train_test['Total_Square_Feet'] = (train_test['BsmtFinSF1'] + train_test['BsmtFinSF2'] + train_test['1stFlrSF'] + train_test['2ndFlrSF'] + train_test['TotalBsmtSF'])
train_test['Total_Bath'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) + train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))
train_test['Total_Porch_Area'] = (train_test['OpenPorchSF'] + train_test['3SsnPorch'] + train_test['EnclosedPorch'] + train_test['ScreenPorch'] + train_test['WoodDeckSF'])
train_test['exists_pool'] = train_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['exists_garage'] = train_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train_test['exists_fireplace'] = train_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train_test['exists_bsmt'] = train_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train_test['old_house'] = train_test['YearBuilt'].apply(lambda x: 1 if x <1990 else 0)

for i in train_test.columns:
    if 'SalePrice' not in i:
        if 'object' in str(train_test[str(i)].dtype):
            train_test[str(i)]=train_test[str(i)].fillna(method='ffill')
train_test.info()

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import numpy as np 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, KFold,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler,RobustScaler,LabelEncoder,PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor,StackingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

numeric_features = train_test.dtypes[train_test.dtypes != "object"].index
skewed_features = train_test[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
print(skewed_features)

high_skewness = skewed_features[abs(skewed_features) > 0.9]
skewed_features = high_skewness.index

for feature in skewed_features:
    train_test[feature] = boxcox1p(train_test[feature], boxcox_normmax(train_test[feature] + 1))

train_test.head()

train=train_test[0:1460]
test=train_test[1460:2919]
train_test.info()
len(train)
train.interpolate(method='linear',inplace=True)
test.interpolate(method='linear',inplace=True)
corr_new_train=train.corr()
plt.figure(figsize=(5,15))
sb.heatmap(corr_new_train[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(30),annot_kws={"size": 16},vmin=-1, cmap='PiYG', annot=True)
sb.set(font_scale=2)

# ----------------------------------------------------------------- #
# Select best features based on correlation

corr_dict2=corr_new_train['SalePrice'].sort_values(ascending=False).to_dict()
corr_dict2

best_columns=[]
for key,value in corr_dict2.items():
    if ((value>=0.3175) & (value<0.9)) | (value<=-0.315):
        best_columns.append(key)
best_columns

# ----------------------------------------------------------------- #
# remove the outliers

train = train.drop(train[(train.OverallQual==10) & (train.SalePrice<200000)].index)
train = train.drop(train[(train.Total_Square_Feet>=10000) & (train.SalePrice<200000)].index)
train = train.drop(train[(train.GarageArea>1200) & (train.SalePrice<165000)].index)
train = train.drop(train[(train.Total_Bath.isin([5,6])) & (train.SalePrice<200000)].index)
train = train.drop(train[(train.TotRmsAbvGrd==10) & (train.SalePrice>700000)].index)
train = train.drop(train[(train.YearBuilt<1900) & (train.SalePrice>250000)].index)
train = train.drop(train[(train.YearBuilt>2000) & (train.SalePrice<100000)].index)
train = train.drop(train[(train.YearRemodAdd<1970) & (train.SalePrice>350000)].index)
train = train.drop(train[(train.MasVnrArea>=1400) & (train.SalePrice<250000)].index)
train = train.drop(train[(train.GarageYrBlt<1960) & (train.SalePrice>340000)].index)
train = train.drop(train[(train.Total_Porch_Area>600) & (train.SalePrice<50000)].index)
train = train.drop(train[(train.LotFrontage>150) & (train.SalePrice<100000)].index)
train = train.drop(train[(train.GarageFinish.isin([1,2])) & (train.SalePrice>470000)].index)
train = train.drop(train[(train.old_house==0) & (train.SalePrice<100000)].index)
train = train.drop(train[(train.old_house==1) & (train.SalePrice>400000)].index)
train = train.drop(train[(train.KitchenQual==2) & (train.SalePrice>600000)].index)
train = train.drop(train[(train.KitchenQual==3) & (train.SalePrice>360000)].index)

train = train[train.GarageArea * train.GarageCars < 3700]
train = train[(train.FullBath + (train.HalfBath*0.5) + train.BsmtFullBath + (train.BsmtHalfBath*0.5))<5]

y_train = train_data["SalePrice"]    
train.isnull().sum()
test.isnull().sum()

del test['SalePrice']

train['SalePrice_Log1p'] = np.log1p(train.SalePrice)


X=train.drop(['SalePrice','SalePrice_Log1p'],axis=1)
y=train.SalePrice_Log1p


# ----------------------------------------------------------------- #
# Next step are: do some visualisation and imputation of missing values

col_names = list(train_data.columns.values)
for col in col_names:
    train_data[col] = train_data[col].fillna(train_data[col].mean())
    
def overfit_reducer(df):

    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99.9:
            overfit.append(i)
    overfit = list(overfit)
    return overfit
overfitted_features = overfit_reducer(X)

X.drop(overfitted_features,axis=1,inplace=True)
test.drop(overfitted_features,axis=1,inplace=True)
print('X.shape',X.shape)
print('test.shape',test.shape)


std_scaler=StandardScaler()
rbst_scaler=RobustScaler()
power_transformer=PowerTransformer()
X_std=std_scaler.fit_transform(X)
X_rbst=rbst_scaler.fit_transform(X)
X_pwr=power_transformer.fit_transform(X)

test_std=std_scaler.transform(test)
test_rbst=rbst_scaler.transform(test)
test_pwr=power_transformer.transform(test)

X_train,X_test,y_train,y_test=train_test_split(X_std,y,test_size=0.002,random_state=52)
print('X_train Shape :',X_train.shape)
print('X_test Shape :',X_test.shape)
print('y_train Shape :',y_train.shape)
print('y_test Shape :',y_test.shape)

gb_reg = GradientBoostingRegressor(n_estimators=1992, learning_rate=0.03005, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=14, loss='huber', random_state =42)
gb_reg.fit(X_train, y_train)
y_head=gb_reg.predict(X_test)
print('-'*10+'GBR'+'-'*10)
print('R square Accuracy: ',r2_score(y_test,y_head))
print('Mean Absolute Error Accuracy: ',mean_absolute_error(y_test,y_head))
print('Mean Squared Error Accuracy: ',mean_squared_error(y_test,y_head))

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
from sklearn.model_selection import KFold, cross_val_score
warnings.filterwarnings('ignore')

# Creating the models
models = [LinearRegression(), LogisticRegression(), SVR(), 
          SGDRegressor(), SGDRegressor(max_iter=1000, tol=1e-3),
          GradientBoostingRegressor(), 
          RandomForestRegressor(),
            Lasso(), Lasso(alpha=0.01, max_iter=10000), Ridge(), BayesianRidge(),
            KernelRidge(), KernelRidge(alpha=0.6, kernel='polynomial',degree=2, coef0=2.5),
             ElasticNet(), ElasticNet(alpha=0.001, max_iter=10000), ExtraTreesRegressor()]

names = ['Linear Regression','Logistic Regression', 
         'Support Vector Regression','Stochastic Gradient Descent',
         'Stochastic Gradient Descent 2','Gradient Boosting Tree',
         'Random Forest',
         'Lasso Regression','Lasso Regression 2','Ridge Regression',
         'Bayesian Ridge Regression','Kernel Ridge Regression',
         'Kernel Ridge Regression 2',
         'Elastic Net Regularization','Elastic Net Regularization 2',
         'Extra Trees Regression']

def rmse(model, X, y, cvv):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cvv))
    return rmse

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





