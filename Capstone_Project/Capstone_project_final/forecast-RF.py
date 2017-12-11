
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


# In[2]:

energydata = pd.DataFrame.from_csv('../energydata_complete.csv', index_col=None, encoding="utf-8")
energydata["date"] = pd.to_datetime(energydata.date)
energydata["consumedE"] = energydata["Appliances"]+energydata["lights"]
energydata = energydata[(energydata.date>=pd.to_datetime("2016-01-12")) & (energydata.date<pd.to_datetime("2016-05-27"))]
energydata.describe()


# In[3]:

energydata.head(1)


# ### Number of features

# In[4]:

len(energydata.columns)


# # Change in granularity

# In[5]:

dfdate = energydata.set_index("date")
dfdate = dfdate.resample('H').mean()
dfdate = dfdate.drop(["Appliances","lights"], axis = 1)
dfdate.describe()


# In[6]:

dfdate.head(2)


# # Feature Reduction
# ## mutual_info

# In[18]:

midate = mutual_info_regression(dfdate.loc[:, dfdate.columns != 'consumedE'], dfdate.consumedE)
midate /= np.max(midate)
mean_midate = sum(midate)/len(midate)
mean_midate


# In[19]:

midatefeatures = pd.DataFrame(list(zip(dfdate.loc[:, dfdate.columns != 'consumedE'], midate)), columns = ["name", 'miCoefficient'])
midatefeatures[midatefeatures.miCoefficient > mean_midate]


# In[20]:

midatefeatures = midatefeatures[midatefeatures.miCoefficient < mean_midate]
dfdate = dfdate.drop(list(midatefeatures.name.values), axis = 1)
dfdate = np.log(dfdate)
dfdate.describe()


# ### Number of features

# In[21]:

len(dfdate.columns)


# here I have to load the forecast_energy.csv

# # Creating big matrices

# In[22]:

def RF_1timeseries(X,y):
    #model random forest
    model = RandomForestRegressor(n_estimators=50,criterion="mae",max_depth=4,random_state=1,n_jobs=-1)
    model.fit( X , y )
    
    newXarray = np.append( np.delete(X.iloc[-1].values,0), (y[-1]) )
    newXarray = np.reshape( newXarray, (1,len(newXarray)))
    
    newX = pd.DataFrame( newXarray, columns=X.columns.values )
    newX["date"] = pd.Series(pd.date_range(X.index[-1]+pd.to_timedelta(1, unit='h'), periods=1, freq='H'))
    newX = newX.set_index("date")
    
    newy = model.predict(newX)
    
    return(newX,newy)


# In[23]:

bigdf = pd.DataFrame()
for column in dfdate.columns.values:
    for i in range(24,0,-1):
        bigdf[str(column)+"t-"+str(i)] = dfdate[str(column)].shift(i)
    bigdf[str(column)] = dfdate[str(column)]
bigdf = bigdf.dropna()


# In[24]:

table_names = []
for column in dfdate.loc[:,dfdate.columns != 'consumedE'].columns.values:
    globals()['bigdf%s' % column] = bigdf.loc[:,bigdf.columns.str.contains(str(column))]
    table_names.append('bigdf%s' % column)


# # here is the important part

# In[ ]:

forecast = pd.DataFrame()
for name in table_names:
    X = globals()[name].loc[ :, globals()[name].columns != globals()[name].columns[-1] ]
    y = globals()[name].iloc[:,-1]
    newX, newy = RF_1timeseries(X,y)
    newX[globals()[name].columns[-1]] = newy
    globals()[name] = globals()[name].append(newX)
    forecast = pd.concat([forecast,newX ], axis=1)

bigmodel = RandomForestRegressor(n_estimators=50,criterion="mae",max_depth=4,random_state=1,n_jobs=-1)
bigmodel.fit( bigdf.loc[:, bigdf.columns != 'consumedE'] , bigdf.consumedE )

conEarray = np.delete(bigdf.loc[:,bigdf.columns.str.contains("consumed",na=True)].iloc[-1].values,0)
conEarray = np.reshape(conEarray,(1,len(conEarray)))
conEcolumns = bigdf.loc[:,(bigdf.columns.str.contains("consumed",na=True)) & (bigdf.columns != 'consumedE')].columns.values
conE = pd.DataFrame( conEarray, columns=conEcolumns )
conE["date"] = pd.Series(pd.date_range(bigdf.index[-1]+pd.to_timedelta(1, unit='h'), periods=1, freq='H'))
conE = conE.set_index("date")

forecast = pd.concat([forecast,conE], axis=1)
conEy = bigmodel.predict(forecast)
forecast["consumedE"] = conEy
bigdf = bigdf.append(forecast)


# In[99]:

bigdf.to_csv("forecast_energy.csv")


# In[100]:

bigdf


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# # Feature reduction using random forest

# In[17]:

model = RandomForestRegressor(n_estimators=2000,criterion="mae",max_features=3,max_depth=2,random_state=1,n_jobs=-1)
model.fit( bigdf.loc[:, bigdf.columns != 'consumedE'] , bigdf.consumedE)


# In[18]:

names = bigdf.loc[:, bigdf.columns != 'consumedE'].columns.values
ticks = [i for i in range(len(names))]
plt.figure(figsize=(25,5))
plt.bar(ticks, model.feature_importances_)
plt.xticks(ticks,names)
plt.show()


# In[19]:

importance = pd.DataFrame()
importance["feature"] = names
importance["value"] = model.feature_importances_


# In[20]:

importance


# In[21]:

importance = importance[importance.value < importance.value.mean()]
bigdf = bigdf.drop(list(importance.feature.values), axis = 1)


# In[22]:

model = RandomForestRegressor(n_estimators=2000,criterion="mae",max_features=3,max_depth=2,random_state=1,n_jobs=-1)
model.fit( bigdf.loc[:, bigdf.columns != 'consumedE'] , bigdf.consumedE)


# In[23]:

names = bigdf.loc[:, bigdf.columns != 'consumedE'].columns.values
ticks = [i for i in range(len(names))]
plt.figure(figsize=(25,5))
plt.bar(ticks, model.feature_importances_)
plt.xticks(ticks,names)
plt.show()


# In[24]:

len(bigdf.columns)


# In[ ]:




# In[23]:

depth = 10
errorvalue = np.zeros((depth-1,2))
for d in range(1,depth):
    y = pd.DataFrame()
    y["consumedE"] = bigdf.consumedE
    model = RandomForestRegressor(n_estimators=50,criterion="mae",max_depth=d,random_state=1,n_jobs=-1)
    model.fit( bigdf.loc[:, bigdf.columns != 'consumedE'] , bigdf.consumedE)
    predicted = model.predict(bigdf.loc[:, bigdf.columns != 'consumedE'])
    y["fitted"] = predicted
    y["error"] = np.abs(y.fitted - y.consumedE)/y.consumedE
    errorvalue[d-1,0] = int(d)
    errorvalue[d-1,1] = y.error.mean()*100


# In[24]:

errorplt = pd.DataFrame(errorvalue,columns=["depth","error"])
errorplt = errorplt.set_index("depth")
plt.style.use('ggplot')
errorplt.plot()


# In[25]:

dfdate.tail(2)


# In[25]:

from sklearn.model_selection import TimeSeriesSplit


# In[45]:

final_test = bigdf.reset_index()
final_test.date = pd.to_datetime(final_test.date)
final_test = final_test[final_test.date >= pd.to_datetime("2016-05-26")]
final_test = final_test.set_index("date")

train_set = bigdf.reset_index()
train_set.date = pd.to_datetime(train_set.date)
train_set = train_set[train_set.date < pd.to_datetime("2016-05-26")]
train_set = train_set.set_index("date")


# In[46]:

train_set.tail()


# In[47]:

final_test.tail()


# In[48]:

len(final_test)


# In[53]:

d = 4
model = RandomForestRegressor(n_estimators=50,criterion="mae",max_depth=d,random_state=1,n_jobs=-1)
model.fit( train_set.loc[:, train_set.columns != 'consumedE'] , train_set.consumedE )


# In[54]:

predicted = model.predict(final_test.loc[:, final_test.columns != 'consumedE'])
y = pd.DataFrame()
y["consumedE"] = final_test.consumedE.values
y["fitted"] = predicted
y["error"] = np.abs(y.fitted - y.consumedE)/y.consumedE


# In[55]:

y.error.mean()


# In[56]:

y[["consumedE","fitted"]].plot()


# In[60]:




# In[ ]:




# In[ ]:




# In[36]:

tscv = TimeSeriesSplit( n_splits=int(len(train_set)/25) )
depth0 = 1
depth1 = 5
errors = []
for d in range(depth0,depth1):
    hypothesisresults = []
    split = 0
    for train, test in tscv.split(train_set):
        y = pd.DataFrame()
        model = RandomForestRegressor(n_estimators=50,criterion="mae",max_depth=d,random_state=1,n_jobs=-1)
        model.fit( train_set.loc[:, train_set.columns != 'consumedE'].iloc[train] , train_set.consumedE.iloc[train] )
        predicted = model.predict(train_set.loc[:, bigdf.columns != 'consumedE'].iloc[test])
        y["consumedE"] = bigdf.consumedE.iloc[test]
        y["fitted"] = predicted
        y["error"] = np.abs(y.fitted - y.consumedE)/y.consumedE
    hypothesisresults.append(y.error.mean())
    errors.append( hypothesisresults  )
print(len(test))


# In[32]:

errors= np.asarray(errors)
errors = pd.DataFrame(errors,columns=["error"])
errors.index +=1
errors[errors.error==errors.error.min()]


# In[33]:

plt.style.use('classic')

errors.error = errors.error
ax = errors.plot(c="blue",legend=False)
ax.set_xlabel("Number of lag terms")
ax.set_ylabel("error (%)")
plt.savefig('error.eps', format='eps', dpi=1000)


# In[35]:

errors


# In[ ]:



