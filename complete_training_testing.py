#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


# In[2]:


data = pd.read_csv(r"D:\internship\dataset\data1.csv")


# # study of basic info.

# In[3]:


data


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.cov()


# In[7]:


data.corr()


# HorsePower and Number of cylinders shows positive relation
# HorsePower and mileage are inversly proportional to each other
# Number of cylinders and mileage are also inversly proportional to each other
# Price is directly proportional to HorsePower and Cylinder

# In[8]:


data.columns


# In[9]:


data["Make"].value_counts()


# In[10]:


data['Make'].value_counts().count()


# In[11]:


data["Year"].value_counts().count()


# In[12]:


data.rename(columns = {"Fuel ":"Fuel"} , inplace = True)


# In[13]:


data["Fuel"].value_counts()


# In[14]:


data = data[data["Fuel"]!="flex-fuel (unleaded/E85)"]
data = data[data["Fuel"]!="flex-fuel (premium unleaded required/E85)"]
data = data[data["Fuel"]!="flex-fuel (premium unleaded recommended/E85)"]
data = data[data["Fuel"]!="flex-fuel (unleaded/natural gas)"]
data = data[data["Fuel"]!="natural gas"]


# In[15]:


data


# In[16]:


data["Fuel"].replace(["regular unleaded","premium unleaded (required)","premium unleaded (recommended)"],["Petrol","Premium","Premium"],inplace = True)


# In[17]:


data["Fuel"].value_counts()


# removed flex-fuel as they are out of context and alter the data

# In[18]:


data["HP"].mean()


# In[19]:


data["HP"].mode()


# In[20]:


data["Cylinders"].value_counts()


# In[21]:


data["Transmission Type"].value_counts()


# In[22]:


data = data[data["Transmission Type"]!="UNKNOWN"]


# In[23]:


data["Driven_Wheels"].value_counts()


# In[24]:


data["Doors"].value_counts()


# In[25]:


data["Doors"].replace([3],[2],inplace = True)


# In[26]:


data["Size"].value_counts()


# #Changing Miles per Gallon to Kilometer per litre && price from Dollar to Rupees

# In[27]:


data["highway MPG"] = data["highway MPG"].apply(lambda x : x*0.495).round()
data["city mpg"] = data["city mpg"].apply(lambda x : x*0.495).round()


# In[28]:


data


# In[29]:


data["Price"] = data["Price"].apply(lambda x : x * 49)


# In[30]:


data


# # Data Visualization

# In[31]:


data["Make"].hist(bins = 50)


# In[32]:


data.hist(column = "Year",bins = 45)


# # from this we can say that, more no. of cars are 2014 - 2017 

# In[33]:


data["Fuel"].hist(bins = 4)


# # number of cars ,supporting type of fuel :  Petrol > Premiun > diesel > electric 

# In[34]:


data["Cylinders"].hist(bins = 15)


# # most number of cars have 4 , 6 , 8 cylinders engine

# In[35]:


data["Driven_Wheels"].hist(bins =8)


# In[36]:


data["Size"].hist(bins = 6)


# # number of compact and midsize are almost same

# In[37]:


data.boxplot(column ="HP")


# horsepower has many number of outliers, will be handled in later part

# In[38]:


data.boxplot(column ="highway MPG")


# In[39]:


data.boxplot(column = "city mpg")


# there are some many outliers because electric cars has high kmple(kilometer per litre equivalent)
# 
# # Bivariate Comparison

# In[40]:


data.boxplot(column ="HP", by ="Year")


# # As the year is increasing HorsePower is also increasing, So we can say that by using new technology we are able to harness more horsepower. 

# In[41]:


data.boxplot(column = 'HP' ,by = 'Fuel')


# # premium fuel gives more horsepower, as they have high octane value and more pure than regular. 

# In[42]:


data.boxplot(column = "HP" ,by = "Cylinders")


# # increase in cylinder causes increase in Horsepower

# In[43]:


data.boxplot(column = "HP" ,by = "Transmission Type")


# # Cars with high horsepower mainly use semi-automatic gear system

# In[44]:


data.boxplot(column = "HP" ,by = "Doors")


# In[45]:


data.boxplot(column = "highway MPG" ,by = "Cylinders")


# # 0 cylinders means its electric cars which has high mileage, and as increase in cylinder decreases in mileage

# In[46]:


data.boxplot(column = "highway MPG" ,by = "Fuel")


# # electric gives high mileage and other give almost same mileage

# In[47]:


data.boxplot(column = "highway MPG" ,by = "Transmission Type")


# In[48]:


pd.crosstab(data["city mpg"],data["Year"],margins = True)


# In[49]:


plt.scatter(data["HP"],data["highway MPG"])


# In[50]:


plt.scatter(data["HP"],data["Price"])


# In[51]:


import seaborn as sns


# In[52]:


sns.regplot(x = "HP",y = "Price",data = data,color = "black")


# In[53]:


sns.regplot(x = "highway MPG",y="Price",data = data)


# # mileage and price are inversely proportional to each other

# In[54]:


sns.distplot(data["highway MPG"])


# In[55]:


sns.distplot(data["Price"])


# In[137]:


from plotly.offline import iplot,init_notebook_mode


# In[138]:


import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected = True)


# In[139]:


data[["Cylinders","highway MPG"]].iplot(kind = "surface",colorscale = "blues")


# In[140]:


data[["Cylinders","HP"]].iplot()


# In[141]:


data.iplot(kind = "bar" , x= "Fuel",y = "HP" )


# In[142]:


corr = data.corr()


# In[143]:


sns.heatmap(corr,xticklabels = corr.columns,yticklabels = corr.columns)


# In[144]:


data["Avg mpl"] = data[["highway MPG","city mpg"]].mean(axis = 1).round()


# # Taking care of missing values

#     HP and Cylinder of electric vehicles are missing so filling HP of electric by mean of HP of electric and Cylinder by 0

# In[145]:


electric = data[data["Fuel"]=="electric"]


# In[146]:


electric["HP"].mean()


# In[147]:


data["HP"].fillna(145.0,inplace = True)
data["Cylinders"].fillna(0,inplace = True)
data["Fuel"].fillna("Petrol",inplace = True)


# In[148]:


data.apply(lambda x: sum(x.isnull()),axis = 0)


# In[149]:


import statsmodels.api as sm
model = sm.OLS(data["Avg mpl"],data["HP"]).fit()


# In[150]:


model.summary()


# In[151]:


model = sm.OLS(data["HP"],data["Cylinders"]).fit()
model.summary()


# In[152]:


model = sm.OLS(data["HP"],data["Avg mpl"]).fit()


# In[153]:


model.summary()


# In[154]:



model = sm.OLS(data["Price"],data[["HP","Cylinders"]]).fit()
model.summary()


# # MODEL 1

# In[155]:


x = data.loc[:,["Fuel","HP","Cylinders"]].values
y = data.loc[:,["Avg mpl"]].values


# # performing OneHotEncoding on Fuel column 

# In[156]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[157]:


from sklearn.externals import joblib


# In[158]:


ct = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[0])] , remainder = "passthrough")
ct.fit(x)
x = ct.transform(x)


# In[159]:


x


# In[160]:


y = y.ravel()


# # Splitting in training and testing

# In[161]:


from sklearn.model_selection import train_test_split
x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[162]:


from sklearn.ensemble import RandomForestRegressor


# In[163]:


model1_forest = RandomForestRegressor(n_estimators=100,random_state = 0)
model1_forest.fit(x_train,y_train)


# In[164]:


model1_forest.predict(x_test)


# In[165]:


model1_forest.score(x_train,y_train)


# In[166]:


model1_forest.score(x_test,y_test)


# In[169]:


l = np.array([["electric",145.0,0],["Petrol",145,4],["Premium",145,4],["diesel",145,4]])


# In[170]:


l = ct.transform(l)


# In[171]:


model1_forest.predict(l)


# In[172]:


from sklearn.metrics import mean_squared_error,r2_score,mean_squared_log_error


# In[173]:


y_prediction = model1_forest.predict(x_test)


# In[174]:


print("Mean Squared Error : ",mean_squared_error(y_test,y_prediction))


# In[175]:


print("Root Mean Squared Error :",np.sqrt(mean_squared_error(y_test,y_prediction)))


# In[176]:


print("r2_score :",r2_score(y_test,y_prediction))


# In[177]:


print("Root mean squared log error :",np.sqrt(mean_squared_error(y_test,y_prediction)))


# # From the above observation we can say that RandomForestRegressor has performed very well

# # for linear regression we gonna perform feature Scaling

# In[178]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_train[:,4:] = sc.fit_transform(x_train[:,4:])
x_test[:,4:] = sc.transform(x_test[:,4:])


# In[179]:


from sklearn.linear_model import LinearRegression


# In[180]:


model1_linear = LinearRegression()


# In[181]:


model1_linear.fit(x_train,y_train)


# In[182]:


model1_linear.predict(x_test)


# In[183]:


model1_linear.score(x_test,y_test)


# In[184]:


from sklearn.metrics import mean_squared_error,r2_score,mean_squared_log_error
y_prediction = model1_forest.predict(x_test)
print("Mean Squared Error : ",mean_squared_error(y_test,y_prediction))
print("Root Mean Squared Error :",np.sqrt(mean_squared_error(y_test,y_prediction)))
print("r2_score :",r2_score(y_test,y_prediction))
print("Root mean squared log error :",np.sqrt(mean_squared_error(y_test,y_prediction)))


# # from the above observation we will use RandomForestRegressor

# # Model2

# In[185]:


x2 = data.loc[:,["Fuel","Cylinders","Avg mpl"]].values
y2 = data.loc[:,"HP"].values


# In[187]:


x2 = ct.transform(x2)


# In[188]:


x2


# In[189]:


y2 = y2.ravel()


# In[190]:


from sklearn.model_selection import train_test_split
x_train ,x_test , y_train , y_test = train_test_split(x2,y2,test_size = 0.2,random_state = 0)


# In[191]:


model2 = RandomForestRegressor(n_estimators=100,random_state = 0)
model2.fit(x_train,y_train)


# In[192]:


model2.predict(x_test)


# In[193]:


model2.score(x_test,y_test)


# In[194]:


y_pred = model2.predict(x_test)


# In[195]:


print("Mean Squared Error : ",mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error :",np.sqrt(mean_squared_error(y_test,y_pred)))
print("r2_score :",r2_score(y_test,y_pred))
print("Root mean squared log error :",np.sqrt(mean_squared_error(y_test,y_pred)))


# In[196]:


from sklearn.tree import DecisionTreeRegressor


# In[197]:


regressor = DecisionTreeRegressor(random_state = 0,max_depth = 15)
regressor.fit(x_train,y_train)


# In[198]:


regressor.predict(x_test)


# In[199]:


regressor.score(x_test,y_test)


# In[200]:


y_pred = regressor.predict(x_test)


# In[201]:


print("Mean Squared Error : ",mean_squared_error(y_test,y_prediction))
print("Root Mean Squared Error :",np.sqrt(mean_squared_error(y_test,y_prediction)))
print("r2_score :",r2_score(y_test,y_prediction))
print("Root mean squared log error :",np.sqrt(mean_squared_error(y_test,y_prediction)))


# # Model 3

# In[215]:


x3 = data.loc[:,["Fuel","Cylinders","HP","Transmission Type","Driven_Wheels"]].values
y3 = data.loc[:,["Price"]].values


# In[216]:


x3


# In[217]:


y3 = y3.ravel()


# In[218]:


ct2 = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[0,3,4])] , remainder = "passthrough")
ct2.fit(x3)
joblib.dump(ct2 ,"transformer2.pkl")
x3 = ct2.transform(x3)


# In[219]:


x3.shape


# In[220]:


data.info()


# In[221]:


from sklearn.model_selection import train_test_split
x_train ,x_test , y_train , y_test = train_test_split(x3,y3,test_size = 0.2,random_state = 0)


# In[222]:


model3 = RandomForestRegressor(n_estimators=100,random_state = 0)
model3.fit(x_train,y_train)


# In[223]:


model3.score(x_test,y_test)


# In[224]:


y_pred = model3.predict(x_test)


# In[225]:


print("Mean Squared Error : ",mean_squared_error(y_test,y_prediction))
print("Root Mean Squared Error :",np.sqrt(mean_squared_error(y_test,y_prediction)))
print("r2_score :",r2_score(y_test,y_prediction))
print("Root mean squared log error :",np.sqrt(mean_squared_error(y_test,y_prediction)))


# In[226]:


model3.score(x_train,y_train)


# In[227]:


model3.predict(x_train)

