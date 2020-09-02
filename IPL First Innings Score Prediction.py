First Innings Score Prediction
In [38]:
# Importing essential libraries
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('ipl.csv')
In [39]:
df.head()
Out[39]:
mid	date	venue	bat_team	bowl_team	batsman	bowler	runs	wickets	overs	runs_last_5	wickets_last_5	striker	non-striker	total
0	1	2008-04-18	M Chinnaswamy Stadium	Kolkata Knight Riders	Royal Challengers Bangalore	SC Ganguly	P Kumar	1	0	0.1	1	0	0	0	222
1	1	2008-04-18	M Chinnaswamy Stadium	Kolkata Knight Riders	Royal Challengers Bangalore	BB McCullum	P Kumar	1	0	0.2	1	0	0	0	222
2	1	2008-04-18	M Chinnaswamy Stadium	Kolkata Knight Riders	Royal Challengers Bangalore	BB McCullum	P Kumar	2	0	0.2	2	0	0	0	222
3	1	2008-04-18	M Chinnaswamy Stadium	Kolkata Knight Riders	Royal Challengers Bangalore	BB McCullum	P Kumar	2	0	0.3	2	0	0	0	222
4	1	2008-04-18	M Chinnaswamy Stadium	Kolkata Knight Riders	Royal Challengers Bangalore	BB McCullum	P Kumar	2	0	0.4	2	0	0	0	222
In [40]:
# --- Data Cleaning ---
# Removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)
In [41]:
df.head()
Out[41]:
date	bat_team	bowl_team	runs	wickets	overs	runs_last_5	wickets_last_5	total
0	2008-04-18	Kolkata Knight Riders	Royal Challengers Bangalore	1	0	0.1	1	0	222
1	2008-04-18	Kolkata Knight Riders	Royal Challengers Bangalore	1	0	0.2	1	0	222
2	2008-04-18	Kolkata Knight Riders	Royal Challengers Bangalore	2	0	0.2	2	0	222
3	2008-04-18	Kolkata Knight Riders	Royal Challengers Bangalore	2	0	0.3	2	0	222
4	2008-04-18	Kolkata Knight Riders	Royal Challengers Bangalore	2	0	0.4	2	0	222
In [42]:
df['bat_team'].unique()
Out[42]:
array(['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Deccan Chargers', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Delhi Daredevils',
       'Kochi Tuskers Kerala', 'Pune Warriors', 'Sunrisers Hyderabad',
       'Rising Pune Supergiants', 'Gujarat Lions',
       'Rising Pune Supergiant'], dtype=object)
In [43]:
# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
In [44]:
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
In [45]:
# Removing the first 5 overs data in every match
df = df[df['overs']>=5.0]
In [46]:
df.head()
Out[46]:
date	bat_team	bowl_team	runs	wickets	overs	runs_last_5	wickets_last_5	total
32	2008-04-18	Kolkata Knight Riders	Royal Challengers Bangalore	61	0	5.1	59	0	222
33	2008-04-18	Kolkata Knight Riders	Royal Challengers Bangalore	61	1	5.2	59	1	222
34	2008-04-18	Kolkata Knight Riders	Royal Challengers Bangalore	61	1	5.3	59	1	222
35	2008-04-18	Kolkata Knight Riders	Royal Challengers Bangalore	61	1	5.4	59	1	222
36	2008-04-18	Kolkata Knight Riders	Royal Challengers Bangalore	61	1	5.5	58	1	222
In [47]:
print(df['bat_team'].unique())
print(df['bowl_team'].unique())
['Kolkata Knight Riders' 'Chennai Super Kings' 'Rajasthan Royals'
 'Mumbai Indians' 'Kings XI Punjab' 'Royal Challengers Bangalore'
 'Delhi Daredevils' 'Sunrisers Hyderabad']
['Royal Challengers Bangalore' 'Kings XI Punjab' 'Delhi Daredevils'
 'Rajasthan Royals' 'Mumbai Indians' 'Chennai Super Kings'
 'Kolkata Knight Riders' 'Sunrisers Hyderabad']
In [48]:
# Converting the column 'date' from string into datetime object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
In [49]:
# --- Data Preprocessing ---
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
In [51]:
encoded_df.head()
Out[51]:
date	runs	wickets	overs	runs_last_5	wickets_last_5	total	bat_team_Chennai Super Kings	bat_team_Delhi Daredevils	bat_team_Kings XI Punjab	...	bat_team_Royal Challengers Bangalore	bat_team_Sunrisers Hyderabad	bowl_team_Chennai Super Kings	bowl_team_Delhi Daredevils	bowl_team_Kings XI Punjab	bowl_team_Kolkata Knight Riders	bowl_team_Mumbai Indians	bowl_team_Rajasthan Royals	bowl_team_Royal Challengers Bangalore	bowl_team_Sunrisers Hyderabad
32	2008-04-18	61	0	5.1	59	0	222	0	0	0	...	0	0	0	0	0	0	0	0	1	0
33	2008-04-18	61	1	5.2	59	1	222	0	0	0	...	0	0	0	0	0	0	0	0	1	0
34	2008-04-18	61	1	5.3	59	1	222	0	0	0	...	0	0	0	0	0	0	0	0	1	0
35	2008-04-18	61	1	5.4	59	1	222	0	0	0	...	0	0	0	0	0	0	0	0	1	0
36	2008-04-18	61	1	5.5	58	1	222	0	0	0	...	0	0	0	0	0	0	0	0	1	0
5 rows × 23 columns

In [16]:
encoded_df.head()
Out[16]:
date	runs	wickets	overs	runs_last_5	wickets_last_5	total	bat_team_Chennai Super Kings	bat_team_Delhi Daredevils	bat_team_Kings XI Punjab	...	bat_team_Royal Challengers Bangalore	bat_team_Sunrisers Hyderabad	bowl_team_Chennai Super Kings	bowl_team_Delhi Daredevils	bowl_team_Kings XI Punjab	bowl_team_Kolkata Knight Riders	bowl_team_Mumbai Indians	bowl_team_Rajasthan Royals	bowl_team_Royal Challengers Bangalore	bowl_team_Sunrisers Hyderabad
32	2008-04-18	61	0	5.1	59	0	222	0	0	0	...	0	0	0	0	0	0	0	0	1	0
33	2008-04-18	61	1	5.2	59	1	222	0	0	0	...	0	0	0	0	0	0	0	0	1	0
34	2008-04-18	61	1	5.3	59	1	222	0	0	0	...	0	0	0	0	0	0	0	0	1	0
35	2008-04-18	61	1	5.4	59	1	222	0	0	0	...	0	0	0	0	0	0	0	0	1	0
36	2008-04-18	61	1	5.5	58	1	222	0	0	0	...	0	0	0	0	0	0	0	0	1	0
5 rows × 23 columns

In [52]:
encoded_df.columns
Out[52]:
Index(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
       'total', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad'],
      dtype='object')
In [53]:
# Rearranging the columns
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
In [54]:
# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]
In [55]:
y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values
In [56]:
# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)
In [57]:
# --- Model Building ---
# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
Out[57]:
LinearRegression()
In [22]:
# Creating a pickle file for the classifier
filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))
Ridge Regression
In [58]:
## Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
In [25]:
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)
c:\users\krish naik\anaconda3\envs\myenv1\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=3.27836e-22): result may not be accurate.
  overwrite_a=True).T
c:\users\krish naik\anaconda3\envs\myenv1\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.50489e-18): result may not be accurate.
  overwrite_a=True).T
c:\users\krish naik\anaconda3\envs\myenv1\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.57944e-18): result may not be accurate.
  overwrite_a=True).T
c:\users\krish naik\anaconda3\envs\myenv1\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.54481e-18): result may not be accurate.
  overwrite_a=True).T
c:\users\krish naik\anaconda3\envs\myenv1\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.5532e-18): result may not be accurate.
  overwrite_a=True).T
c:\users\krish naik\anaconda3\envs\myenv1\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning: Ill-conditioned matrix (rcond=1.54515e-18): result may not be accurate.
  overwrite_a=True).T
Out[25]:
GridSearchCV(cv=5, estimator=Ridge(),
             param_grid={'alpha': [1e-15, 1e-10, 1e-08, 0.001, 0.01, 1, 5, 10,
                                   20, 30, 35, 40]},
             scoring='neg_mean_squared_error')
In [26]:
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
{'alpha': 40}
-328.4152792487923
In [32]:
prediction=ridge_regressor.predict(X_test)
In [33]:
import seaborn as sns
sns.distplot(y_test-prediction)
Out[33]:
<matplotlib.axes._subplots.AxesSubplot at 0x1967a180e48>

In [36]:
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
MAE: 12.117294527005031
MSE: 251.03172964112676
RMSE: 15.843980864704639
Lasso Regression
In [27]:
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
In [31]:
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-31-2000d07901bd> in <module>
      1 lasso=Lasso()
      2 parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
----> 3 lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',iteration=10,cv=5)
      4 
      5 lasso_regressor.fit(X_train,y_train)

c:\users\krish naik\anaconda3\envs\myenv1\lib\site-packages\sklearn\utils\validation.py in inner_f(*args, **kwargs)
     71                           FutureWarning)
     72         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
---> 73         return f(**kwargs)
     74     return inner_f
     75 

TypeError: __init__() got an unexpected keyword argument 'iteration'