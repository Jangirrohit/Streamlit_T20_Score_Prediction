from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import pickle

data=pd.read_csv('T20I_ball_by_ball_updated.csv')
# print(data.head())
# print(data.shape)
# print(data.columns)
# print(data.head())
# data = data[['venue', 'innings', 'ball', 'batting_team', 'bowling_team', 'runs_off_bat','extras', 'wides', 'noballs', 'byes', 'legbyes', 'penalty', 'wicket_type', 'player_dismissed']]
data = data[['match_id','venue', 'innings', 'ball', 'batting_team', 'bowling_team', 'runs_off_bat','extras', 'wides', 'noballs', 'byes', 'legbyes','wicket_type']]
data['runs_off_bat_cumsum'] = data.groupby([ 'match_id','innings'])['runs_off_bat'].cumsum()
data['extras_in_runs'] = data.groupby(['match_id', 'innings'])['extras'].cumsum()
data['runs']=data['runs_off_bat_cumsum']+data['extras_in_runs']
# print(data.columns)
data['total_bat_runs']=data.groupby(['match_id','innings'])['runs_off_bat'].transform('sum')
data['total_extra_runs']=data.groupby(['match_id','innings'])['extras'].transform('sum')
data['total_runs']=data['total_bat_runs']+data['total_extra_runs']
data.rename(columns={'ball': 'Overs'}, inplace=True)

data['wicket_number'] = data['wicket_type'].apply(lambda x: 1 if pd.notna(x) else 0)
data['wickets'] = data.groupby([ 'match_id','innings'])['wicket_number'].cumsum()

data['wickets_last_5_over'] = data.groupby(['innings', 'match_id'])['wicket_number'].rolling(window=30, min_periods=1).sum().reset_index(level=[0, 1], drop=True)

data['runs_off_bat_last_5_over'] = data.groupby(['innings', 'match_id'])['runs_off_bat'].rolling(window=30, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
data['extras_last_5_over'] = data.groupby(['innings', 'match_id'])['extras'].rolling(window=30, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
data['runs_last_5_over']=data['runs_off_bat_last_5_over']+data['extras_last_5_over']



data['boundary_number_in_bats_run'] = data['runs_off_bat'].apply(lambda x: 1 if x in [4,6]  else 0)
data['boundary_number_in_extras'] = data['extras'].apply(lambda x: 1 if x in [4,6]  else 0)
data['boundary_number']=data['boundary_number_in_bats_run']+data['boundary_number_in_extras']
data['boundaries_in_last_5_over'] = data.groupby(['innings', 'match_id'])['boundary_number'].rolling(window=30, min_periods=1).sum().reset_index(level=[0, 1], drop=True)


data['dot_ball_number_in_bats_run'] = data['runs_off_bat'].apply(lambda x: 1 if x==0  else 0)
# data['dot_ball_number_in_extras'] = data['extras'].apply(lambda x: 1 if x==0  else 0)
# data['dot_ball_number']=data['dot_ball_number_in_bats_run']+data['dot_ball_number_in_extras']
data['dot_ball_number']=data['dot_ball_number_in_bats_run']
data['dot_ball_in_last_5_over'] = data.groupby(['innings', 'match_id'])['dot_ball_number'].rolling(window=30, min_periods=1).sum().reset_index(level=[0, 1], drop=True)

# print(data.columns)
# print(data.head(125))
data.to_csv("modified.csv", index=False)


# filltering the final useful_columns
# final_data=data[['venue', 'innings', 'Overs', 'batting_team', 'bowling_team','runs','total_runs','wickets_last_5_over','runs_last_5_over','boundaries_in_last_5_over', 'dot_ball_in_last_5_over','wickets']]
final_data=data[['Overs', 'batting_team', 'bowling_team','runs','total_runs','wickets_last_5_over','runs_last_5_over','boundaries_in_last_5_over', 'dot_ball_in_last_5_over','wickets']]
final_data.to_csv("final_data.csv", index=False)

print(f'Before Removing Overs : {final_data.shape}')
final_data = final_data[final_data['Overs'] >= 5.0]
print(f'After Removing Overs : {final_data.shape}')
final_data.head()



 
# print(data.columns)
# print(data.head(125))

# # Identify missing values
# missing_values = final_data.isnull()

# # Visualize missing data
# sns.heatmap(missing_values, cbar=False, cmap='viridis')
# plt.show()

# # print(final_data.describe())
# # print(final_data.info())
# # print(final_data.nunique())

 
sns.displot(final_data['wickets'],kde=False,bins=10)
plt.title("Wickets Distribution")
# plt.savefig('wickets distribution')
plt.show()

 
sns.displot(final_data['total_runs'],kde=False,bins=10)
plt.title("Runs Distribution")
# plt.savefig('total runs distribution')
plt.show()


# 2. Correlation Heatmap
correlation_matrix = final_data.drop(['batting_team', 'bowling_team'],axis=1).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
# plt.savefig(' CorrelationHeatmap.png')
plt.show()

 


# print(final_data['batting_team'].unique())

const_teams=['Afghanistan', 'Australia', 'Bangladesh', 'England', 'India', 'Ireland', 'Namibia', 'Netherlands', 'New Zealand', 'Pakistan', 'Scotland', 'South Africa', 'Sri Lanka', 'United Arab Emirates', 'West Indies', 'Zimbabwe']
print(f'Before Removing Inconsistent Teams : {final_data.shape}')
final_data = final_data[(final_data['batting_team'].isin(const_teams)) & (final_data['bowling_team'].isin(const_teams))]
print(f'After Removing Irrelevant Columns : {final_data.shape}')
print(f"Consistent Teams : \n{final_data['batting_team'].unique()}")

# print(final_data.shape,"befor performing incoding")

# ohe=OneHotEncoder(drop='first',sparse=False)
# final_data_bat_bowl=ohe.fit_transform(final_data[['batting_team','bowling_team']])
# print(final_data_bat_bowl.shape)
columnTransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False,drop='first'), ['batting_team', 'bowling_team'])], remainder='passthrough')
df=columnTransformer.fit_transform(final_data)
column_names = columnTransformer.get_feature_names_out()
column_names = [col.replace('encoder__', '').replace('remainder__', '') for col in column_names]
df = pd.DataFrame(df,columns=column_names)
# print(df.columns.to_list())
df.to_csv("final_modified_data.csv", index=False)


# model making
features = df.drop(['total_runs'], axis='columns')
y = df['total_runs']

train_features, test_features, train_y, test_y = train_test_split(features, y, test_size=0.20, shuffle=True)
print(f"Training Set : {train_features.shape}\nTesting Set : {test_features.shape}")

# ml algorithms
# 1. dicision tree 
tree= DecisionTreeRegressor()
tree.fit(train_features,train_y)
print("---- Decision Tree Regressor - Model Evaluation ----")
print("score of dicision tree algorithm: ",tree.score(test_features,test_y)*100)
print("Mean Absolute Error (MAE): {}".format(mae(test_y, tree.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_y, tree.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_y, tree.predict(test_features)))))

# 2. linear regression model

linreg = LinearRegression()
linreg.fit(train_features, train_y)
print("---- linear Regressor - Model Evaluation ----")
print("score of linear Regressor algorithm: ",linreg.score(test_features,test_y)*100)
print("Mean Absolute Error (MAE): {}".format(mae(test_y, linreg.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_y, linreg.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_y, linreg.predict(test_features)))))

# print(train_features.columns.to_list())

# # 3. random forest model
forest = RandomForestRegressor()
forest.fit(train_features, train_y)
print("---- RandomForestRegressor - Model Evaluation ----")
print("score of RandomForestRegressor algorithm: ",forest.score(test_features,test_y)*100)
print("Mean Absolute Error (MAE): {}".format(mae(test_y, forest.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_y, forest.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_y, forest.predict(test_features)))))


# # 4. support vector machine 
# svm = SVR()
# svm.fit(train_features, train_y)
# print("---- support vector machine - Model Evaluation ----")
# print("score of support vector machine algorithm: ",svm.score(test_features,test_y)*100)
# print("Mean Absolute Error (MAE): {}".format(mae(test_y, svm.predict(test_features))))
# print("Mean Squared Error (MSE): {}".format(mse(test_y, svm.predict(test_features))))
# print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_y, svm.predict(test_features)))))

# # 5. KNR
knr = KNeighborsRegressor()
knr.fit(train_features, train_y)
print("---- KNR - Model Evaluation ----")
print("score of KNR algorithm: ",knr.score(test_features,test_y)*100)
print("Mean Absolute Error (MAE): {}".format(mae(test_y, knr.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_y, knr.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_y, knr.predict(test_features)))))

# # 6. XGBRegressor
xgb = XGBRegressor()
xgb.fit(train_features, train_y)
print("---- XGBRegressor - Model Evaluation ----")
print("score of XGBRegressor algorithm: ",xgb.score(test_features,test_y)*100)
print("Mean Absolute Error (MAE): {}".format(mae(test_y, xgb.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_y, xgb.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_y, xgb.predict(test_features)))))


model_names = ['Decision Tree', 'Linear Regression', 'Random Forest', 'KNeighbors', 'XGBoost']
mae_values = [mae(test_y, model.predict(test_features)) for model in [tree, linreg, forest, knr, xgb]]

plt.figure(figsize=(10, 6))
plt.bar(model_names, mae_values, color='blue')
plt.xlabel('Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Comparison of Models Based on MAE')
plt.show()

# Comparison of Models Based on Score
score_values = [model.score(test_features, test_y) * 100 for model in [tree, linreg, forest, knr, xgb]]

plt.figure(figsize=(10, 6))
plt.bar(model_names, score_values, color='green')
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Comparison of Models Based on Score')
plt.show()
 
 
def score_predict(batting_team, bowling_team, Overs, runs, wickets_last_5_over, runs_last_5_over, boundaries_in_last_5_over, dot_ball_in_last_5_over, wickets, model=forest):
  prediction_array = []

  # Batting Team
  if batting_team == 'Australia':
    prediction_array = prediction_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif batting_team == 'Bangladesh':
    prediction_array = prediction_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif batting_team == 'England':
    prediction_array = prediction_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
  elif batting_team == 'India':
    prediction_array = prediction_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
  elif batting_team == 'Ireland':
    prediction_array = prediction_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
  elif batting_team == 'Namibia':
    prediction_array = prediction_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
  elif batting_team == 'Netherlands':
    prediction_array = prediction_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
  elif batting_team == 'New Zealand':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
  elif batting_team == 'Pakistan':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
  elif batting_team == 'Scotland':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
  elif batting_team == 'South Africa':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
  elif batting_team == 'Sri Lanka':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
  elif batting_team == 'United Arab Emirates':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
  elif batting_team == 'West Indies':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
  elif batting_team == 'Zimbabwe':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

  # Bowling Team

  if bowling_team == 'Australia':
    prediction_array = prediction_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif bowling_team == 'Bangladesh':
    prediction_array = prediction_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
  elif bowling_team == 'England':
    prediction_array = prediction_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
  elif bowling_team == 'India':
    prediction_array = prediction_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
  elif bowling_team == 'Ireland':
    prediction_array = prediction_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
  elif bowling_team == 'Namibia':
    prediction_array = prediction_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
  elif bowling_team == 'Netherlands':
    prediction_array = prediction_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
  elif bowling_team == 'New Zealand':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
  elif bowling_team == 'Pakistan':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
  elif bowling_team == 'Scotland':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
  elif bowling_team == 'South Africa':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
  elif bowling_team == 'Sri Lanka':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
  elif bowling_team == 'United Arab Emirates':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
  elif bowling_team == 'West Indies':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
  elif bowling_team == 'Zimbabwe':
    prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
  prediction_array = prediction_array + [Overs, runs, wickets_last_5_over, runs_last_5_over, boundaries_in_last_5_over, dot_ball_in_last_5_over, wickets]
  prediction_array = np.array([prediction_array])
  # Check if feature names are provided
  if feature_names is not None:
        df = pd.DataFrame(prediction_array, columns=feature_names)
        prediction_array = df.to_numpy()

  pred = model.predict(prediction_array)
  return int(round(pred[0]))

# List of feature names
feature_names = train_features.columns.to_list()



# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Create the grid search model
# grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# # Fit the grid search to the data
# grid_search.fit(train_features, train_y)

# # Print the best parameters
# print("Best Parameters:", grid_search.best_params_)

# # Get the best model
# best_model = grid_search.best_estimator_
# print(best_model.predict(test_features,test_y))
   

# testing
batting_team='Australia'
bowling_team='New Zealand'
Overs=18.1
runs=176.0
wickets_last_5_over=1.0
runs_last_5_over=57.0
boundaries_in_last_5_over=7.0
dot_ball_in_last_5_over=6.0
wickets=5.0

print(score_predict(batting_team, bowling_team, Overs, runs, wickets_last_5_over, runs_last_5_over, boundaries_in_last_5_over, dot_ball_in_last_5_over, wickets))

# save the  forest model using pickel
with open('forest_score_predi_pickel','wb') as f:
  pickle.dump(forest,f)