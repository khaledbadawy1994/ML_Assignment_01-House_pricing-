# ML_Assignment_01-House_pricing-

import pandas as pd
housing = pd.read_csv("/content/drive/MyDrive/housing.csv")
housing.head()

# Take a Quick Look at the Data Structure

housing.head()

housing.info()

housing.isnull().mean() * 100

housing.describe()

housing["ocean_proximity"].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px

px.scatter(housing, x="longitude", y="latitude", color="median_house_value",  title="Housing Prices", width=800, height=600)

housing.hist(bins=50, figsize=(20,15));

# Create a Test Set

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing , test_size=0.2, random_state=42)

train_set.shape, test_set.shape

train_set.hist(bins=50,figsize=(20,15));

test_set.hist(bins=50,figsize=(20,15));

train_set.median_house_value.describe(),test_set.median_house_value.describe()

px.scatter(train_set, x="longitude" , y="latitude" , color="median_house_value" ,title="Housing prices" , width=800 , height=600)

# EDA

train_set.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])

train_set.corr()

train_set.corr()["median_house_value"].sort_values(ascending=False)

train_set.corr()["median_house_value"].apply(lambda x: abs(x)).sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(train_set[attributes], figsize=(12, 8));

sns.pairplot(train_set);

px.scatter(train_set, x="median_income" , y="median_house_value"  ,title="Housing prices" , width=800 , height=600)

# The most promising attribute to predict the median house value is the median income

# Feature Engineering

# Rooms per household
train_set["rooms_per_household"] = train_set["total_rooms"]/train_set["households"]

# Bedrooms per room
train_set["bedrooms_per_room"] = train_set["total_bedrooms"]/train_set["total_rooms"]

# Population per household
train_set["population_per_household"] = train_set["population"]/train_set["households"]

train_set.corr()["median_house_value"].sort_values(ascending=False)

# Absolute correlation matrix
train_set.corr()["median_house_value"].apply(lambda x: abs(x)).sort_values(ascending=False)

# bedrooms_per_room is more informative than total_bedrooms or total_rooms.
# rooms_per_household is also more informative than total_rooms or households.

# Outlier Detection

train_set.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])

# Create a copy of the training set
housing = train_set.copy()

# Remove outliers
housing = housing[housing["median_income"] < train_set["median_income"].quantile(0.99)]
housing = housing[housing["rooms_per_household"] < train_set["rooms_per_household"].quantile(0.99)]
housing = housing[housing["bedrooms_per_room"] < train_set["bedrooms_per_room"].quantile(0.99)]
housing = housing[housing["population_per_household"] < train_set["population_per_household"].quantile(0.99)]

housing.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])

# Prepare the Data for Machine Learning Algorithms

# Split the data into features and labels

train_features = train_set.drop("median_house_value", axis=1)
train_labels = train_set["median_house_value"].copy()

# Missing Values

test_set.total_bedrooms.isnull().sum()

housing.total_bedrooms.isnull().sum()

housing[housing.total_bedrooms.isnull()].index
Int64Index([], dtype='int64')

median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

# Impute missing values with median using SimpleImputer

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# Remove the text attribute because median can only be calculated on numerical attributes
housing_num = train_features.drop("ocean_proximity", axis=1)

# Fit the imputer instance to the training data using the fit() method
imputer.fit(housing_num)

# The imputer has simply computed the median of each attribute and stored the result in its statistics_ instance variable
imputer.statistics_

# format the output to be more readable
import numpy as np
np.set_printoptions(suppress=True)
imputer.statistics_

X = imputer.transform(housing_num)
X 

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
housing_tr.head()

#Handling Text and Categorical Attributes

housing_cat = train_features[["ocean_proximity"]]
housing_cat.head(10)

housing_cat["ocean_proximity"].value_counts()

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# Convert sparse matrix to a dense matrix
housing_cat_1hot.toarray()

# Make 'sparse=False' to get a dense matrix
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

cat_encoder.categories_

housing_num.describe().loc[["min", "max"]]

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
housing_num_scaled = scaler.fit_transform(housing_num)
housing_num_scaled

housing_num_scaled.min(axis=0), housing_num_scaled.max(axis=0)

# MinMaxScaler with custom range
custom_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_custom_scaled = custom_scaler.fit_transform(housing_num)
housing_num_custom_scaled

housing_num_custom_scaled.min(axis=0), housing_num_custom_scaled.max(axis=0)

# StandardScaler
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
housing_num_std_scaled

housing_num_std_scaled.mean(axis=0), housing_num_std_scaled.std(axis=0)

# num_pipeline

from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([   ('imputer', SimpleImputer(strategy="median")),
                            ('std_scaler', StandardScaler()),
                        ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# using make_pipeline
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(   SimpleImputer(strategy="median"),
                                StandardScaler()
                            )

housing_num_tr = num_pipeline.fit_transform(housing_num)

# Combine the numerical and categorical pipelines

from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([ ("num", num_pipeline, num_attribs),
                                    ("cat", OneHotEncoder(sparse= False, drop = 'first'), cat_attribs),
                                  ])

# Using make_column_transformer
from sklearn.compose import make_column_transformer
full_pipeline = make_column_transformer(    (num_pipeline, num_attribs),
                                            (OneHotEncoder(sparse= False), cat_attribs)
                                       )

# Transform the training data
train_features_prepared = full_pipeline.fit_transform(train_features)

full_pipeline = make_column_transformer(    (num_pipeline, num_attribs),
                                            (OneHotEncoder(sparse= False, drop='first'), cat_attribs)
                                       )

# Transform the training data
train_features_prepared = full_pipeline.fit_transform(train_features)
train_features_prepared.shape

# Final Pipeline

num_pipeline = Pipeline([   ('imputer', SimpleImputer(strategy="median")),
                            ('std_scaler', StandardScaler()),
                        ])

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([ ("num", num_pipeline, num_attribs),
                                    ("cat", OneHotEncoder(sparse= False, drop = 'first'), cat_attribs),
                                  ])

train_features_prepared = full_pipeline.fit_transform(train_features)

# access one hot encoder categories
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_encoder.categories_

#Linear Regression

train_features_prepared

median_income = train_features_prepared[:,7]
median_income

median_house_value = train_labels.values
median_house_value

# draw a scatter plot with regression line
sns.regplot(x=median_income, y=median_house_value, scatter_kws={"color": "blue"}, line_kws={"color": "red"});

# Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_features_prepared, train_labels)

# Predictions
lin_reg_predictions = lin_reg.predict(train_features_prepared)

#Decision Tree

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(train_features_prepared, train_labels)

# Predictions
tree_predictions = tree.predict(train_features_prepared)

# Training Accuracy
from sklearn.metrics import r2_score
print("Linear Regression Accuracy: ", r2_score(train_labels, lin_reg_predictions))
print("Decision Tree Accuracy: ", r2_score(train_labels, tree_predictions))

# Cross Validation for Linear Regression

from sklearn.model_selection import cross_val_score
lin_reg_scores = cross_val_score(lin_reg, train_features_prepared, train_labels, scoring="r2", cv=10)

print("Linear Regression Accuracy: ", lin_reg_scores)
print("Linear Regression Accuracy: ", round(lin_reg_scores.mean(),2))
print("Linear Regression Standard Deviation: ", round(lin_reg_scores.std(),2))

# Cross Validation for Decision Tree
from sklearn.model_selection import cross_val_score
tree_scores = cross_val_score(tree, train_features_prepared, train_labels, scoring="r2", cv=10)

print("Scores:", tree_scores)
print("Mean:", round(tree_scores.mean(),2))
print("Standard deviation:", round(tree_scores.std(),2))

# Find the best model using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [2, 3] , 'max_features': [2, 4,6,7]}
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='r2')
grid_search.fit(train_features_prepared, train_labels)
grid_search.best_params_

# Cross Validation for Decision Tree
tree_best = DecisionTreeRegressor(** grid_search.best_params_)
tree_best.fit(train_features_prepared, train_labels)

# Predictions
tree_bestpredictions = tree_best.predict(train_features_prepared)

# Testing Accuracy
test_features = test_set.drop("median_house_value", axis=1)
test_features['rooms_per_household'] = test_features['total_rooms']/test_features['households']
test_features['bedrooms_per_room'] = test_features['total_bedrooms']/test_features['total_rooms']
test_features['population_per_household'] = test_features['population']/test_features['households']
test_labels = test_set["median_house_value"].copy()

test_features_prepared = full_pipeline.transform(test_features)

tree_best_predictions_test = tree_best.predict(test_features_prepared)

print("Decision Tree Accuracy on Training Data: ", r2_score(train_labels, tree_bestpredictions))
print("Decision Tree Accuracy on Test Data: ", r2_score(test_labels, tree_best_predictions_test))

# Find the best model using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [2, 3, 4, 5] , 'max_features': [2, 4, 6, 8]}
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_features_prepared, train_labels)
grid_search.best_params_

# Cross Validation for Decision Tree
tree5 = DecisionTreeRegressor(max_depth=5, max_features=8)
tree5.fit(train_features_prepared, train_labels)

# Predictions
tree5_predictions = tree5.predict(train_features_prepared)
[ ]
feature_names = full_pipeline._columns[0]+ cat_encoder.categories_[0].tolist()
feature_names.remove('<1H OCEAN')

# Testing Accuracy
test_features = test_set.drop("median_house_value", axis=1)
test_features['rooms_per_household'] = test_features['total_rooms']/test_features['households']
test_features['bedrooms_per_room'] = test_features['total_bedrooms']/test_features['total_rooms']
test_features['population_per_household'] = test_features['population']/test_features['households']
test_labels = test_set["median_house_value"].copy()

test_features_prepared = full_pipeline.transform(test_features)

lin_reg_predictions_test = lin_reg.predict(test_features_prepared)
tree5_predictions_test = tree5.predict(test_features_prepared)

print("Linear Regression Accuracy on Test Data: ", r2_score(test_labels, lin_reg_predictions_test))
print("Decision Tree Accuracy on Test Data: ", r2_score(test_labels, tree5_predictions_test))

# Linear Regression vs. Decision Tree Training and Testing Accuracy
pd.DataFrame({ "Linear Regression": [r2_score(train_labels, lin_reg_predictions), r2_score(test_labels, lin_reg_predictions_test)],
                "Decision Tree": [r2_score(train_labels, tree5_predictions), r2_score(test_labels, tree5_predictions_test)]},
                index=["Training Accuracy", "Testing Accuracy"])

#Save Your Model

# save linear regression model
import joblib
joblib.dump(lin_reg, "lin_reg.pkl")

# Load the model
lin_reg = joblib.load("lin_reg.pkl")

# Use the model to make predictions
lin_reg_predictions_test = lin_reg.predict(test_features_prepared)

# Save Pipeline
import joblib
joblib.dump(full_pipeline, "full_pipeline.pkl")
['full_pipeline.pkl']
#Web App

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
lin_reg = joblib.load("lin_reg.pkl")

# Load the pipeline
full_pipeline = joblib.load("full_pipeline.pkl")

# Load the data
housing = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")

# Create a title and sub-title
st.title("California Housing Price Prediction App")

st.write("""
This app predicts the **California Housing Price**!
""")

# Take the input from the user
longitude = st.slider('longitude', float(housing['longitude'].min()), float(housing['longitude'].max()))
latitude = st.slider('latitude', float(housing['latitude'].min()), float(housing['latitude'].max()))

housing_median_age = st.slider('housing_median_age', float(housing['housing_median_age'].min()), float(housing['housing_median_age'].max()))
total_rooms = st.slider('total_rooms', float(housing['total_rooms'].min()), float(housing['total_rooms'].max()))
total_bedrooms = st.slider('total_bedrooms', float(housing['total_bedrooms'].min()), float(housing['total_bedrooms'].max()))
population = st.slider('population', float(housing['population'].min()), float(housing['population'].max()))
households = st.slider('households', float(housing['households'].min()), float(housing['households'].max()))
median_income = st.slider('median_income', float(housing['median_income'].min()), float(housing['median_income'].max()))

ocean_proximity = st.selectbox('ocean_proximity', ('<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'))

# Store a dictionary into a variable
user_data = {'longitude': longitude,

'latitude': latitude,
'housing_median_age': housing_median_age,
'total_rooms': total_rooms,
'total_bedrooms': total_bedrooms,
'population': population,
'households': households,
'median_income': median_income,
'ocean_proximity': ocean_proximity}

# Transform the data into a data frame
features = pd.DataFrame(user_data, index=[0])

# Additional transformations
features['rooms_per_household'] = features['total_rooms']/features['households']
features['bedrooms_per_room'] = features['total_bedrooms']/features['total_rooms']
features['population_per_household'] = features['population']/features['households']

# Pipeline
features_prepared = full_pipeline.transform(features)

# Predict the output
prediction = lin_reg.predict(features_prepared)[0]

# Set a subheader and display the prediction
st.subheader('Prediction')
st.markdown('''# $ {} '''.format(round(prediction), 2))

!streamlit run app.py
