#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sklearn
import numpy as np
import pandas as pd
import os
DATASET_PATH = '/home/shaikhkamil0337982/BootML/Datasets/finalprojectbml_1/'
def load_dataset_data(dataset_path=DATASET_PATH): 
    csv_path = os.path.join(dataset_path, "used_car_dataset.csv") 
    return pd.read_csv(csv_path)


# 

# In[4]:


dataset = load_dataset_data()
dataset.head()


# In[5]:


dataset.info()


# In[6]:


# Count missing values in 'kmDriven'
missing_values = dataset.kmDriven.isna().sum()

# Display the result
print(f'The kmDriven feature is missing {missing_values} values')


# In[7]:


# Show rows where column kmDriven has missing values
rows_with_missing = dataset[dataset['kmDriven'].isna()]

# Display the result
print(rows_with_missing)


# In[8]:


dataset.describe()


# In[9]:


dataset.shape


# In[10]:


#  Clean up the kmDriven text
dataset["kmDriven"] = (
    dataset["kmDriven"]
    .str.replace(",", "", regex=False)
    .str.replace(" km", "", regex=False)
    .astype(float)  # Convert to float
)

# Drop rows with NaN in kmDriven
dataset = dataset.dropna(subset=["kmDriven"])

# Convert to integer
dataset["kmDriven"] = dataset["kmDriven"].astype(int)


# In[11]:


dataset["AskPrice"] = (
    dataset["AskPrice"]
    .str.replace(",", "", regex=False)
    .str.replace("₹", "", regex=False)
    .astype(float)  # Convert to float
)

# Convert to integer
dataset["AskPrice"] = dataset["AskPrice"].astype(int)


# In[12]:


dataset.info()


# In[13]:


unique_counts = dataset.nunique(dropna=False)

# Display the result
print(unique_counts)


# In[14]:


dataset = dataset.drop(columns=['AdditionInfo', 'PostedDate', 'Age'])


# In[15]:


dataset.info()


# In[16]:


dataset.head()


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

# Group by transmission type and calculate average price
price_trends_by_transmission = (
    dataset.groupby('Transmission')['AskPrice']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

# Create a bar plot for price trends by transmission
plt.figure(figsize=(8, 5))
sns.barplot(
    data=price_trends_by_transmission, 
    x='Transmission', 
    y='AskPrice'
)
plt.title('Average Ask Prices by Transmission Type', fontsize=14)
plt.xlabel('Transmission Type', fontsize=12)
plt.ylabel('Average Ask Price (₹)', fontsize=12)

plt.show()


# In[18]:


# Group by owner type and calculate average price
price_trends_by_owner = (
    dataset.groupby('Owner')['AskPrice']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

# Create a bar plot for price trends by owner
plt.figure(figsize=(8, 5))
sns.barplot(
    data=price_trends_by_owner, 
    x='Owner', 
    y='AskPrice'
)
plt.title('Average Ask Prices by Owner Type', fontsize=14)
plt.xlabel('Owner Type', fontsize=12)
plt.ylabel('Average Ask Price (₹)', fontsize=12)

plt.show()


# In[19]:


price_trends_by_fuel_type = (
    dataset.groupby('FuelType')['AskPrice']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

# Create a bar plot for price trends by fuel type
plt.figure(figsize=(8, 5))
sns.barplot(
    data=price_trends_by_fuel_type, 
    x='FuelType', 
    y='AskPrice'
)
plt.title('Average Ask Prices by Fuel Type', fontsize=14)
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('Average Ask Price (₹)', fontsize=12)

plt.show()


# In[20]:


# Calculate the average asking price for each brand
brand_avg_price_summary = dataset.groupby('Brand')['AskPrice'].mean().reset_index()

# Sort the data for better visualization
brand_avg_price_summary = brand_avg_price_summary.sort_values(by='AskPrice', ascending=False)

# Create a seaborn bar plot for the average asking price by brand
plt.figure(figsize=(12, 6))
sns.barplot(data=brand_avg_price_summary, x='Brand', y='AskPrice')
plt.title('Average Asking Price for Each Brand')
plt.xlabel('Brand')
plt.ylabel('Average Asking Price (₹)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

# Scatterplot to identify outliers in 'kmDriven' vs. 'AskPrice'
plt.figure(figsize=(10, 6))
sns.scatterplot(x=dataset['kmDriven'], y=dataset['AskPrice'])
plt.title('Scatterplot of kmDriven vs. AskPrice')
plt.show()


# In[22]:


from scipy.stats import zscore

# Calculate Z-Scores
dataset['Z-Score'] = zscore(dataset['AskPrice'])

# Filter out outliers with |Z-Score| > 3
dataset = dataset[np.abs(dataset['Z-Score']) <= 3]

# Drop the 'Z-Score' column.
dataset = dataset.drop(columns=['Z-Score'])


# In[23]:


dataset.shape


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# Scatterplot to identify outliers in 'kmDriven' vs. 'AskPrice'
plt.figure(figsize=(10, 6))
sns.scatterplot(x=dataset['kmDriven'], y=dataset['AskPrice'])
plt.title('Scatterplot of kmDriven vs. AskPrice')
plt.show()


# In[ ]:


dataset.info()


# In[ ]:


print(f'Percentage of rows dropped as Outliers: {100 * (9535 - 9373) / 9535:.2f}%')


# In[26]:


# Regression plot with a linear regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=dataset, x='Year', y='AskPrice', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.title('Regression Plot of Car Year vs Asking Price')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Asking Price', fontsize=12)
plt.show()


# In[ ]:


dataset.info()


# In[28]:


# Select numerical features for pairplot
numerical_features = dataset.select_dtypes(include=['int64', 'float64']).columns

# Pairplot of numerical features including the target
sns.pairplot(dataset, vars=numerical_features, y_vars=['AskPrice'], height=3, aspect=1.2)
plt.show()


# In[30]:


dataset.info()


# In[31]:


numerical_features = ['Year', 'kmDriven', 'AskPrice', 'model_encoded',
                      'Brand_encoded', 'km_per_year', 'brand_model_interaction']  
categorical_features = ['Transmission', 'FuelType', 'Owner']


# In[33]:


X = dataset.drop(columns=['AskPrice'])  # Replace 'AskPrice' with your target column
y = dataset['AskPrice']  # Replace 'AskPrice' with your target column


# In[34]:


from sklearn.model_selection import train_test_split
# Split the dataset into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=0)


# In[35]:


columns = X_train.columns


# In[40]:


from sklearn.preprocessing import OneHotEncoder

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the training data
X_train_encoded = encoder.fit_transform(X_train[categorical_features])

# Get the actual column names from the encoder
columns = encoder.get_feature_names(categorical_features)

# Convert the encoded data back to a DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns=columns)

# Repeat for the test data
X_test_encoded = encoder.transform(X_test[categorical_features])
X_test_encoded = pd.DataFrame(X_test_encoded, columns=columns)


# In[41]:


X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# In[42]:


X_train = pd.concat([X_train, X_train_encoded], axis=1, join='outer')
X_test = pd.concat([X_test, X_test_encoded], axis=1, join='outer')


# In[44]:


X_train.drop(columns=['Transmission', 'Owner', 'FuelType'], inplace=True)
X_test.drop(columns=['Transmission', 'Owner', 'FuelType'], inplace=True)


# In[45]:


X_train.info()


# In[46]:


# Define a function for target encoding with smoothing
def target_encode(train, test, feature, target, smoothing=10):
    """
    Performs target encoding on a categorical feature with smoothing.
    
    Args:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Testing data.
        feature (str): Categorical feature to be target encoded.
        target (str): Target variable in the training data.
        smoothing (int): Smoothing factor to balance category mean with global mean.
    
    Returns:
        pd.Series, pd.Series: The target encoded feature for training and testing.
    """
    # Calculate global mean of the target
    global_mean = train[target].mean()
    
    # Calculate the mean and count for each category
    agg = train.groupby(feature)[target].agg(['mean', 'count'])
    mean = agg['mean']
    count = agg['count']
    
    # Apply smoothing formula
    smoothed_mean = (count * mean + smoothing * global_mean) / (count + smoothing)
    
    # Map the smoothed means to training and testing sets
    train_encoded = train[feature].map(smoothed_mean)
    test_encoded = test[feature].map(smoothed_mean)
    
    # Fill NaN in test_encoded with the global mean
    test_encoded = test_encoded.fillna(global_mean)
    
    return train_encoded, test_encoded

# Ensure 'AskPrice' is numeric in the training data
X_train['AskPrice'] = y_train  # Add target column temporarily to X_train if not included

# Apply target encoding to the 'model' feature
X_train['model_encoded'], X_test['model_encoded'] = target_encode(
    train=X_train, 
    test=X_test, 
    feature='model', 
    target='AskPrice', 
    smoothing=10
)

# Apply target encoding to the 'Brand' feature
X_train['Brand_encoded'], X_test['Brand_encoded'] = target_encode(
    train=X_train, 
    test=X_test, 
    feature='Brand', 
    target='AskPrice', 
    smoothing=10
)

# Drop the original 'model' and 'Brand' columns
X_train = X_train.drop(columns=['model', 'Brand', 'AskPrice'])
X_test = X_test.drop(columns=['model', 'Brand'])


# In[47]:


# Ensure 'AskPrice' is numeric in the training data if not already
X_train['AskPrice'] = y_train

# Create interaction features for training data
X_train['km_per_year'] = X_train['kmDriven'] / (2025 - X_train['Year'])
X_train['brand_model_interaction'] = X_train['Brand_encoded'] * X_train['model_encoded']

# Create interaction features for test data
X_test['km_per_year'] = X_test['kmDriven'] / (2025 - X_test['Year'])
X_test['brand_model_interaction'] = X_test['Brand_encoded'] * X_test['model_encoded']

# Remove 'AskPrice' from training data after feature creation if it was added temporarily
X_train = X_train.drop(columns=['AskPrice'])

# Check correlation with the target in the training data
interaction_features = ['km_per_year', 'brand_model_interaction']
correlation = pd.concat([X_train[interaction_features], y_train], axis=1).corr()
print(correlation)


# In[48]:


# Combine X_train and y_train
train_data = pd.concat([X_train, y_train], axis=1)

# Plot a pairplot
sns.pairplot(train_data, diag_kind='kde')
plt.show()


# In[49]:


from sklearn.feature_selection import mutual_info_regression
mutual_info = mutual_info_regression(X_train, y_train)

# Convert results to a Pandas Series
mutual_info_series = pd.Series(mutual_info, index=X_train.columns)
mutual_info_series.sort_values(ascending=False, inplace=True)


# In[50]:


# Plot mutual information scores
plt.figure(figsize=(10, 6))
mutual_info_series.plot(kind='bar')
plt.title('Mutual Information Scores')
plt.xlabel('Features')
plt.ylabel('Mutual Information')
plt.show()


# In[51]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training set and transform both the training and test sets
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)       # Transform test data

# Convert back to DataFrame to preserve column names and index
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


# In[52]:


# Replace 'your_dataframe' with the name of your DataFrame
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)


# In[53]:


from sklearn.model_selection import KFold

# Define a shared KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=0)


# In[54]:


#Decision Tree Model
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Initialize the Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=0)

# Define RMSE as the evaluation metric
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# Perform cross-validation
cv_scores = cross_val_score(
    estimator=dt_model,
    X=X_train,
    y=y_train,
    scoring=rmse_scorer,
    cv=kf  # Use the shared KFold object
)

# Fit the model on the entire training set
dt_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred_test = dt_model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

# Display results
print(f"Cross-Validation RMSE Scores: {cv_scores}")
print(f"Mean RMSE (Cross-Validation): {cv_scores.mean():.3f}")
print(f"Standard Deviation of RMSE (Cross-Validation): {cv_scores.std():.3f}")
print(f"Test RMSE: {test_rmse:.3f}")


# In[56]:


#Random Forest Regressor Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize the RandomForestRegressor with default parameters
rf = RandomForestRegressor(random_state=0)

# Train the model on the training data
rf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = rf.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)

# Display results
print(f"RMSE: {test_rmse:.3f}")


# In[59]:


pip install catboost


# In[57]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

# Define the RandomForestRegressor
rf = RandomForestRegressor(random_state=0)

# Define hyperparameter distribution
param_dist = {
    'n_estimators': [10, 25, 50, 100],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1.0, 'sqrt', 'log2']
}

# param_dist = {
#     'n_estimators': [100],
#     'max_depth': [20],
#     'min_samples_split': [2],
#     'min_samples_leaf': [1],
#     'max_features': [1.0]
# }

# Define the scoring metric
scoring = 'neg_mean_squared_error'  # Directly use sklearn's predefined scoring metric
rmse_scorer = make_scorer(mean_squared_error, squared=False)  # Custom RMSE scorer

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_dist,  # Pass the grid of hyperparameters
    scoring=scoring,
    cv=kf,  # Use the shared KFold
    verbose=0,
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Extract the best parameters
best_params = grid_search.best_params_

# Check if the score is valid and convert to RMSE
best_cv_neg_mse = grid_search.best_score_  # Best score (negative MSE)
if best_cv_neg_mse < 0:  # Ensure it is negative
    best_cv_rmse = np.sqrt(-best_cv_neg_mse)  # Convert to RMSE
else:
    raise ValueError("Unexpected positive or NaN best_score_ in GridSearchCV.")

# Train a new model with the best parameters on the full training set
best_model = RandomForestRegressor(**best_params, random_state=0)
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)

# Display results
print("Best Parameters:", best_params)
print(f"Best Cross-Validation RMSE: {best_cv_rmse:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")


# In[60]:


from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Define the model with default hyperparameters
cat_model = CatBoostRegressor(random_seed=0, verbose=0)

# Train the model on the training data
cat_model.fit(X_train, y_train)

# Predict on the test set
y_pred = cat_model.predict(X_test)

# Compute RMSE on the test set
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Display results
print(f"RMSE: {rmse:.3f}")


# In[61]:


from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Define the model
cat_model = CatBoostRegressor(
    random_seed=0,
    verbose=0
)

# Define the parameter grid
param_grid = {
    'iterations': [500, 1000],
    'learning_rate': [0.01, 0.03, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'bagging_temperature': [0, 0.5, 1]
}

# param_grid = {
#     'iterations': [1000],
#     'learning_rate': [0.1],
#     'depth': [6],
#     'l2_leaf_reg': [1],
#     'bagging_temperature': [0]
# }

# Perform Grid Search
grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=kf,
    verbose=0,
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Display best parameters and score
print("Best Parameters:", grid_search.best_params_)
best_rmse = (-grid_search.best_score_) ** 0.5
print(f"Best RMSE: {best_rmse:.3f}")

# Evaluate the best model on a test set
y_pred = grid_search.best_estimator_.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test RMSE: {rmse:.3f}")


# In[ ]:




