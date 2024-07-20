import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import math
import random
from keras.models import Sequential
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from keras import layers, models
from keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import Lasso


def remove_outliers_iqr(df, df2):

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df2[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df_filtered

def pop_row(df):

    index_to_pop = random.choice(df.index)
    popped_row = df.loc[index_to_pop]
    df = df.drop(index_to_pop)
    df = df.reset_index(drop=True)

    return popped_row, df


def create_objective(row, time_mean, time_std, puntuacion_maxima, recompensa_top, recompensa_mean, recompensa_std, tiempoMaxStats_std, tiempoMaxStats_mean):

    if row['Aciertos'] / row['Fallos'] < 1:
        row['Aciertos'] = math.ceil(row['Numero de piezas'] * 0.66)
        row['Fallos'] = math.ceil(row['Numero de piezas']) - row['Aciertos']
    elif row['Aciertos'] / row['Fallos'] >= 5:
        row['Aciertos'] = math.ceil(row['Numero de piezas'] * 0.5)
        row['Fallos'] = math.ceil(row['Numero de piezas']) - row['Aciertos']

    if row['Tiempo total de la prueba'] < time_mean - time_std*1.5:
        row['Tiempo total de la prueba'] = time_mean + random.uniform(0, time_std)
    elif row['Tiempo total de la prueba'] > time_mean + time_std*1.5:
        row['Tiempo total de la prueba'] = time_mean - random.uniform(0, time_std)

    if row['Puntuación maxima'] < 0:
        row['Puntuación maxima'] = random.uniform(0, 3)
    elif row['Puntuación maxima'] < puntuacion_maxima:
        row['Puntuación aáxima'] = random.uniform(0, 3)

    if row['Recompensa maxima'] > recompensa_top :
        row['Recompensa maxima'] = random.uniform(recompensa_mean, recompensa_std)

    if row['Tiempo de respuesta maximo'] > tiempoMaxStats_mean + tiempoMaxStats_std*1.5 :
        row['Tiempo de respuesta maximo'] = tiempoMaxStats_mean + random.uniform(0, tiempoMaxStats_std)

    return row


def train_light(df, target):

    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lgb_model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)

    lgb_model.fit(X_train, y_train)
    predictions = lgb_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

"""
def train_neural_grid(df, target):


    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    param_grid = {
        # Define hyperparameters to tune
        'epochs': [10, 20],  # Example values, you can adjust these
        'batch_size': [32, 64]  # Example values, you can adjust these
    }

    # Define the scoring metrics
    scoring = {
        'mae': make_scorer(mean_absolute_error),
        'mse': make_scorer(mean_squared_error),
        'r2': make_scorer(r2_score)
    }

    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=create_model, param_grid=param_grid, scoring=scoring, refit=False, cv=3)
    grid_search.fit(X_train, y_train)

    # Print results
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score (MAE):", grid_search.best_score_)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the best model
    predictions = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
"""

def train_lstm(df, target):
    X = df.drop(target, axis=1).values  # Features
    y = df[target].values  # Target

    # Reshape X for LSTM input [samples, timesteps, features]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = models.Sequential([
        layers.LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        layers.LSTM(32),
        layers.Dense(1)  # Output layer with one neuron for regression
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

    # Evaluate the model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

def train_neural(df, target):


    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer with one neuron for regression
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

    # Evaluate the model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)


def train_neural_with_grid_search(df, target):
    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Output layer with one neuron for regression
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'epochs': [30, 50],
        'batch_size': [32, 64]
    }

    # Define the scoring function for GridSearchCV
    scoring = {'neg_mean_absolute_error': 'neg_mean_absolute_error'}

    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=scoring, refit='neg_mean_absolute_error')

    # Fit the model using GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the best model
    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Best Parameters:", grid_search.best_params_)
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    return best_model

def train_elastic(df, target):

    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # You can adjust alpha and l1_ratio

    # Fit the model on the training data
    elasticnet_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = elasticnet_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)


def train_lasso(df, target):

    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    lasso_model = Lasso(alpha=0.4)  # You can adjust the regularization parameter alpha

    # Fit the model on the training data
    lasso_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = lasso_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)

def train_gbr(df, target):


    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    gbm = GradientBoostingRegressor()

    # Train the model
    gbm.fit(X_train, y_train)

    # Predict on the test set
    y_pred = gbm.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

def train_svm2(df, target):

    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVR(kernel='poly')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Support Vector Machine (SVM) Model Metrics:")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

def train_svm(df, target):

    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale'],
        'degree': [3],
        'epsilon': [0.1]
    }

    # Define the scoring metrics
    scoring = {
        'mae': make_scorer(mean_absolute_error)
    }

    svr = SVR()

    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, scoring=scoring, refit=False, cv=3)
    grid_search.fit(X_train, y_train)

    # Print results
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score (MAE):", grid_search.best_score_)

    return grid_search.best_estimator_

def train_model_xgboost(df, target):

    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'learning_rate': [ 0.005, 0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4],
        'n_estimators': [50, 100, 200],
        'colsample_bytree': [0.5, 0.6, 0.7]
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Score (MSE):", best_score)

    best_model = grid_search.best_estimator_

    # Predictions using the best model
    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    return best_model,

def train_model_random_forest_with_grid_search(df, target):
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Random Forest with Grid Search Model Metrics:")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    print("Best metrics:", grid_search.best_params_)

    return best_model

def train_bayesian(df, target):
    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    bayesian_reg = BayesianRidge()
    bayesian_reg.fit(X_train, y_train)

    y_pred = bayesian_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)

def train_simple(df, target):

    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define degree of polynomial
    degree = 2  # Example degree, you can change this

    # Generate polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Instantiate Lasso regression model
    lasso_reg_model = Lasso(alpha=0.1)  # You can adjust alpha as needed

    # Train the model
    lasso_reg_model.fit(X_train_poly, y_train)

    # Make predictions on the test set
    y_pred = lasso_reg_model.predict(X_test_poly)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Root Mean Squared Error (RMSE):", rmse)

def train_model_1(df, target):


    X = df.drop(target, axis=1)  # Features
    y = df[target]  # Target

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer with one neuron for regression
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

    # Evaluate the model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    return model

def train_model_2(df, target  ):
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf_regressor = RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200)

    rf_regressor.fit(X_train, y_train)

    predictions = rf_regressor.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Random Forest with Grid Search Model Metrics:")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    return rf_regressor



target_columns = ["Tamaño de los objetos.", "Ratio de recompensa de los objetos.", "Ratio de aparición de los objetos en el juego.", "Distancia al jugador de los objetos."]


df = pd.read_csv("datasetIA.csv", sep = ";")
df = df.apply(pd.to_numeric, errors='coerce').dropna()

sample_row, df = pop_row(df)

features = df.drop(target_columns, axis=1)

df = remove_outliers_iqr(features, df)

corr_matrix = df.corr().abs()

df['Proporcion'] = df['Aciertos'] / df['Fallos']

max_value = df['Proporcion'].replace([np.inf, -np.inf], np.nan).max()
df['Proporcion'].replace([np.inf, -np.inf], max_value, inplace=True)

relevant_columns1 = corr_matrix[corr_matrix[target_columns[0]].abs() > 0.1][target_columns[0]].index.tolist()
df_1 = df[relevant_columns1]

relevant_columns2 = corr_matrix[corr_matrix[target_columns[1]].abs() > 0.25][target_columns[1]].index.tolist()
df_2 = df[relevant_columns2]

relevant_columns3 = corr_matrix[corr_matrix[target_columns[2]].abs() > 0.25][target_columns[2]].index.tolist()
df_3 = df[relevant_columns3]

relevant_columns4 = corr_matrix[corr_matrix[target_columns[3]].abs() > 0.1][target_columns[3]].index.tolist()
df_4 = df[relevant_columns4]

timeStats = df['Tiempo total de la prueba'].describe()
time_std = timeStats['std']
time_mean = timeStats['mean']

puntuacion_top = df['Puntuación maxima'].quantile(0.9)

reconpensa_top = df['Recompensa maxima'].quantile(0.80)
recompensaStats = df['Recompensa maxima'].describe()
recompensa_mean = recompensaStats['mean']
recompensa_std = recompensaStats['std']

tiempoMaxStats = df['Tiempo de respuesta maximo'].describe()
tiempoMaxStats_mean = tiempoMaxStats['mean']
tiempoMaxStats_std = tiempoMaxStats['std']




scaler1 = MinMaxScaler()
features_scaled_1 = pd.DataFrame(scaler1.fit_transform(df_1.drop(target_columns[0], axis=1)))
df_scaled_1 = pd.concat([features_scaled_1, df_1[target_columns[0]].reset_index(drop=True)], axis=1)

scaler2 = MinMaxScaler()
features_scaled_2 = pd.DataFrame(scaler2.fit_transform(df_2.drop(target_columns[1], axis=1)))
df_scaled_2 = pd.concat([features_scaled_2, df_2[target_columns[1]].reset_index(drop=True)], axis=1)

scaler3 = MinMaxScaler()
features_scaled_3 = pd.DataFrame(scaler3.fit_transform(df_3.drop(target_columns[2], axis=1)))
df_scaled_3 = pd.concat([features_scaled_3, df_3[target_columns[2]].reset_index(drop=True)], axis=1)

scaler4 = MinMaxScaler()
features_scaled_4 = pd.DataFrame(scaler4.fit_transform(df_4.drop(target_columns[3], axis=1)))
df_scaled_4 = pd.concat([features_scaled_4, df_4[target_columns[3]].reset_index(drop=True)], axis=1)


model2 = train_model_2(df_scaled_2, target_columns[1])



corrected_row = create_objective(sample_row, time_mean, time_std, puntuacion_top, reconpensa_top, recompensa_mean, recompensa_std, tiempoMaxStats_mean, tiempoMaxStats_std)

target1 = corrected_row[target_columns[0]]
target2 = corrected_row[target_columns[1]]
target3 = corrected_row[target_columns[2]]
target4 = corrected_row[target_columns[3]]


corrected_row_1 = corrected_row[relevant_columns1]
corrected_row_2 = corrected_row[relevant_columns2]
corrected_row_3 = corrected_row[relevant_columns3]
corrected_row_4 = corrected_row[relevant_columns4]

corrected_row_1 = corrected_row_1.drop(target_columns[0])
corrected_row_2 = corrected_row_2.drop(target_columns[1])
corrected_row_3 = corrected_row_3.drop(target_columns[2])
corrected_row_4 = corrected_row_4.drop(target_columns[3])


corrected_row_1_df = corrected_row_1.to_frame().transpose()
corrected_row_2_df = corrected_row_2.to_frame().transpose()
corrected_row_3_df = corrected_row_3.to_frame().transpose()
corrected_row_4_df = corrected_row_4.to_frame().transpose()


corrected_row_1_df = scaler1.transform(corrected_row_1_df)
corrected_row_2_df = scaler2.transform(corrected_row_2_df)
corrected_row_3_df = scaler3.transform(corrected_row_3_df)
corrected_row_4_df = scaler4.transform(corrected_row_4_df)


target2_predicted = model2.predict(corrected_row_2_df)


print("T: ", target2)
print("P: ", target2_predicted)


print("test end")