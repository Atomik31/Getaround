import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from dotenv import load_dotenv

import time

load_dotenv()

# Loading dataset
df = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_pricing_project.csv", index_col=0)

# Dropping rows with anomaly
df = df[(df['mileage'] >= 0) & (df['engine_power'] > 0)]

# Splitting dataset into X features and Target variable
target = 'rental_price_per_day'
Y = df[target]
X = df.drop(target, axis = 1)

# categorizing features
numeric_features = []
categorical_features = []
for i,t in X.dtypes.items():
    if ('float' in str(t)) or ('int' in str(t)) :
        numeric_features.append(i)
    else :
        categorical_features.append(i)

print('Found numeric features ', numeric_features)
print('Found categorical features ', categorical_features)

# Split our training set and our test set 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Features preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocessings on train set
print("Performing preprocessings on train set...")
print(X_train.head())
X_train = preprocessor.fit_transform(X_train)
print('...Done.')
print(X_train[0:5]) 
print()

# Preprocessings on test set
print("Performing preprocessings on test set...")
print(X_test.head()) 
X_test = preprocessor.transform(X_test) 
print('...Done.')
print(X_test[0:5,:])

# Set your variables for your environment
EXPERIMENT_NAME="getaround-pricing-v2"

# Set tracking URI to your Heroku application
os.environ["APP_URI"]="https://atomik31-mlflow-cdsd.hf.space"
mlflow.set_tracking_uri(os.environ["APP_URI"])

# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

# Time execution
start_time = time.time()

# Call mlflow autolog (toujours utile pour loguer les hyperparamètres automatiquement)
mlflow.sklearn.autolog(disable=True)


print("Linear Regression Training ...")
run_name = 'linear_regression'
with mlflow.start_run(run_name=run_name) as run:
    model_lr = LinearRegression()
    model_lr.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model_lr.predict(X_train)
    Y_test_pred = model_lr.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",root_mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",root_mean_squared_error(Y_test, Y_test_pred))
    mlflow.end_run()

print("Random Forest Training ...")
run_name = 'random_forest'
with mlflow.start_run(run_name=run_name) as run:
    model_rf = RandomForestRegressor(max_depth=10)
    model_rf.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model_rf.predict(X_train)
    Y_test_pred = model_rf.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",root_mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",root_mean_squared_error(Y_test, Y_test_pred))
    mlflow.end_run()

print("Ridge Training ...")
run_name = 'ridge'
with mlflow.start_run(run_name=run_name) as run:
    model = Ridge(alpha=1)
    model.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",root_mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",root_mean_squared_error(Y_test, Y_test_pred))
    mlflow.end_run()

print("Lasso Training ...")
run_name = 'lasso'
with mlflow.start_run(run_name=run_name) as run:
    model_lasso =  Lasso(alpha=1)
    model_lasso.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model_lasso.predict(X_train)
    Y_test_pred = model_lasso.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",root_mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",root_mean_squared_error(Y_test, Y_test_pred))
    mlflow.end_run()

print("GridSearchCV RandomForest Training ...")
run_name = 'random_forest_gridsearch'
with mlflow.start_run(run_name=run_name) as run:
    params_rf = {
    'max_depth': [16, 18, 20],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [150, 200, 250]
    }
    rf = RandomForestRegressor()
    model_gridrf = GridSearchCV(rf, params_rf, cv=5, verbose=True, n_jobs=-1)
    model_gridrf.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model_gridrf.predict(X_train)
    Y_test_pred = model_gridrf.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",root_mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",root_mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_param("best_params", str(model_gridrf.best_params_))
    mlflow.end_run()

print("XGBRegressor Training ...")
run_name = 'xgbr'
with mlflow.start_run(run_name=run_name) as run:
    model_xgb = XGBRegressor(n_estimators=200, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, alpha=0.1, random_state=42)
    model_xgb.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model_xgb.predict(X_train)
    Y_test_pred = model_xgb.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",root_mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",root_mean_squared_error(Y_test, Y_test_pred))
    mlflow.end_run()

print("All training is done!")
print(f"---Total training time: {time.time()-start_time}")

# Enregistrement du meilleur modèle dans le MLflow Model Registry
print("\nEnregistrement du meilleur modèle dans le Model Registry...")

from sklearn.pipeline import Pipeline as SKPipeline
from mlflow.models.signature import infer_signature

# On encapsule preprocessor + GridSearchCV RF dans un pipeline sklearn
best_pipeline = SKPipeline([
    ("preprocessor", preprocessor),
    ("model", model_gridrf.best_estimator_)
])

# Reconstruction sur données brutes (avant transform)
X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(X, Y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="production_model"):
    best_pipeline.fit(X_train_raw, Y_train_raw)
    Y_pred = best_pipeline.predict(X_test_raw)

    mlflow.log_param("model_type", "RandomForestRegressor (GridSearchCV)")
    mlflow.log_param("best_params", str(model_gridrf.best_params_))
    mlflow.log_metric("testing_r2_score", r2_score(Y_test_raw, Y_pred))
    mlflow.log_metric("testing_mae", mean_absolute_error(Y_test_raw, Y_pred))

    signature = infer_signature(X_train_raw, best_pipeline.predict(X_train_raw))

    mlflow.sklearn.log_model(
        best_pipeline,
        name="getaround_pricing_model",
        registered_model_name="GetAround_price_predictor",
        signature=signature,
        input_example=X_train_raw.iloc[:3]
    )

# Promouvoir la dernière version en "production"
from mlflow.tracking import MlflowClient
client = MlflowClient()
model_versions = client.get_registered_model("GetAround_price_predictor").latest_versions
latest_version = model_versions[-1].version
client.set_registered_model_alias("GetAround_price_predictor", "production", latest_version)

print(f"Modèle version {latest_version} enregistré et promu en 'production'")